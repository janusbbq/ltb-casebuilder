"""``case_build`` route node: build a period-specific (month / year) case and make it active.

Registered with LLM_LTB as a route extension (see ``casebuilder.ltb_plugin``).

Division of labour:

* the **LLM** turns the request into a :class:`PeriodCaseSpec` (structured output);
* a **deterministic parser** re-reads the year / month / day count / region / base case from
  the text and overrides the LLM wherever it finds them, so a mis-extracted date can never
  silently build the wrong period (and the node still works if the LLM call fails);
* ``casebuilder.period_case`` does all numerics, validation and file writing.

After a successful build the new xlsx is loaded and, if the active routine is single-period
(RTED / DCOPF / ...), the routine is switched to ED so that a following "solve" actually uses
the hourly horizon.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agent.ams_engine.case_catalog import resolve_case
from agent.schemas.response import NodeResponse
from agent.state.app_state import State
from agent.utils.display import display_executing_node
from casebuilder.period_case import (
    REGION_ALIASES,
    REGIONS,
    PeriodCaseError,
    _uced_sibling,
    artifact_summary,
    available_periods,
    build_period_case,
)
from casebuilder.schema import PeriodCaseSpec

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
           "十": 10, "十一": 11, "十二": 12}
_COUNT_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "ten": 10, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
                "八": 8, "九": 9, "十": 10}
_YEAR = r"(?<!\d)(20\d{2})(?!\d)"


def _count(tok: str) -> int:
    return int(tok) if tok.isdigit() else _COUNT_WORDS[tok]


def _is_month_may(t: str, m: "re.Match") -> bool:
    """'may' is the month only next to a year or after in/for/of/during."""
    before, after = t[:m.start()], t[m.end():]
    return bool(re.match(r"\s*,?\s*20\d{2}", after) or re.search(r"20\d{2}\s*$", before)
                or re.search(r"\b(?:in|for|of|during)\s+$", before))
_MONTH_NAME = r"(?<![a-z])(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")(?![a-z])"
# Explicit case mentions only. The bare bus-count resolver is NOT run on the whole message:
# "first 5 days" or "14 days" would otherwise resolve to the 5-bus / 14-bus case.
_CASE_RE = re.compile(
    r"(ieee[\s_-]*\d{2,3}(?:[\s_]+uced(?:[\s_]+(?:esd1|pvd1|vis))?)?|pjm[\s_-]*5?[\s_-]*bus(?:[\s_]+demo)?|pjm"
    r"|npcc(?:[\s_]+uced)?|wecc(?:[\s_]+uced)?|\d{1,4}[\s-]*(?:bus|节点)|[\w./\\-]+\.xlsx)", re.IGNORECASE)


def parse_period(text: str) -> Dict[str, object]:
    """Deterministically read period fields from free text (English or Chinese).

    Returns only the keys it is sure about: ``year``, ``month``, ``days``, ``start_day``,
    ``region``, ``timezone``.
    """
    t = (text or "").lower()
    out: Dict[str, object] = {}

    m = (re.search(_YEAR + r"\s*[-/.年]\s*(\d{1,2})(?!\d)(?!\s*(?:days?|天|日))", t)       # 2020-07, 2020年7月
         or None)
    if m and 1 <= int(m.group(2)) <= 12:
        out["year"], out["month"] = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?<!\d)(\d{1,2})\s*[-/.]\s*" + _YEAR, t)                           # 07/2020
        if m and 1 <= int(m.group(1)) <= 12:
            out["month"], out["year"] = int(m.group(1)), int(m.group(2))
    if "year" not in out:
        y = re.search(_YEAR, t)
        if y:
            out["year"] = int(y.group(1))
    if "month" not in out:
        mn = None
        for cand in re.finditer(_MONTH_NAME, t):
            if cand.group(1) == "may" and not _is_month_may(t, cand):
                continue                     # modal verb ("you may build ..."), not the month
            mn = cand
            break
        if mn:
            out["month"] = _MONTHS[mn.group(1)]
        else:
            mc = re.search(r"(?<!\d)(\d{1,2})\s*月", t) or re.search(r"(十[一二]?|[一二三四五六七八九])\s*月", t)
            if mc:
                g = mc.group(1)
                val = int(g) if g.isdigit() else _CN_NUM.get(g)
                if val and 1 <= val <= 12:
                    out["month"] = val

    num = r"(\d{1,3}|" + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True)) + r")"
    d = re.search(r"(?<![\d第])" + num + r"\s*(?:-\s*)?(?:days?\b|天)", t)
    w = re.search(r"(?<![\d第])" + num + r"\s*(?:-\s*)?(?:weeks?\b|个?星期|周)", t)
    if d:
        out["days"] = _count(d.group(1))
    elif w:
        out["days"] = 7 * _count(w.group(1))
    elif re.search(r"\bfirst\s+week\b|第一周|首周", t):
        out["days"] = 7
    elif re.search(r"\bfirst\s+day\b|第一天|首日", t):
        out["days"] = 1

    sd = re.search(r"(?:from|starting|start(?:ing)?\s+on)\s+(?:day\s+)?(\d{1,2})(?:st|nd|rd|th)?(?!\d)(?!\s*(?:days?|weeks?))", t) \
        or re.search(r"从\s*(\d{1,2})\s*[日号]", t)
    if sd and 1 <= int(sd.group(1)) <= 31:
        out["start_day"] = int(sd.group(1))

    for name, code in REGION_ALIASES.items():
        if name in t:
            out["region"] = code
            break
    if re.search(r"\butc\b", t):
        out["timezone"] = "utc"
    return out


def parse_base_case(text: str) -> Optional[str]:
    """Case catalog key for a case the user named explicitly, else ``None``."""
    m = _CASE_RE.search(text or "")
    return _case_key(m.group(1)) if m else None


def _named_cases(text: str) -> set:
    """Catalog keys of every case named in ``text``."""
    keys = (_case_key(m.group(1)) for m in _CASE_RE.finditer(text or ""))
    return {k for k in keys if k}


def _named_regions(text: str) -> set:
    t = (text or "").lower()
    return {code for name, code in REGION_ALIASES.items() if name in t}


def _case_key(token: str) -> Optional[str]:
    if token.lower().endswith(".xlsx"):
        return token
    res = resolve_case(re.sub(r"节点", " bus", token))
    if res.entry is None:
        return None
    key = res.entry.key
    if "uced" not in key and key != "pjm5bus":
        # A period case needs ED/UC slot tables; the plain case has none, its *_uced twin does.
        key = _uced_sibling(key) or key
    return key


def _number_in_text(value, text: str) -> bool:
    """True if ``value`` (e.g. 9000.0 or 2019) is written in ``text`` ("9000", "9,000", "9000.0")."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            if abs(float(tok.replace(",", "")) - v) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def _extract_spec(text: str, llm, prompts, inputs,
                  user_text: Optional[str] = None) -> Optional[PeriodCaseSpec]:
    """LLM proposal overridden by whatever the deterministic parser found.

    ``user_text``: in a planned turn ``text`` is the planner's step, the user's own message is
    ``user_text``. The period comes from the step (a step may name one of several periods);
    optional fields must be stated by the user.
    """
    said = text if user_text is None else user_text
    fields: Dict[str, object] = {}
    try:
        extractor = llm.with_structured_output(PeriodCaseSpec)
        proposal = extractor.invoke([
            {"role": "system", "content": prompts["case_build_extract"]["system"].format(
                case_path=inputs.case_path, regions=", ".join(sorted(REGIONS)))},
            {"role": "user", "content": text},
        ])
        if isinstance(proposal, PeriodCaseSpec):
            fields = proposal.model_dump(exclude_unset=True)
        elif isinstance(proposal, dict):
            fields = dict(proposal)
    except Exception:
        fields = {}          # fall back to the parser alone; a missing year is reported below

    parsed = parse_period(text)
    # The parser found a date in the text -> the LLM's month/days must not survive unless the
    # parser confirms them (prevents "2020" + hallucinated month).
    if "year" in parsed:
        for k in ("year", "month", "days", "start_day"):
            fields.pop(k, None)
    # Optional fields count only when the user actually said them. With llama3.1:8b the
    # extractor filled timezone="utc" for "生成2020年7月的ieee39算例" (no UTC in the text), which
    # silently sliced the period in UTC; it can also copy the session's case_path into base_case.
    for k in ("timezone", "region", "base_case"):
        fields.pop(k, None)                        # re-added below from the parser if stated
    for k in ("peak_mw", "ref_year"):
        if fields.get(k) is not None and not (_number_in_text(fields[k], text or "")
                                              and _number_in_text(fields[k], said or "")):
            fields.pop(k)
    # A planner step can carry words the user never said: with llama3.1:8b the step for "Build
    # the IEEE39 case for July 2020 and run ED" read "... based on the 5bus/pjm5bus_demo.xlsx
    # base case, for the PJM region" (the session's case). Keep an optional field from the step
    # only if the user said it; if the step dropped it, take the user's when it is unambiguous.
    if said is not text:
        stated = parse_period(said)
        if "timezone" in parsed and "timezone" not in stated:
            parsed.pop("timezone")
        elif "timezone" in stated:
            parsed["timezone"] = stated["timezone"]
        regions = _named_regions(said)
        if parsed.get("region") not in regions:
            parsed.pop("region", None)
            if len(regions) == 1:
                parsed["region"] = next(iter(regions))
    fields.update(parsed)
    named = parse_base_case(text)
    if said is not text:
        cases = _named_cases(said)
        if named not in cases:
            named = next(iter(cases)) if len(cases) == 1 else None
    if named:
        fields["base_case"] = named
    if "year" not in fields:
        return None
    return PeriodCaseSpec(**fields)


def _user_text(state: State) -> Optional[str]:
    """The user's own message of a planned turn, else ``None``.

    LLM_LTB's step controller appends each step's ``content`` as a HumanMessage, so the user's
    message is the HumanMessage just before the one carrying step 1.
    """
    plan = state.get("plan")
    if not state.get("is_compound") or plan is None or not getattr(plan, "steps", None):
        return None
    first = plan.steps[0].content
    humans = [m for m in state.get("messages") or [] if isinstance(m, HumanMessage)]
    for i in range(len(humans) - 1, 0, -1):
        if humans[i].content == first:
            return humans[i - 1].content
    return None


def _reply(content: str, success: bool, data: dict):
    return {"messages": [AIMessage(content=content)],
            "node_response": NodeResponse(node_type="case_build", success=success, data=data,
                                          message=content, timestamp=datetime.now())}


def case_build_agent(state: State, llm, prompts, ams_ctx):
    display_executing_node("case_build")

    text = state["messages"][-1].content
    inputs = state["inputs"]

    try:
        spec = _extract_spec(text, llm, prompts, inputs, _user_text(state))
    except Exception as exc:                       # pydantic validation of the merged fields
        return _reply(f"I could not turn that into a valid period request: {exc}", False,
                      {"action": "invalid_spec"})
    if spec is None:
        try:
            cov = available_periods("NE")
            hint = (f" The New England record covers {cov['first'][:10]} to {cov['last'][:10]} "
                    f"(complete years: {cov['complete_years']}).")
        except PeriodCaseError as exc:
            hint = f" ({exc})"
        return _reply("Which period should I build? Give a year and optionally a month, e.g. "
                      "'build the IEEE39 case for July 2020' or '生成2020年1月的ieee39算例'." + hint,
                      False, {"action": "need_period"})

    try:
        art = build_period_case(spec)
    except PeriodCaseError as exc:
        # Expected, user-facing refusals: answer directly, do not route to the LLM error handler.
        return _reply(f"I did not build the case. {exc}", False,
                      {"action": "refused", "spec": spec.model_dump()})
    except Exception as exc:
        return {"error_info": {"error_type": "case_build_error", "error_message": str(exc),
                               "user_input": text, "current_inputs": inputs.model_dump()},
                "failed_node": "case_build"}

    # Make the built case the active one.
    try:
        info = ams_ctx.load_case(art.xlsx_path)
    except Exception as exc:
        return {"error_info": {"error_type": "case_load_error", "error_message": str(exc),
                               "user_input": text, "current_inputs": inputs.model_dump()},
                "failed_node": "case_build"}

    routine = inputs.routine
    routine_note = ""
    try:
        multi_period = hasattr(getattr(ams_ctx.system, routine), "timeslot")
    except Exception:
        multi_period = False
    if not multi_period:
        routine_note = (f"- Active routine switched from {routine} to **ED**: {routine} is "
                        f"single-period and would ignore the hourly horizon.\n")
        routine = "ED"
    ams_ctx.set_routine(routine)

    new_inputs = inputs.model_copy(update={
        "case_path": info["case_path"], "routine": routine,
        "load_overrides": {}, "gen_off": [], "line_off": [],
        "line_rate_overrides": {}, "disabled_constraints": [],
    })

    s = art.stats
    icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    region = REGIONS[spec.region]
    lines = [
        f"Built and loaded case `{art.case_id}`",
        f"- Period: {spec.period_label()} — {art.n_slots} hourly slots, {art.start} to {art.end}",
        f"- Load shape: {region.label} hourly demand (EIA Grid Monitor), distributed to "
        f"{info['n_pq']} loads by the base case's fixed shares",
        f"- System load: mean {s['system_mean_mw']:.0f} MW, peak {s['system_peak_mw']:.0f} MW, "
        f"min {s['system_min_mw']:.0f} MW (scale {s['scale_factor']:.4f} × regional MW; "
        f"shape is real, magnitude is the test system's)",
        f"- Base case: `{spec.base_case}` — generators, costs, line limits and topology unchanged",
        routine_note.rstrip("\n") if routine_note else None,
        "- Checks: " + "; ".join(f"{icon.get(c['status'], '?')} {c['name']}" for c in art.checks),
    ]
    for c in art.checks:
        if c["status"] != "ok":
            lines.append(f"  - {icon.get(c['status'], '?')} {c['name']}: {c['detail']}")
    for n in art.notes:
        lines.append(f"- Note: {n}")
    lines += [f"- Case file: `{art.xlsx_path}`",
              f"- Provenance and checks: `{art.manifest_path}`",
              f"- Next: say `run {routine}` to solve the {art.n_slots}-slot horizon."]
    content = "\n".join(x for x in lines if x)

    out = _reply(content, True, {"action": "build", "artifact": artifact_summary(art),
                                 "info": info, "routine": routine})
    out["inputs"] = new_inputs
    return out
