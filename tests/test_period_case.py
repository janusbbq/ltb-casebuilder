"""Tests for the case_build route extension.

Run from the repository root (LLM_LTB next to it, or ``LTB_LLM_AMS`` set)::

    ENLITEN_DATA_DIR=/path/to/ENLITEN-Grid-Econ-Data python -m pytest tests -q

Tests that need the regional load record are skipped when the dataset is not found.
No LLM is called: the agent tests use a stub that returns fixed structured outputs.
"""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agent.ams_engine.engine import AMSContext  # noqa: E402
from agent.nodes.route import router  # noqa: E402
from agent.prompts import get_prompts  # noqa: E402
from agent.utils.common_utils import create_initial_state  # noqa: E402
from agent.workflows.workflow import create_workflow  # noqa: E402
from casebuilder.ltb_plugin import CASE_BUILD  # noqa: E402
from casebuilder.node import case_build_agent, parse_base_case, parse_period  # noqa: E402
from casebuilder.period_case import (  # noqa: E402
    _WEEK_CASE_SUMMARY_ROWS,
    PeriodCaseError,
    available_periods,
    build_period_case,
    default_out_dir,
    find_data_dir,
    read_regional_load,
    slice_period,
)
from casebuilder.schema import PeriodCaseSpec  # noqa: E402

try:
    find_data_dir()
    HAVE_DATA = True
except PeriodCaseError:
    HAVE_DATA = False
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="ENLITEN-Grid-Econ-Data not found (set ENLITEN_DATA_DIR)")


class StubLLM:
    """Replies per structured-output schema NAME (the classifier / planner schemas are
    extension-aware subclasses built at run time). A reply is an instance, a dict of fields
    (built with the schema actually requested) or a callable; raises if none is registered."""

    def __init__(self, replies=None):
        self.replies = replies or {}
        self.calls = []

    def with_structured_output(self, schema):
        stub = self

        class _Runner:
            def invoke(self, messages):
                stub.calls.append(schema.__name__)
                if schema.__name__ not in stub.replies:
                    raise RuntimeError(f"stub has no reply for {schema.__name__}")
                reply = stub.replies[schema.__name__]
                if callable(reply):
                    return reply(messages)
                return schema(**reply) if isinstance(reply, dict) else reply

        return _Runner()

    def invoke(self, messages):
        return AIMessage(content="stub explanation")


def _state(text, routine="RTED"):
    st = create_initial_state(case_path="5bus/pjm5bus_demo.xlsx", routine=routine)
    st["messages"] = [HumanMessage(content=text)]
    return st


def _graph(llm, ctx):
    return create_workflow(llm, get_prompts(), ctx, extensions=[CASE_BUILD])


# ------------------------------------------------------------------ package boundary
def test_casedata_imports_neither_ams_nor_llm_ltb():
    # the data layer must stay movable to its own package: numpy / pandas only
    code = ("import sys, casebuilder.casedata, casebuilder.schema; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('ams', 'agent', 'andes')); "
            "print(bad); sys.exit(1 if bad else 0)")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": repo})
    assert res.returncode == 0, res.stdout + res.stderr


# ------------------------------------------------------------------ parser (no data needed)
@pytest.mark.parametrize("text,expect", [
    ("build the IEEE39 case for July 2020", {"year": 2020, "month": 7}),
    ("生成2020年1月的ieee39算例并运行ED", {"year": 2020, "month": 1}),
    ("build a 2020-07 case, first 5 days", {"year": 2020, "month": 7, "days": 5}),
    ("create a case for 07/2020 with 7 days", {"year": 2020, "month": 7, "days": 7}),
    ("2021年 十一月 的负荷，前14天", {"year": 2021, "month": 11, "days": 14}),
    ("2020年7月前两周", {"year": 2020, "month": 7, "days": 14}),
    ("make a case for 2019", {"year": 2019}),
    ("you may build the 2020 case", {"year": 2020}),                 # modal 'may' is not a month
    ("the load in may 2020", {"year": 2020, "month": 5}),
    ("first week of March 2020", {"year": 2020, "month": 3, "days": 7}),
    ("January 2018 from day 10 for 3 days using new england load",
     {"year": 2018, "month": 1, "days": 3, "start_day": 10, "region": "NE"}),
    ("March 2020 in UTC", {"year": 2020, "month": 3, "timezone": "utc"}),
    ("build a case please", {}),
])
def test_parse_period(text, expect):
    assert parse_period(text) == expect


@pytest.mark.parametrize("text,expect", [
    ("build the IEEE39 case for July 2020", "ieee39_uced"),          # plain case -> its *_uced twin
    ("create ieee39_uced case for 07/2020", "ieee39_uced"),
    ("build 39 bus case for 2020", "ieee39_uced"),
    ("one week of feb 2020 on pjm 5 bus", "pjm5bus"),
    ("build a 2020-07 case, first 5 days", None),                    # '5 days' is not the 5-bus case
    ("2021年11月 前14天", None),                                       # '14天' is not the 14-bus case
])
def test_parse_base_case(text, expect):
    assert parse_base_case(text) == expect


def test_spec_validation():
    with pytest.raises(ValueError):
        PeriodCaseSpec(year=2020, start_day=5)                        # start_day needs a month
    with pytest.raises(ValueError):
        PeriodCaseSpec(year=2020, month=13)
    assert PeriodCaseSpec(year=2020, month=7).case_id("ieee39_uced") == "ieee39_uced_NE_2020_07"
    assert PeriodCaseSpec(year=2020, month=1, days=3).case_id("x") == "x_NE_2020_01_d01x3"


# ------------------------------------------------------------------ data layer
@needs_data
def test_read_regional_load_metadata():
    rl = read_regional_load("NE")
    assert rl.series_id == "EBA.NE-ALL.D.H"
    assert str(rl.series.index.tz) == "UTC"
    assert rl.series.index.is_monotonic_increasing and not rl.series.index.has_duplicates
    assert "eia.gov" in rl.url and len(rl.sha256) == 64


@needs_data
@pytest.mark.parametrize("month,hours", [(7, 744), (2, 696), (3, 743), (11, 721)])   # leap Feb; DST months
def test_slice_local_month_hours(month, hours):
    sel = slice_period(read_regional_load("NE"), PeriodCaseSpec(year=2020, month=month))
    assert len(sel) == hours
    assert sel.index[0].strftime("%Y-%m-%d %H:%M") == f"2020-{month:02d}-01 00:00"
    assert sel.index[0].tzinfo is not None and sel.index[-1].hour == 23


@needs_data
def test_available_periods():
    cov = available_periods("NE")
    assert "2020-07" in cov["complete_months"] and 2020 in cov["complete_years"]
    assert "2015-06" not in cov["complete_months"]


# ------------------------------------------------------------------ builder
@needs_data
def test_build_month_and_solve(tmp_path):
    art = build_period_case(PeriodCaseSpec(year=2020, month=1, days=3), out_dir=str(tmp_path))
    assert art.n_slots == 72 and art.ok
    assert {c["name"] for c in art.checks} == {"coverage", "source_values", "capacity", "min_generation", "roundtrip"}

    # the profile is exactly share_i * peak_mw * R(t) / max_2020 R
    rl = read_regional_load("NE")
    sel = slice_period(rl, PeriodCaseSpec(year=2020, month=1, days=3))
    local = rl.series.tz_convert("America/New_York")
    ref_peak = local[local.index.year == 2020].max()
    prof = pd.read_csv(art.profile_csv)
    loads = [c for c in prof.columns if c not in ("hour", "timestamp")]
    total = prof[loads].sum(axis=1).to_numpy()
    base_total = art.stats["base_total_load_mw"]
    np.testing.assert_allclose(total, base_total * sel.to_numpy() / ref_peak, rtol=1e-12)
    assert total.max() <= base_total + 1e-9                        # reference-year peak maps to base total

    # manifest carries provenance and the checks
    man = json.load(open(art.manifest_path))
    assert man["provenance"]["series_id"] == "EBA.NE-ALL.D.H"
    assert man["provenance"]["source_file"]["sha256"] == rl.sha256
    assert man["scale"]["ref_year"] == 2020 and man["period"]["n_slots"] == 72
    summ = pd.read_excel(art.xlsx_path, "Summary")
    load_rows = summ[summ["field"] == "Load"]["comment"].tolist()
    if _WEEK_CASE_SUMMARY_ROWS:                                    # base's stale Load note replaced
        assert len(load_rows) == 1 and "EBA.NE-ALL.D.H" in load_rows[0]
        assert man["provenance"]["xlsx_summary_note"] == "replaced"
    else:                                                          # older week_case: said so, not hidden
        assert man["provenance"]["xlsx_summary_note"] == "base case's (stale)"
        assert any("summary_rows" in n for n in art.notes)

    # the solver sees the profile and ED converges
    ctx = AMSContext()
    ctx.load_case(art.xlsx_path)
    ctx.set_routine("ED")
    res = ctx.solve(solver="CLARABEL", ignore_dpp=True)
    assert res["converged"] and np.shape(res["pg"])[1] == 72
    pds_mw = np.asarray(ctx.active_routine().pds.v) * ctx.system.config.mva
    np.testing.assert_allclose(pds_mw.sum(axis=0), total, rtol=1e-6)
    np.testing.assert_allclose(np.sum(res["pg"], axis=0) * ctx.system.config.mva, total, rtol=1e-4)


@needs_data
def test_months_differ(tmp_path):
    jan = build_period_case(PeriodCaseSpec(year=2020, month=1, days=7), out_dir=str(tmp_path))
    jul = build_period_case(PeriodCaseSpec(year=2020, month=7, days=7), out_dir=str(tmp_path))
    assert jan.case_id != jul.case_id and jan.xlsx_path != jul.xlsx_path
    assert abs(jan.stats["system_mean_mw"] - jul.stats["system_mean_mw"]) > 1.0


@needs_data
@pytest.mark.parametrize("kwargs,needle", [
    (dict(year=2015, month=6), "not fully covered"),
    (dict(year=2023, month=1), "not fully covered"),
    (dict(year=2020, month=7, peak_mw=12000), "infeasible"),
    (dict(year=2020, month=2, start_day=30), "invalid start date"),
    (dict(year=2020, month=7, base_case="ieee39"), "ieee39_uced"),
    (dict(year=2020, month=7, ref_year=2022), "ref_year 2022 is not fully covered"),
])
def test_refusals(tmp_path, kwargs, needle):
    with pytest.raises(PeriodCaseError) as exc:
        build_period_case(PeriodCaseSpec(**kwargs), out_dir=str(tmp_path))
    assert needle in str(exc.value)
    assert not list(tmp_path.glob("*.xlsx"))                          # nothing written on refusal


@needs_data
def test_partial_year_falls_back_to_full_reference_year(tmp_path):
    art = build_period_case(PeriodCaseSpec(year=2022, month=3, days=2), out_dir=str(tmp_path))
    man = json.load(open(art.manifest_path))
    assert man["scale"]["ref_year"] == 2021 and any("2021" in n for n in art.notes)


# ------------------------------------------------------------------ agent node
@needs_data
def test_node_builds_loads_and_switches_routine(tmp_path, monkeypatch):
    monkeypatch.setenv("AMS_PERIOD_DIR", str(tmp_path))
    ctx = AMSContext()
    ctx.load_case("5bus/pjm5bus_demo.xlsx")
    # the LLM proposes a WRONG month; the deterministic parser must override it
    llm = StubLLM({"PeriodCaseSpec": PeriodCaseSpec(year=2020, month=12)})
    prompts = get_prompts() | CASE_BUILD.prompts
    out = case_build_agent(_state("build the IEEE39 case for January 2020, first 2 days"), llm, prompts, ctx)
    nr = out["node_response"]
    assert nr.success and nr.data["artifact"]["n_slots"] == 48
    assert nr.data["artifact"]["case_id"] == "ieee39_uced_NE_2020_01_d01x2"
    assert out["inputs"].routine == "ED" and out["inputs"].case_path.endswith("ieee39_uced_NE_2020_01_d01x2.xlsx")
    assert ctx.system.Bus.n == 39 and ctx.system.EDSlot.n == 48 and ctx.routine_name == "ED"
    assert "switched from RTED to **ED**" in out["messages"][0].content


@needs_data
def test_node_works_when_llm_fails_and_asks_when_no_period(tmp_path, monkeypatch):
    monkeypatch.setenv("AMS_PERIOD_DIR", str(tmp_path))
    prompts = get_prompts() | CASE_BUILD.prompts
    ctx = AMSContext()
    out = case_build_agent(_state("生成2020年7月的ieee39算例，前1天", routine="ED"), StubLLM(), prompts, ctx)
    assert out["node_response"].success and ctx.system.EDSlot.n == 24

    out = case_build_agent(_state("build me a new case"), StubLLM(), prompts, AMSContext())
    assert not out["node_response"].success and out["node_response"].data["action"] == "need_period"
    assert "inputs" not in out and "error_info" not in out


@needs_data
def test_node_refusal_is_direct(tmp_path, monkeypatch):
    monkeypatch.setenv("AMS_PERIOD_DIR", str(tmp_path))
    ctx = AMSContext()
    out = case_build_agent(_state("build the ieee39 case for January 2023"), StubLLM(),
                           get_prompts() | CASE_BUILD.prompts, ctx)
    assert not out["node_response"].success and out["node_response"].data["action"] == "refused"
    assert "not fully covered" in out["messages"][0].content and ctx.system is None


# ------------------------------------------------------------------ full graph (registered as an extension)
@needs_data
def test_graph_single_route(tmp_path, monkeypatch):
    monkeypatch.setenv("AMS_PERIOD_DIR", str(tmp_path))
    ctx = AMSContext()
    llm = StubLLM({"MessageClassifier": {"message_type": "case_build", "is_multi_step": False}})
    final = _graph(llm, ctx).invoke(_state("Build the IEEE39 case for July 2020, first 1 day"),
                                    config={"recursion_limit": 50})
    assert final["inputs"].routine == "ED" and ctx.system.EDSlot.n == 24
    assert "Built and loaded case `ieee39_uced_NE_2020_07_d01x1`" in final["messages"][-1].content


@needs_data
def test_graph_build_then_solve(tmp_path, monkeypatch):
    monkeypatch.setenv("AMS_PERIOD_DIR", str(tmp_path))
    monkeypatch.setenv("AMS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    ctx = AMSContext()
    llm = StubLLM({
        "MessageClassifier": {"message_type": "solve", "is_multi_step": True},
        "MultiStepPlan": {"description": "build July 2020 and run ED", "steps": [
            {"action": "case_build", "content": "build the ieee39 case for July 2020, first 1 day"},
            {"action": "solve", "content": "run ED"},
        ]},
    })
    final = _graph(llm, ctx).invoke(_state("生成2020年7月第一天的ieee39算例并运行ED"),
                                    config={"recursion_limit": 50})
    assert final["results"]["routine"] == "ED" and final["results"]["converged"]
    assert np.shape(final["results"]["pg"]) == (10, 24)
    assert "Step 1 (case_build)" in final["messages"][-1].content
    assert "Step 2 (solve)" in final["messages"][-1].content


def test_graph_via_environment(monkeypatch):
    monkeypatch.setenv("LTB_EXTENSIONS", "casebuilder.ltb_plugin:CASE_BUILD")
    graph = create_workflow(StubLLM(), get_prompts(), AMSContext())
    assert "case_build" in graph.get_graph().nodes


# ------------------------------------------------------------------ comparison + CLI
@needs_data
def test_compare_periods_and_cli(tmp_path, capsys):
    from casebuilder import cli
    from casebuilder.period_compare import compare_periods

    res = compare_periods([PeriodCaseSpec(year=2020, month=m, days=2) for m in (1, 7)], out_dir=str(tmp_path))
    assert [r["period"] for r in res["rows"]] == ["2020-01 (2 days from day 1)", "2020-07 (2 days from day 1)"]
    assert all(r["converged"] and r["slots"] == 48 and r["balance_max_abs_mw"] < 1e-3 for r in res["rows"])
    assert res["rows"][0]["lmp_mean"] != res["rows"][1]["lmp_mean"]
    hourly = pd.read_csv(res["hourly_csv"])
    assert len(hourly) == 96 and {"period", "slot", "timestamp", "load_mw", "lmp_usd_per_mwh"} <= set(hourly.columns)
    assert os.path.getsize(res["png"]) > 10_000

    assert cli.main(["--year", "2020", "--month", "7", "--days", "1", "--out", str(tmp_path)]) == 0
    assert "built ieee39_uced_NE_2020_07_d01x1: 24 slots" in capsys.readouterr().out
    assert cli.main(["--year", "2023", "--month", "1", "--out", str(tmp_path)]) == 1
    assert "not fully covered" in capsys.readouterr().err


# ------------------------------------------------------------------ review fixes carried over
def test_default_out_dir_is_not_web_served(monkeypatch, tmp_path):
    # LLM_LTB's web backend publishes <cwd>/generated at /generated; built cases must not land there
    monkeypatch.delenv("AMS_PERIOD_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    out = default_out_dir()
    assert out == os.path.join(str(tmp_path), "data", "period_cases")
    assert "generated" not in os.path.relpath(out, str(tmp_path)).split(os.sep)


@pytest.mark.parametrize("text,message_type,expect", [
    # conceptual questions that contain a build verb and a solve hint stay single-route
    ("How does AMS create the optimization model for ED?", "question_general", "question_general"),
    ("What does generate mean when you run RTED?", "question_general", "question_general"),
    ("如何生成计算结果的图表", "question_general", "question_general"),
    # a period build + solve is compound even if the classifier under-reports it
    ("Build the IEEE39 case for July 2020 and solve it", "case_build", "planner"),
    ("生成2020年7月的ieee39算例并运行ED", "case_build", "planner"),
    # a period build alone goes straight to the extension route
    ("Build the IEEE39 case for July 2020", "case_build", "case_build"),
    # existing change-then-solve heuristic is untouched
    ("change load PQ_1 to 3.2 and solve", "solve", "planner"),
])
def test_router_build_hints_need_a_year(text, message_type, expect):
    st = _state(text)
    st["message_type"], st["is_multi_step"] = message_type, False
    assert router(st, (CASE_BUILD,))["next"] == expect


# ------------------------------------------------------------------ LLM extraction guard (seen live)
def _extract(text, proposal):
    from casebuilder.node import _extract_spec
    inputs = create_initial_state(case_path="5bus/pjm5bus_demo.xlsx")["inputs"]
    return _extract_spec(text, StubLLM({"PeriodCaseSpec": proposal}), get_prompts() | CASE_BUILD.prompts, inputs)


def test_extract_drops_unstated_optional_fields():
    # llama3.1:8b proposed timezone="utc" for this text; the session case path is another trap
    spec = _extract("生成2020年7月的ieee39算例，前3天",
                    {"year": 2020, "month": 7, "days": 3, "timezone": "utc", "region": "NY",
                     "base_case": "5bus/pjm5bus_demo.xlsx", "peak_mw": 9999, "ref_year": 2019})
    assert (spec.year, spec.month, spec.days) == (2020, 7, 3)
    assert spec.timezone == "local" and spec.region == "NE" and spec.base_case == "ieee39_uced"
    assert spec.peak_mw is None and spec.ref_year is None


def test_extract_keeps_stated_optional_fields():
    spec = _extract("July 2020 in UTC from new york load, peak 4,000 MW, scaled to the 2019 peak",
                    {"year": 2020, "month": 7, "timezone": "utc", "region": "NY",
                     "peak_mw": 4000, "ref_year": 2019})
    assert spec.timezone == "utc" and spec.region == "NY"
    assert spec.peak_mw == 4000 and spec.ref_year == 2019


def _extract_step(step, user, proposal=None):
    from casebuilder.node import _extract_spec
    inputs = create_initial_state(case_path="5bus/pjm5bus_demo.xlsx")["inputs"]
    return _extract_spec(step, StubLLM({"PeriodCaseSpec": proposal or {}}),
                         get_prompts() | CASE_BUILD.prompts, inputs, user)


def test_extract_planner_step_cannot_add_optional_fields():
    # llama3.1:8b planner step, seen live 2026-10-27 (user named no base case here)
    step = ("Build a case for July 2020, using real load data for the month, based on the "
            "5bus/pjm5bus_demo.xlsx base case, for the PJM region, in UTC")
    spec = _extract_step(step, "Generate a July 2020 case and run ED")
    assert (spec.year, spec.month) == (2020, 7)
    assert spec.base_case == "ieee39_uced" and spec.region == "NE" and spec.timezone == "local"


def test_extract_planner_step_keeps_what_the_user_said():
    # the step dropped the user's base case, region and time zone; the user's words win
    spec = _extract_step("build case for 2020 July", "生成2020年7月的pjm5bus算例，用new york负荷，UTC，并运行ED")
    assert spec.base_case == "pjm5bus" and spec.region == "NY" and spec.timezone == "utc"
    # a step naming one of several stated regions keeps it
    spec = _extract_step("build July 2020 from New York load",
                         "build July 2020 cases from New England and New York load and run ED")
    assert spec.region == "NY"


def test_user_text_of_a_planned_turn():
    from casebuilder.node import _user_text
    from agent.schemas.planner import MultiStepPlan
    st = _state("Build the IEEE39 case for July 2020 and run ED")
    assert _user_text(st) is None
    st["plan"] = MultiStepPlan(description="d", steps=[
        {"action": "solve", "content": "build July 2020"}, {"action": "solve", "content": "run ED"}])
    st["is_compound"] = True
    st["messages"].append(HumanMessage(content="build July 2020"))
    assert _user_text(st) == "Build the IEEE39 case for July 2020 and run ED"
