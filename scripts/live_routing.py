"""Live routing check of ``case_build`` with a real LLM (no AMS solve, no case build).

For each phrasing: classifier -> router -> (planner when the router plans the turn),
exactly as LLM_LTB's graph runs them, with the plugin registered and, for the built-in
controls, also without it. Run from ``LLM_LTB/LLM_AMS`` in LLM_LTB's environment::

    python ~/projects/utk/ltb-casebuilder/scripts/live_routing.py --runs 3 --out routing.json

Needs Ollama (``OLLAMA_MODEL``, default llama3.1:8b) or ``--provider openai``.
"""

import argparse
import json
import sys
import time
from collections import Counter

from langchain_core.messages import HumanMessage

from agent.core import make_llm
from agent.extensions import build_schemas, extend_prompts
from agent.nodes.classify import classify_message
from agent.nodes.planner import planner_agent
from agent.nodes.route import router
from agent.prompts import get_prompts
from agent.schemas.inputs import Inputs
from casebuilder.ltb_plugin import CASE_BUILD

# (text, expected): expected is a route name for a direct turn, or a tuple of step
# actions for a planned turn.
BUILD = [
    ("Build the IEEE39 case for July 2020", "case_build"),
    ("Build the IEEE39 case for July 2020 and run ED", ("case_build", "solve")),
    ("生成2020年1月的ieee39算例并运行ED", ("case_build", "solve")),
    ("生成2020年7月的ieee39算例", "case_build"),
    ("生成2020年1月的ieee39算例，前3天", "case_build"),
    ("Build a 2019 case from New England load", "case_build"),
    ("Generate a January 2021 case", "case_build"),
    ("Create a case for the first week of March 2020", "case_build"),
    ("Make an IEEE39 case for August 2019 in UTC", "case_build"),
    ("构建2021年1月前两周的算例", "case_build"),
    ("Build the IEEE39 case for July 2020 and run UC", ("case_build", "configure", "solve")),
    ("用2020年7月的负荷数据建一个ieee39算例", "case_build"),
    # no base case named: the planner step must not bring one in (see node._extract_spec)
    ("Generate a January 2021 case and run ED", ("case_build", "solve")),
    ("生成2020年7月的算例并运行ED", ("case_build", "solve")),
    ("Build a July 2020 case from New York load and run ED", ("case_build", "solve")),
]
CONTROLS = [
    ("load case ieee14", "case_io"),
    ("run RTED", "solve"),
    ("change load PQ_1 to 3.2 then solve", ("modify", "solve")),
    ("switch to ieee39 and run UC", ("case_io", "configure", "solve")),
    ("换成ieee39节点，运行uc问题", ("case_io", "configure", "solve")),
    ("load case ieee39_uced and run ED", ("case_io", "configure", "solve")),
    ("How does AMS create the optimization model?", "question_general"),
    ("What cases are available?", "question_parameter"),
]


def route_once(text, llm, prompts, cls_schema, plan_schema, exts):
    state = {"messages": [HumanMessage(content=text)], "inputs": Inputs(),
             "is_multi_step": False}
    t0 = time.time()
    state.update(classify_message(state, llm, prompts, cls_schema))
    state.update(router(state, exts))
    out = {"message_type": state["message_type"], "is_multi_step": state["is_multi_step"],
           "next": state["next"]}
    if state["next"] == "planner":
        plan = planner_agent(state, llm, prompts, plan_schema)["plan"]
        out["steps"] = [(s.action, s.content) for s in plan.steps]
        out["result"] = tuple(s.action for s in plan.steps)
    else:
        out["result"] = state["next"]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def ok(result, expected):
    return result == expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    llm = make_llm(args.provider)
    base = get_prompts()
    configs = {
        "plugin": ((CASE_BUILD,), extend_prompts(base, (CASE_BUILD,))),
        "no_plugin": ((), base),
    }
    cases = [("build", t, e, ("plugin",)) for t, e in BUILD] + \
            [("control", t, e, ("plugin", "no_plugin")) for t, e in CONTROLS]

    records = []
    for group, text, expected, cfg_names in cases:
        for cfg in cfg_names:
            exts, prompts = configs[cfg]
            cls_schema, plan_schema = build_schemas(exts)
            runs = [route_once(text, llm, prompts, cls_schema, plan_schema, exts)
                    for _ in range(args.runs)]
            results = Counter(str(r["result"]) for r in runs)
            passed = sum(ok(r["result"], expected) for r in runs)
            records.append({"group": group, "config": cfg, "text": text,
                            "expected": expected, "passed": passed, "runs": runs})
            mark = "PASS" if passed == args.runs else ("FLAKY" if passed else "FAIL")
            print(f"{mark:5} {passed}/{args.runs} [{cfg}] {text!r} -> {dict(results)}"
                  f"  (expected {expected})", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": getattr(llm, "_model_name", "?"), "runs": args.runs,
                       "records": records}, f, ensure_ascii=False, indent=1, default=list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
