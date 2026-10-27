"""End-to-end check of ``case_build`` with a real LLM: the full LLM_LTB graph, one fresh
session per request (pjm5bus + RTED, like ``python main.py``), case built and solved.
Run from ``LLM_LTB/LLM_AMS`` in LLM_LTB's environment::

    PYTHONPATH=. python ~/projects/utk/ltb-casebuilder/scripts/live_e2e.py "Build the IEEE39 case for July 2020"
"""

import argparse
import sys
import time

from agent.core import setup_dependencies
from agent.session import SessionManager
from agent.workflows.workflow import create_workflow
from casebuilder.ltb_plugin import CASE_BUILD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--provider", default="ollama")
    args = ap.parse_args()

    llm, prompts, ams_ctx = setup_dependencies(args.provider)
    graph = create_workflow(llm, prompts, ams_ctx, extensions=[CASE_BUILD])
    for text in args.texts:
        session = SessionManager(graph, llm, ams_ctx, provider=args.provider)
        session.bootstrap()
        print(f"\n=== {text}", flush=True)
        t0 = time.time()
        nodes = [name for name, _ in session.execute_turn_streaming(text)]
        inputs = session.state["inputs"]
        plan = session.state.get("plan")
        print("nodes:", " -> ".join(nodes))
        if plan is not None:
            print("plan:", [(s.action, s.content) for s in plan.steps])
        print(f"case: {inputs.case_path}  routine: {inputs.routine}  ({time.time() - t0:.0f} s)")
        print("reply:", session.state["messages"][-1].content[:1500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
