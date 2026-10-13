"""The ``case_build`` route as an LLM_LTB route extension.

Register it with LLM_LTB (both packages importable)::

    export LTB_EXTENSIONS=casebuilder.ltb_plugin:CASE_BUILD

or pass it explicitly: ``create_workflow(llm, prompts, ams_ctx, extensions=[CASE_BUILD])``.
"""

import re

from agent.extensions import RouteExtension
from casebuilder.node import case_build_agent
from casebuilder.prompts import CASE_BUILD_EXTRACT_SYSTEM, CLASSIFIER_HINT, PLANNER_HINT

_BUILD_HINTS = ("build", "generate", "create", "生成", "构建", "创建", "建立")
_YEAR_RE = re.compile(r"(?<!\d)20\d{2}(?!\d)")


def is_period_build(text: str) -> bool:
    """A build verb next to a year: "build ... July 2020 and solve" is change-then-solve.

    Build verbs alone are common in conceptual questions ("how does AMS create the
    optimization model?"), so they count only together with a year.
    """
    return any(b in text for b in _BUILD_HINTS) and _YEAR_RE.search(text) is not None


CASE_BUILD = RouteExtension(
    name="case_build",
    node=case_build_agent,
    summary="build/generate/create a NEW case for a calendar period — a month or year — "
            "from real regional hourly load data",
    classifier_hint=CLASSIFIER_HINT,
    planner_hint=PLANNER_HINT,
    prompts={"case_build_extract": {"system": CASE_BUILD_EXTRACT_SYSTEM}},
    is_change=is_period_build,
)
