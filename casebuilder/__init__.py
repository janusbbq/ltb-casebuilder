"""Period-specific AMS cases from real regional hourly load, as an LLM_LTB route extension.

Layers (imports only go downward):

* ``casebuilder.ltb_plugin``  — the ``RouteExtension`` registered with LLM_LTB
* ``casebuilder.node``        — the ``case_build`` agent node and its text parser
* ``casebuilder.period_case`` / ``period_compare`` — build / solve cases (needs ams + LLM_LTB)
* ``casebuilder.schema``      — ``PeriodCaseSpec`` (pydantic only; what the LLM fills)
* ``casebuilder.casedata``    — regional load records (numpy / pandas only)
"""

__version__ = "0.1.0"
