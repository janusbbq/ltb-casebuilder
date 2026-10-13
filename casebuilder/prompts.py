"""Prompt texts the case_build route adds to LLM_LTB (via ``RouteExtension``).

``CLASSIFIER_HINT`` and ``PLANNER_HINT`` are inserted into LLM_LTB's classifier and planner
prompts by ``agent.extensions.extend_prompts``; ``CASE_BUILD_EXTRACT_SYSTEM`` is this route's
own prompt (key ``case_build_extract``). Wording carried over from the in-tree version.
"""

CLASSIFIER_HINT = """
BUILD / GENERATE / CREATE a NEW case for a calendar PERIOD (a month, a year, or some days of a month) from real regional hourly load data. The tell-tale is a year and/or month together with a verb like build / generate / create / make / 生成 / 构建 / 创建. This is NOT case_io (which only loads an existing file) and NOT modify (which edits one load value).
  Examples: "Build the IEEE39 case for July 2020", "Generate a January 2021 case", "Create a 2020-07 case with the first 7 days", "make a 2019 case from New England load", "生成2020年7月的ieee39算例", "构建2021年1月前两周的算例"
""".strip()

PLANNER_HINT = """
BUILDING a new case for a calendar period (month / year) from real load data is
   ``case_build``. Put the WHOLE period description (year, month, days, base case,
   region) into that ONE step's ``content``, copied from the user's wording. It loads
   the built case itself and selects ED — do NOT add a ``case_io`` step for it, and
   add a ``configure`` step only if the user asks for a routine other than ED.
""".strip()

CASE_BUILD_EXTRACT_SYSTEM = """
Extract the parameters of a PERIOD CASE BUILD from the user's request (English or Chinese).

A period case = a base case + the hourly load of one calendar period, taken from a real
regional demand record. Fill ONLY what the user stated; leave everything else unset so the
defaults apply. Never invent a year or a month.

Fields:
- ``year``      — calendar year, REQUIRED (e.g. 2020).
- ``month``     — 1-12. Omit when the user asks for a whole year.
- ``start_day`` — first day of the month to include. Omit unless stated.
- ``days``      — number of days. Omit for the whole month / year ("first week" = 7).
- ``base_case`` — only if the user names a case (e.g. "ieee39", "ieee39_uced", "pjm5bus").
- ``region``    — one of {regions}; only if the user names a region (New England = NE).
- ``peak_mw``   — only if the user gives a target system peak in MW.
- ``ref_year``  — only if the user names a reference year for scaling.
- ``timezone``  — "utc" only if the user asks for UTC.

Examples:
- "Build the IEEE39 case for July 2020" → year=2020, month=7, base_case="ieee39"
- "生成2021年1月前两周的算例" → year=2021, month=1, days=14
- "make a 2019 case" → year=2019

**Current session**: case_path: {case_path}
""".strip()
