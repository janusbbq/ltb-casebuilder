# ltb-casebuilder

Build AMS cases whose hourly load follows a **real calendar period** — a month, a year, or
some days of a month — and use them from the [LLM_LTB](https://github.com/CURENT/LLM_LTB)
agent as an extra route, `case_build`:

```
"Build the IEEE39 case for July 2020 and run ED"      "生成2020年1月的ieee39算例并运行ED"
```

It plugs into LLM_LTB through the route-extension interface (`agent.extensions`), so LLM_LTB's
own routes, prompts and schemas are not edited.

## What it does

```
user text ──► LLM fills PeriodCaseSpec; a deterministic parser re-reads year / month /
              days / base case from the text and overrides the LLM
          ──► casebuilder.casedata  (numpy / pandas only)
                regional hourly demand (EIA, UTC) → slice the period in local time
                (every hour required) → total(t) = peak_mw · R(t) / max_ref_year R
          ──► casebuilder.period_case  (ams + LLM_LTB)
                load_i(t) = total(t) · p0_i / Σp0 → profile.csv → week_case.build_week_case
                → <case_id>.xlsx + <case_id>.period.json (provenance, checks)
          ──► the node loads the case and switches the routine to ED
```

**Only the load shape is real.** Generators, costs, line limits, reserves and topology are
the base case's (`ieee39_uced` by default); the reference year's regional peak is mapped onto
the base case's total load. On `ieee39_uced` every bus therefore has the same LMP — prices
change over time, not across buses.

## Layout

| Module | Depends on | Role |
|---|---|---|
| `casebuilder/casedata/` | numpy, pandas | read EIA regional load, coverage, slicing, scaling, data checks, provenance |
| `casebuilder/schema.py` | pydantic | `PeriodCaseSpec` — what the LLM fills |
| `casebuilder/period_case.py` | ams, LLM_LTB | base-case facts, write the case via `week_case`, manifest |
| `casebuilder/period_compare.py` | ams, LLM_LTB | solve several periods → tables + figure |
| `casebuilder/node.py`, `prompts.py` | LLM_LTB | the `case_build` agent node, its parser and prompts |
| `casebuilder/ltb_plugin.py` | LLM_LTB | `CASE_BUILD`, the `RouteExtension` |
| `casebuilder/cli.py` | all of the above | command line, no LLM |

`casedata` must not import `ams` or LLM_LTB (`tests/test_period_case.py` checks it), so it can
become its own package later.

## Setup

Directory layout this repo assumes by default:

```
~/projects/utk/
├── LLM_LTB/                     # with agent/extensions.py (route extensions)
├── ltb-casebuilder/             # this repo
└── ENLITEN-Grid-Econ-Data/      # git clone https://github.com/enliten/ENLITEN-Grid-Econ-Data.git
```

Use LLM_LTB's environment (`conda activate llm-ams`). LLM_LTB is not pip-installable, so
both directories go on `PYTHONPATH`. Other locations: `ENLITEN_DATA_DIR` for the dataset,
`LTB_LLM_AMS` (tests only) for LLM_LTB's `LLM_AMS` directory.

## Use

**From the LLM_LTB agent** (CLI or web), run from `LLM_LTB/LLM_AMS`:

```bash
export PYTHONPATH=~/projects/utk/ltb-casebuilder:.
export LTB_EXTENSIONS=casebuilder.ltb_plugin:CASE_BUILD
python main.py                                   # or: python -m uvicorn web.backend.main:app
```

Then e.g. `Build the IEEE39 case for July 2020`, `生成2020年1月的ieee39算例并运行ED`.
Note: LLM_LTB's web `/api/chat` runs a parameter-control pre-check before the agent; with a
small local model it may reject the request before it reaches `case_build`.

**Without the LLM** (same code path), from `LLM_LTB/LLM_AMS`:

```bash
export PYTHONPATH=~/projects/utk/ltb-casebuilder:.
python -m casebuilder.cli --coverage                       # what the record covers
python -m casebuilder.cli --year 2020 --month 7 --run ED
python -m casebuilder.cli --year 2020 --compare 1 7        # tables + figure for two months
```

Outputs go to `./data/period_cases/` (`AMS_PERIOD_DIR` to override) — not `generated/`,
which LLM_LTB's web server publishes at `/generated`.

**Checks** written to `<case_id>.period.json`: `coverage`, `source_values` (outliers),
`capacity` (a period peak above online Pmax is refused, nothing is written),
`min_generation`, `roundtrip` (the reloaded xlsx reproduces the profile).

## Tests

```bash
cd ~/projects/utk/ltb-casebuilder
python -m pytest -q          # data-dependent tests skip when the dataset is not found
```

No LLM is called (a stub returns fixed structured outputs). After changing
`prompts.py`, also check with a real model that LLM_LTB's built-in compound requests are
planned the same with and without this extension (see LLM_LTB README, "Route extensions").

## Data

EIA Hourly Electric Grid Monitor exports shipped in
[ENLITEN-Grid-Econ-Data](https://github.com/enliten/ENLITEN-Grid-Econ-Data) (Zhang & Li,
*Scientific Data* 10, 646, 2023), regions NE, NY, CA, NW, SW, 2015-07 to 2022-06. The dataset
is GPL v3 and is not vendored here; this package only reads its files.
