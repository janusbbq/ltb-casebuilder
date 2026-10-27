# ltb-casebuilder

Build AMS cases whose hourly load follows a **real calendar period** — a month, a year, or
some days of a month — and use them from the [LLM_LTB](https://github.com/CURENT/LLM_LTB)
agent as an extra route, `case_build`:

```
"Build the IEEE39 case for July 2020 and run ED" 
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

**Requires LLM_LTB with route extensions** (`LLM_AMS/agent/extensions.py`). Until that is
merged into [CURENT/LLM_LTB](https://github.com/CURENT/LLM_LTB), get it from the
`jian-han/route-extensions` branch of [janusbbq/LLM_LTB](https://github.com/janusbbq/LLM_LTB):

```bash
cd ~/projects/utk
git clone https://github.com/janusbbq/LLM_LTB.git
git -C LLM_LTB checkout jian-han/route-extensions
git clone https://github.com/janusbbq/ltb-casebuilder.git
git clone https://github.com/enliten/ENLITEN-Grid-Econ-Data.git      # data, ~240 MB
```

Directory layout this repo assumes by default:

```
~/projects/utk/
├── LLM_LTB/                     # with agent/extensions.py (route extensions)
├── ltb-casebuilder/             # this repo
└── ENLITEN-Grid-Econ-Data/      # git clone https://github.com/enliten/ENLITEN-Grid-Econ-Data.git
```

Use LLM_LTB's environment (`conda activate llm-ams`; see LLM_LTB's README for installing it).
Install this package into it (below) or put it on `PYTHONPATH`; LLM_LTB itself is not
pip-installable and is run from its `LLM_AMS` directory. Other locations: `ENLITEN_DATA_DIR`
for the dataset, `LTB_LLM_AMS` (tests only) for LLM_LTB's `LLM_AMS` directory.

## Use

**One-time setup (recommended)** — then start LLM_LTB as usual, no `export` needed:

```bash
python -m pip install -e ~/projects/utk/ltb-casebuilder --no-deps   # in LLM_LTB's env
echo 'LTB_EXTENSIONS=casebuilder.ltb_plugin:CASE_BUILD' >> ~/projects/utk/LLM_LTB/LLM_AMS/.env
```

LLM_LTB's `agent/core.py` loads `.env` before the workflow is built, so both `python main.py`
and the web server pick the plugin up. A value exported in the shell wins over `.env`
(`export LTB_EXTENSIONS=` disables it for one run). Undo: `pip uninstall ltb-casebuilder`
and delete the line.

**Without installing** (CLI or web), run from `LLM_LTB/LLM_AMS`:

```bash
export PYTHONPATH=~/projects/utk/ltb-casebuilder:.
export LTB_EXTENSIONS=casebuilder.ltb_plugin:CASE_BUILD
python main.py                                   # or: python -m uvicorn web.backend.main:app
```

Then e.g. `Build the IEEE39 case for July 2020`, `生成2020年1月的ieee39算例并运行ED`.
Note: LLM_LTB's web `/api/chat` runs a parameter-control pre-check before the agent; with a
small local model it may reject the request before it reaches `case_build` (see Limitations).

**Without the LLM** (same code path), from `LLM_LTB/LLM_AMS`:

```bash
export PYTHONPATH=~/projects/utk/ltb-casebuilder:.
python -m casebuilder.cli --coverage                       # what the record covers
python -m casebuilder.cli --year 2020 --month 7 --run ED
python -m casebuilder.cli --year 2020 --compare 1 7        # tables + figure for two months
```

### What to ask

| Request | What happens |
|---|---|
| `Build the IEEE39 case for July 2020` | builds and loads the 744-slot case, switches the routine to ED |
| `Build the IEEE39 case for July 2020 and run ED` / `生成2020年7月的ieee39算例并运行ED` | builds, then solves ED in the same turn |
| `生成2020年1月的ieee39算例，前3天` | the first 3 days of January (72 slots) |
| `Build a 2019 case from New England load` | the whole year (8760 slots — untested at that size, see Limitations) |
| `July 2020 in UTC`, `… from new york load`, `… peak 4,000 MW`, `… scaled to the 2019 peak` | optional fields; only used when you say them |

Year, month and days are read from your words by a deterministic parser, which overrides
the LLM. Optional fields (time zone, region, base case, peak, reference year) are used only
when they appear in the request — otherwise the defaults apply (local time, New England,
`ieee39_uced`, base-case total load, the same year).

### Example reply

```
Built and loaded case `ieee39_uced_NE_2020_07`
- Period: 2020-07 — 744 hourly slots, 2020-07-01 00:00-0400 to 2020-07-31 23:00-0400
- Load shape: New England (ISO-NE) hourly demand (EIA Grid Monitor), distributed to 19 loads
- System load: mean 3108 MW, peak 4685 MW, min 2010 MW (scale 0.1897 × regional MW; shape is
  real, magnitude is the test system's)
- Checks: ✓ coverage; ✓ source_values; ✓ capacity; ✓ min_generation; ✓ roundtrip
- Case file: …/data/period_cases/ieee39_uced_NE_2020_07.xlsx
```

In the web app the built case then appears in the sidebar as `ieee39_uced_NE_2020_07 · 744 h`.

### Output files

Under `./data/period_cases/` (`AMS_PERIOD_DIR` to override) — not `generated/`, which
LLM_LTB's web server publishes at `/generated`:

| File | Contents |
|---|---|
| `<case_id>.xlsx` | the AMS case: base case + `EDSlot` / `EDSlotLoad` / `UCSlot` / … rebuilt for the period |
| `<case_id>.profile.csv` | hour, local timestamp, one MW column per load — what went into the case |
| `<case_id>.period.json` | the record of how it was made (below) |
| `<case_id>.manifest.json` | LLM_LTB `week_case`'s own manifest: hashes and round-trip residual |

`<case_id>.period.json` keys: `spec` (the request as understood), `period` (start, end, time
zone, slots), `provenance` (EIA series id, URL, source-file sha256, transformation formula),
`scale` (`peak_mw`, reference year and its regional peak), `stats` (system / regional mean,
peak, min MW), `checks`, `notes`, `files` (paths + hashes).

**Checks**: `coverage` (every hour present; otherwise refused), `source_values` (outliers
vs the period median), `capacity` (a period peak above online Pmax is refused, nothing is
written), `min_generation` (period minimum vs ΣPmin), `roundtrip` (the reloaded xlsx
reproduces the profile for every load and slot).

## Limitations

- **Only the load shape is period-specific.** Generators, costs, line limits, reserves and
  topology are the base case's. On `ieee39_uced` the line limits never bind and the unit costs
  are nearly identical, so the LMP is the **same at every bus** — prices vary over time only.
- **Magnitude is the test system's**: the reference year's regional peak is mapped onto the
  base case's total load (New England's real peak is ~25 GW; `ieee39_uced` carries 4.7 GW).
  All loads follow one regional curve with fixed shares.
- **Data range** 2015-07 to 2022-06 (UTC); a period with any missing hour is refused.
- **Routines**: ED over a month (744 slots) is verified. A whole year (8760 slots) has not
  been run — expect a large problem. **UC / EDES over long horizons are very slow**
  (mixed-integer; EDES took ~9 min for 168 slots on 5 buses) — keep them to days or a week.
- **Summary sheet**: with LLM_LTB's current `week_case` (no `summary_rows`) the xlsx `Summary`
  sheet still carries the base case's own load note; the period's source is in `.period.json`.
- **Web**: built cases live in the server's memory; after a restart ask for the case again.
  LLM_LTB's `/api/chat` pre-check may reject the request with a small local model (the CLI
  has no pre-check).

## Verification status (2026-10)

- Tests: 56 passed (no LLM; data tests skip without the dataset).
- llama3.1:8b, live routing (`scripts/live_routing.py`, 2026-10-27, 5 runs each): all 15
  period-build phrasings (English and Chinese) reach `case_build`; "… and run ED" plans as
  `case_build → configure(ED) → solve` (the `configure` step is redundant, harmless); the 8
  built-in controls route identically with and without the plugin.
- llama3.1:8b, end to end (`scripts/live_e2e.py`): 7 requests built the right case and solved
  ED (744 slots). The planner's step sometimes adds a base case the user never named
  ("… for the PJM 5-bus system"); the node keeps optional fields only when the user's own
  message states them.
- UC after a build fails with the default solver CLARABEL (not MIP) — the same as LLM_LTB's
  built-in "switch to ieee39_uced and run UC"; ask for a MIP solver too.
- Full month in the web app: *"Build the IEEE39 case for July 2020 and run ED"* — 744 slots,
  ED converged, sidebar and report name the built case.
- Not yet checked: OpenAI models; UC over a full month; a whole-year (8760-slot) case.

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

## License

MIT — see [LICENSE](LICENSE). The ENLITEN-Grid-Econ-Data dataset it reads is GPL v3 and is
not included; `ltbams` (LTB AMS), which it runs on, is GPL v3.
