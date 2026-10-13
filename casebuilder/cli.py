"""Command-line entry for the period-case builder (no LLM involved).

Run with LLM_LTB's ``LLM_AMS`` directory on ``PYTHONPATH`` and a checkout of
ENLITEN-Grid-Econ-Data (see README). Examples::

    # one month, default base case ieee39_uced, New England load shape
    python -m casebuilder.cli --year 2020 --month 7

    # the same, then solve ED on the whole month
    python -m casebuilder.cli --year 2020 --month 7 --run ED

    # compare two months of 2020: tables + figure
    python -m casebuilder.cli --year 2020 --compare 1 7

    # what the regional record covers
    python -m casebuilder.cli --coverage
"""

import argparse
import json
import sys
import warnings

from casebuilder.period_case import (
    REGIONS,
    PeriodCaseError,
    available_periods,
    build_period_case,
    default_out_dir,
)
from casebuilder.period_compare import compare_periods, solve_period
from casebuilder.schema import PeriodCaseSpec

_ICON = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a month/year-specific AMS case from regional hourly load.")
    ap.add_argument("--year", type=int, help="calendar year, e.g. 2020")
    ap.add_argument("--month", type=int, help="calendar month 1-12 (omit for the whole year)")
    ap.add_argument("--start-day", type=int, default=1)
    ap.add_argument("--days", type=int, help="number of days (omit for the whole month/year)")
    ap.add_argument("--base", default="ieee39_uced", help="base case key or path (default ieee39_uced)")
    ap.add_argument("--region", default="NE", choices=sorted(REGIONS))
    ap.add_argument("--peak-mw", type=float, help="system MW the reference-year peak maps to "
                                                  "(default: base case total load)")
    ap.add_argument("--ref-year", type=int, help="year whose regional peak sets the scale (default: --year)")
    ap.add_argument("--utc", action="store_true", help="slice the period in UTC instead of local time")
    ap.add_argument("--out", default=None, help=f"output directory (default {default_out_dir()})")
    ap.add_argument("--data", default=None, help="ENLITEN-Grid-Econ-Data checkout (default: $ENLITEN_DATA_DIR)")
    ap.add_argument("--run", metavar="ROUTINE", help="solve the built case with this routine (e.g. ED)")
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--compare", type=int, nargs="+", metavar="MONTH",
                    help="build + solve these months of --year and write comparison tables and a figure")
    ap.add_argument("--coverage", action="store_true", help="print what the region's record covers and exit")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    try:
        if args.coverage:
            cov = available_periods(args.region, args.data, "utc" if args.utc else "local")
            print(f"{args.region}: {cov['first']} .. {cov['last']}")
            print(f"complete years : {cov['complete_years']}")
            print(f"complete months: {len(cov['complete_months'])} "
                  f"({cov['complete_months'][0]} .. {cov['complete_months'][-1]})")
            return 0
        if args.year is None:
            ap.error("--year is required")

        def spec(month):
            return PeriodCaseSpec(base_case=args.base, region=args.region, year=args.year, month=month,
                                  start_day=args.start_day, days=args.days, peak_mw=args.peak_mw,
                                  ref_year=args.ref_year, timezone="utc" if args.utc else "local")

        if args.compare:
            res = compare_periods([spec(m) for m in args.compare], out_dir=args.out,
                                  routine=args.run or "ED", solver=args.solver, data_dir=args.data)
            for r in res["rows"]:
                lmp = (f"LMP mean {r['lmp_mean']:.2f} (min {r['lmp_min']:.2f}, max {r['lmp_max']:.2f}) $/MWh"
                       if r["lmp_mean"] is not None else "no LMP")
                print(f"{r['period']:<28} {r['slots']:>5} slots  converged={r['converged']}  "
                      f"load mean {r['load_mean_mw']:.0f} / peak {r['load_peak_mw']:.0f} MW  {lmp}")
            print(f"summary: {res['summary_csv']}\nhourly : {res['hourly_csv']}\nfigure : {res['png']}")
            return 0 if all(r["converged"] for r in res["rows"]) else 2

        art = build_period_case(spec(args.month), out_dir=args.out, data_dir=args.data)
        s = art.stats
        print(f"built {art.case_id}: {art.n_slots} slots, {art.start} .. {art.end}")
        print(f"  system load mean {s['system_mean_mw']:.0f} MW, peak {s['system_peak_mw']:.0f} MW, "
              f"min {s['system_min_mw']:.0f} MW (scale {s['scale_factor']:.4f})")
        for c in art.checks:
            print(f"  {_ICON[c['status']]} {c['name']}: {c['detail']}")
        for n in art.notes:
            print(f"  note: {n}")
        print(f"  case     : {art.xlsx_path}\n  manifest : {art.manifest_path}")
        if args.run:
            sol = solve_period(art, routine=args.run, solver=args.solver)
            lmp = sol.get("lmp")
            print(f"  {args.run}: converged={sol['converged']} objective={sol['objective']}"
                  + (f" LMP mean {lmp.mean():.2f} $/MWh" if lmp is not None else ""))
            return 0 if sol["converged"] else 2
        return 0
    except PeriodCaseError as exc:
        print(f"not built: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
