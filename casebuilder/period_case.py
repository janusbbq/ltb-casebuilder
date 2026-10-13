"""Build a period-specific (month / year) multi-period AMS case from a regional hourly load record.

Everything here is deterministic; the LLM only fills a :class:`PeriodCaseSpec`.

Pipeline
--------
1. ``casebuilder.casedata`` reads the region's hourly demand, slices the requested period in
   the region's local calendar (or UTC) and refuses it unless every hour is present.
2. Scale and distribute::

       total(t)  = peak_mw * R(t) / max_{ref_year} R        (MW)
       load_i(t) = total(t) * p0_i / sum_j p0_j            (fixed participation factors)

   ``peak_mw`` defaults to the base case's total load, so the reference year's regional peak
   maps onto the base case's own load level and the case stays inside its generation fleet.
   Only the SHAPE is real; the magnitude is the test system's.
3. Write ``profile.csv`` and hand it to LLM_LTB's ``week_case.build_week_case``, which rebuilds
   the ED/UC slot tables and verifies the written file reproduces the profile.
4. Run the input checks and write ``<case_id>.period.json`` with provenance, the
   transformation, period statistics and the check results.

What this does NOT change: generators, costs, line limits and topology all stay those of the
base case. Only the load horizon is period-specific.

Needs ``ams`` and LLM_LTB's ``LLM_AMS`` directory on ``sys.path`` (``agent.ams_engine``).
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import ams
import numpy as np
import pandas as pd

from agent.ams_engine.case_catalog import SHIPPED_CASES
from agent.ams_engine.engine import resolve_case_path
from agent.ams_engine.week_case import WeekCaseArtifact, build_week_case
from casebuilder import casedata
from casebuilder.casedata import (  # noqa: F401  (re-exported for the node and tests)
    REGION_ALIASES,
    REGIONS,
    available_periods,
    find_data_dir,
    read_regional_load,
)
from casebuilder.schema import PeriodCaseSpec

OUT_ENV = "AMS_PERIOD_DIR"
SUMMARY_FIELD = "Load"          # replaces the base case's own "Load" note in the Summary sheet

# A request that cannot be built as asked; data-layer refusals are the same class.
PeriodCaseError = casedata.CaseDataError

# LLM_LTB's build_week_case gained ``summary_rows`` later; use it when it is there.
_WEEK_CASE_SUMMARY_ROWS = "summary_rows" in inspect.signature(build_week_case).parameters


def slice_period(rl: casedata.RegionalLoad, spec: PeriodCaseSpec) -> pd.Series:
    """Hourly MW for ``spec``'s period (see :func:`casedata.slice_period`)."""
    return casedata.slice_period(rl, spec.year, spec.month, spec.start_day, spec.days, spec.timezone)


def reference_peak(rl: casedata.RegionalLoad, spec: PeriodCaseSpec):
    return casedata.reference_peak(rl, spec.year, spec.ref_year, spec.timezone)


# --------------------------------------------------------------------------- #
# base case facts
# --------------------------------------------------------------------------- #
@dataclass
class BaseFacts:
    path: str
    stem: str
    mva: float
    load_idx: List[str]
    p0_mw: np.ndarray
    pmax_mw: float            # online generation capacity
    pmin_mw: float            # online minimum generation
    n_gen: int
    slot_models: List[str]


def _uced_sibling(case: str) -> Optional[str]:
    key = os.path.splitext(os.path.basename(str(case)))[0].lower()
    for cand in (f"{key}_uced", f"{key.split('_')[0]}_uced"):
        if cand in SHIPPED_CASES:
            return cand
    return None


def base_facts(base_case: str) -> BaseFacts:
    try:
        path = resolve_case_path(base_case)
    except Exception as exc:
        raise PeriodCaseError(f"could not resolve base case {base_case!r}: {exc}")
    try:
        ss = ams.load(path, setup=True, no_output=True)
    except Exception as exc:
        raise PeriodCaseError(f"could not load base case {base_case!r} ({path}): {exc}")
    slot_models = [m for m in ("EDSlot", "UCSlot") if getattr(ss, m).n]
    if not slot_models:
        sib = _uced_sibling(base_case)
        hint = f" Use `{sib}` instead." if sib else " Use a '*_uced' case."
        raise PeriodCaseError(f"Base case `{base_case}` has no ED/UC time-slot tables, so it cannot "
                              f"carry an hourly horizon.{hint}")
    mva = float(ss.config.mva)
    load_idx = [str(i) for i in ss.PQ.idx.v]
    p0 = np.asarray(ss.PQ.p0.v, dtype=float) * mva
    if (p0 <= 0).any():
        raise PeriodCaseError(f"base case {base_case!r} has loads with p0 <= 0; fixed participation "
                              "factors cannot represent them")
    pmax = pmin = 0.0
    n_gen = 0
    for model in (ss.PV, ss.Slack):
        if model.n:
            u = np.asarray(model.u.v, dtype=float)
            pmax += float((np.asarray(model.pmax.v, dtype=float) * u).sum()) * mva
            pmin += float((np.asarray(model.pmin.v, dtype=float) * u).sum()) * mva
            n_gen += int(u.sum())
    return BaseFacts(path=path, stem=os.path.splitext(os.path.basename(path))[0], mva=mva,
                     load_idx=load_idx, p0_mw=p0, pmax_mw=pmax, pmin_mw=pmin, n_gen=n_gen,
                     slot_models=slot_models)


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
@dataclass
class PeriodCaseArtifact:
    case_id: str
    xlsx_path: str
    manifest_path: str            # <case_id>.period.json (this module's manifest)
    week_manifest_path: str       # <case_id>.manifest.json (written by week_case)
    profile_csv: str
    n_slots: int
    start: str
    end: str
    stats: Dict[str, float] = field(default_factory=dict)
    checks: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    spec: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c["status"] != "fail" for c in self.checks)


def default_out_dir() -> str:
    # Not under generated/: LLM_LTB's web backend serves <cwd>/generated publicly at /generated,
    # and built workbooks / manifests (absolute local paths) must not be downloadable. Same
    # policy as the web's week cases (data/week_cases); data/ is inside the web's CASE_ROOTS.
    return os.path.abspath(os.environ.get(OUT_ENV) or os.path.join(os.getcwd(), "data", "period_cases"))


def plan_period(spec: PeriodCaseSpec, data_dir: Optional[str] = None):
    """Everything up to (not including) writing files: the sliced series, scale, per-load
    profile and input checks. Raises :class:`PeriodCaseError` on an unbuildable request."""
    rl = read_regional_load(spec.region, data_dir)
    base = base_facts(spec.base_case)
    sel = slice_period(rl, spec)
    ref_peak, ref_year, ref_note = reference_peak(rl, spec)
    base_total = float(base.p0_mw.sum())
    peak_mw = float(spec.peak_mw) if spec.peak_mw else base_total
    total, per_load = casedata.scale_and_distribute(sel.to_numpy(dtype=float), ref_peak, peak_mw, base.p0_mw)

    checks = casedata.source_checks(sel)
    peak, low = float(total.max()), float(total.min())
    margin = (base.pmax_mw - peak) / base.pmax_mw if base.pmax_mw else float("nan")
    if peak > base.pmax_mw:
        checks.append(casedata.check("capacity", "fail",
                                     f"period peak {peak:.0f} MW exceeds online generation capacity "
                                     f"{base.pmax_mw:.0f} MW ({base.n_gen} units)"))
    else:
        checks.append(casedata.check("capacity", "warn" if margin < 0.05 else "ok",
                                     f"period peak {peak:.0f} MW vs online capacity {base.pmax_mw:.0f} MW "
                                     f"(margin {margin:.1%}; reserve requirements not included)"))
    checks.append(casedata.check("min_generation", "warn" if low < base.pmin_mw else "ok",
                                 f"period minimum {low:.0f} MW vs sum of online Pmin {base.pmin_mw:.0f} MW"
                                 + (" - ED keeps every unit committed, so low-load slots may be infeasible"
                                    if low < base.pmin_mw else "")))
    notes = [n for n in (ref_note, rl.region.tz_note if spec.timezone == "local" else "") if n]
    return rl, base, sel, total, per_load, dict(ref_peak=ref_peak, ref_year=ref_year,
                                                peak_mw=peak_mw, base_total=base_total), checks, notes


def build_period_case(spec: PeriodCaseSpec, out_dir: Optional[str] = None,
                      data_dir: Optional[str] = None,
                      allow_infeasible: bool = False) -> PeriodCaseArtifact:
    """Build the case for ``spec`` and return its artifact.

    Writes, under ``out_dir`` (default ``data/period_cases``)::

        <case_id>.profile.csv    hour, timestamp, one MW column per PQ load
        <case_id>.xlsx           the AMS case (base case + rebuilt slot tables)
        <case_id>.manifest.json  week_case's manifest (hashes, round-trip verification)
        <case_id>.period.json    provenance, transformation, statistics, checks

    Raises :class:`PeriodCaseError` if the period is not covered or (unless
    ``allow_infeasible``) the scaled peak exceeds the base case's online capacity.
    """
    rl, base, sel, total, per_load, scale, checks, notes = plan_period(spec, data_dir)
    failed = [c for c in checks if c["status"] == "fail"]
    if failed and not allow_infeasible:
        raise PeriodCaseError(
            "Refusing to write an infeasible case: " + "; ".join(c["detail"] for c in failed)
            + f". Lower peak_mw (now {scale['peak_mw']:.0f} MW; base case total load is "
              f"{scale['base_total']:.0f} MW) or use a base case with more generation.")

    out_dir = os.path.abspath(out_dir or default_out_dir())
    os.makedirs(out_dir, exist_ok=True)
    case_id = spec.case_id(base.stem)
    profile_csv = os.path.join(out_dir, f"{case_id}.profile.csv")
    stamps = [t.strftime("%Y-%m-%d %H:%M%z") for t in sel.index]
    df = pd.DataFrame({"hour": np.arange(1, len(sel) + 1), "timestamp": stamps})
    for j, idx in enumerate(base.load_idx):
        df[idx] = per_load[:, j]
    df.to_csv(profile_csv, index=False)

    transformation = (
        f"total(t) = {scale['peak_mw']:.3f} MW * R(t) / {scale['ref_peak']:.0f} MW "
        f"(regional peak of {scale['ref_year']}); load_i(t) = total(t) * p0_i / sum(p0), "
        f"p0 from {os.path.basename(base.path)}")
    summary = (f"Hourly load shape: {rl.region.label} demand, EIA series {rl.series_id or '?'} "
               f"({os.path.basename(rl.path)}, sha256 {rl.sha256[:12]}), period {spec.period_label()} "
               f"in {'local time ' + rl.region.tz if spec.timezone == 'local' else 'UTC'}. "
               f"{transformation}. Only the shape is real; magnitude is the test system's. "
               f"Generators, costs, line limits and topology are unchanged from the base case.")
    kwargs = {"case_id": case_id, "unit": "MW"}
    if _WEEK_CASE_SUMMARY_ROWS:
        kwargs["summary_rows"] = [{"field": SUMMARY_FIELD, "comment": summary}]
    else:
        notes.append("This LLM_LTB's week_case has no summary_rows, so the xlsx Summary sheet still "
                     "carries the base case's own 'Load' note; the period's load source is recorded "
                     "in the .period.json manifest instead.")
    art: WeekCaseArtifact = build_week_case(base.path, profile_csv, out_dir, **kwargs)
    checks.append(casedata.check("roundtrip", "ok",
                                 f"reloaded file reproduces the profile for every load and slot "
                                 f"(max per-load residual {art.max_residual:.2e})"))

    stats = {
        "n_slots": int(len(sel)),
        "system_mean_mw": float(total.mean()), "system_peak_mw": float(total.max()),
        "system_min_mw": float(total.min()), "energy_mwh": float(total.sum()),
        "regional_mean_mw": float(sel.mean()), "regional_peak_mw": float(sel.max()),
        "regional_min_mw": float(sel.min()),
        "scale_factor": float(scale["peak_mw"] / scale["ref_peak"]),
        "online_capacity_mw": float(base.pmax_mw), "base_total_load_mw": float(scale["base_total"]),
    }
    provenance = casedata.provenance(rl, spec.timezone)
    provenance.update({
        "parameter": "load_p (per PQ load, per hour)",
        "transformation": transformation,
        "magnitude": "test-system scale - shape is real, magnitude is not",
        "participation_factors": "fixed, from base-case p0",
        "unchanged_from_base_case": ["generators", "costs", "line limits", "topology", "reserves"],
        "xlsx_summary_note": "replaced" if _WEEK_CASE_SUMMARY_ROWS else "base case's (stale)",
    })
    manifest_path = os.path.join(out_dir, f"{case_id}.period.json")
    manifest = {
        "case_id": case_id,
        "spec": spec.model_dump(),
        "period": {"label": spec.period_label(), "start": stamps[0], "end": stamps[-1],
                   "timezone": rl.region.tz if spec.timezone == "local" else "UTC",
                   "n_slots": int(len(sel))},
        "provenance": provenance,
        "scale": {"peak_mw": scale["peak_mw"], "ref_year": scale["ref_year"],
                  "ref_peak_regional_mw": scale["ref_peak"],
                  "peak_mw_source": "spec" if spec.peak_mw else "base case total load"},
        "base_case": {"key": spec.base_case, "path": base.path, "sha256": casedata.sha256_file(base.path),
                      "n_loads": len(base.load_idx), "slot_models": base.slot_models},
        "stats": stats, "checks": checks, "notes": notes,
        "files": {"xlsx": art.xlsx_path, "xlsx_sha256": art.sha256, "profile_csv": profile_csv,
                  "profile_sha256": casedata.sha256_file(profile_csv), "week_manifest": art.manifest_path},
        "written_at": datetime.now().isoformat(timespec="seconds"), "ams_version": ams.__version__,
        "solve_note": "multi-period: use routine ED (or UC); RTED/DCOPF ignore the slot tables",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return PeriodCaseArtifact(case_id=case_id, xlsx_path=art.xlsx_path, manifest_path=manifest_path,
                              week_manifest_path=art.manifest_path, profile_csv=profile_csv,
                              n_slots=int(len(sel)), start=stamps[0], end=stamps[-1], stats=stats,
                              checks=checks, notes=notes, spec=spec.model_dump())


def artifact_summary(art: PeriodCaseArtifact) -> Dict[str, object]:
    """JSON-able view of an artifact (for node responses / API payloads)."""
    return asdict(art)
