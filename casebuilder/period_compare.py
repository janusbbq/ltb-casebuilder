"""Solve several period cases and compare them: one table, one hourly csv, one figure.

Deterministic companion to ``period_case``: build each requested period, solve it with a
multi-period routine (ED by default), and collect the system load and the price it implies.

Outputs under ``out_dir``:
    <tag>.summary.csv   one row per period (slots, mean/peak load, cost, mean/min/max LMP, ...)
    <tag>.hourly.csv    long table: period, slot, timestamp, load_mw, lmp
    <tag>.png           two stacked panels sharing the time axis (load on top, LMP below)

LMP is ``pi / mva`` in $/MWh (``pi`` is the dual of the nodal balance in $/p.u.; ED's interval
is 1 h). It is reported as the bus average together with the largest spread across buses, so a
case without binding line limits (spread = 0) is visible as such.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from agent.ams_engine.engine import AMSContext
from casebuilder.period_case import (
    REGIONS,
    PeriodCaseArtifact,
    build_period_case,
    default_out_dir,
)
from casebuilder.schema import PeriodCaseSpec

# Chart tokens. Categorical slots 1-3 validated as a set (all-pairs CVD and normal-vision
# separation on this surface); the third sits below 3:1 contrast, so series are also named
# in the legend and labelled at their peak - identity never rests on colour alone.
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MAX_SERIES = len(SERIES_COLORS)


def solve_period(art: PeriodCaseArtifact, routine: str = "ED", solver: str = "CLARABEL") -> Dict[str, object]:
    """Load a built period case, solve it, and return hourly system load / LMP and totals."""
    ctx = AMSContext()
    ctx.load_case(art.xlsx_path)
    ctx.set_routine(routine)
    res = ctx.solve(solver=solver, ignore_dpp=True)
    rtn = ctx.active_routine()
    mva = float(ctx.system.config.mva)
    load = np.asarray(rtn.pds.v, dtype=float).sum(axis=0) * mva                  # MW per slot
    pg = np.asarray(res["pg"], dtype=float) * mva                               # (gen, slot) MW
    out: Dict[str, object] = {
        "converged": bool(res["converged"]), "status": res.get("status", ""),
        "objective": res.get("objective"), "load_mw": load, "pg_mw": pg,
        "gen_idx": [str(g) for g in res["gen_idx"]],
        "balance_max_abs_mw": float(np.abs(pg.sum(axis=0) - load).max()),
    }
    if "pi" in res:
        pi = np.asarray(res["pi"], dtype=float) / mva                           # (bus, slot) $/MWh
        out["lmp"] = pi.mean(axis=0)
        out["lmp_spread"] = float((pi.max(axis=0) - pi.min(axis=0)).max())
    if "plf" in res:
        rate = np.asarray(ctx.system.Line.rate_a.v, dtype=float)[:, None]
        plf = np.abs(np.asarray(res["plf"], dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            out["max_line_loading"] = float(np.nanmax(np.where(rate > 0, plf / rate, np.nan)))
    return out


def compare_periods(specs: Sequence[PeriodCaseSpec], out_dir: Optional[str] = None,
                    routine: str = "ED", solver: str = "CLARABEL", tag: Optional[str] = None,
                    data_dir: Optional[str] = None, plot: bool = True) -> Dict[str, object]:
    """Build + solve every spec; write the summary / hourly tables and (optionally) the figure."""
    if not specs:
        raise ValueError("no periods to compare")
    out_dir = os.path.abspath(out_dir or default_out_dir())
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    hourly: List[pd.DataFrame] = []
    series: List[Dict[str, object]] = []
    for spec in specs:
        art = build_period_case(spec, out_dir=out_dir, data_dir=data_dir)
        sol = solve_period(art, routine=routine, solver=solver)
        label = spec.period_label()
        prof = pd.read_csv(art.profile_csv, usecols=["hour", "timestamp"])
        lmp = sol.get("lmp")
        rows.append({
            "period": label, "case_id": art.case_id, "region": spec.region, "base_case": spec.base_case,
            "slots": art.n_slots, "routine": routine, "solver": solver, "converged": sol["converged"],
            "load_mean_mw": float(np.mean(sol["load_mw"])), "load_peak_mw": float(np.max(sol["load_mw"])),
            "load_min_mw": float(np.min(sol["load_mw"])), "energy_mwh": float(np.sum(sol["load_mw"])),
            "objective": sol["objective"],
            "lmp_mean": float(np.mean(lmp)) if lmp is not None else None,
            "lmp_min": float(np.min(lmp)) if lmp is not None else None,
            "lmp_max": float(np.max(lmp)) if lmp is not None else None,
            "lmp_max_bus_spread": sol.get("lmp_spread"),
            "max_line_loading": sol.get("max_line_loading"),
            "balance_max_abs_mw": sol["balance_max_abs_mw"],
            "checks": "; ".join(f"{c['name']}={c['status']}" for c in art.checks),
            "xlsx": art.xlsx_path, "manifest": art.manifest_path,
        })
        h = pd.DataFrame({"period": label, "slot": prof["hour"], "timestamp": prof["timestamp"],
                          "load_mw": sol["load_mw"]})
        if lmp is not None:
            h["lmp_usd_per_mwh"] = lmp
        hourly.append(h)
        series.append({"label": label, "load": np.asarray(sol["load_mw"]), "lmp": lmp, "spec": spec})

    tag = tag or ("compare_" + "_vs_".join(s.case_id("x").split("x_", 1)[1] for s in specs))
    summary_csv = os.path.join(out_dir, f"{tag}.summary.csv")
    hourly_csv = os.path.join(out_dir, f"{tag}.hourly.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    pd.concat(hourly, ignore_index=True).to_csv(hourly_csv, index=False)
    out = {"rows": rows, "summary_csv": summary_csv, "hourly_csv": hourly_csv, "png": None}
    if plot:
        out["png"] = _plot(series, rows, os.path.join(out_dir, f"{tag}.png"), routine)
    return out


def _plot(series, rows, path: str, routine: str) -> Optional[str]:
    if len(series) > MAX_SERIES:
        # more periods than validated colour slots: the tables carry them all, the figure does not
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_lmp = all(s["lmp"] is not None for s in series)
    n_panels = 2 if has_lmp else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.3 * n_panels + 1.3), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[:, 0]
    spec0 = series[0]["spec"]
    region = REGIONS[spec0.region].label
    panels = [("load", "System load (MW)")] + ([("lmp", "LMP ($/MWh)")] if has_lmp else [])
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for s, color in zip(series, SERIES_COLORS):
            y = np.asarray(s[key], dtype=float)
            x = np.arange(len(y)) / 24.0 + 1.0                      # day of period, 1-based
            ax.plot(x, y, color=color, linewidth=1.0, solid_joinstyle="round",
                    solid_capstyle="round", label=s["label"])
            if key == "load":                                       # one direct label per series: its peak
                k = int(np.argmax(y))
                ax.plot([x[k]], [y[k]], marker="o", markersize=6, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
                ax.annotate(f"{s['label']} peak {y[k]:,.0f} MW", (x[k], y[k]), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=8.5, color=INK, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    max_days = max(len(s["load"]) for s in series) / 24.0
    step = 1 if max_days <= 14 else (2 if max_days <= 35 else 30)
    axes[-1].set_xticks(np.arange(1, int(np.ceil(max_days)) + 1, step))
    axes[-1].set_xlabel("Day of period (hourly values, local time)", color=INK_2, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.062, 0.945), ncol=len(series),
                     frameon=False, fontsize=9.5, handlelength=1.6, borderaxespad=0)
    for txt in leg.get_texts():
        txt.set_color(INK)
    for line in leg.get_lines():
        line.set_linewidth(2.2)
    fig.text(0.065, 0.975, f"{spec0.base_case}: {' vs '.join(s['label'] for s in series)}",
             ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    spread = max((r.get("lmp_max_bus_spread") or 0.0) for r in rows)
    note = (f"Load shape: {region} hourly demand (EIA), scaled to the test system. "
            f"{routine} solved for the whole horizon; generators, costs and line limits are the base case's.")
    if has_lmp:
        note += (" LMP is the bus average; it is identical at every bus (no line limit binds)."
                 if spread < 1e-6 else f" LMP is the bus average; largest spread across buses {spread:.2f} $/MWh.")
    import textwrap
    fig.text(0.065, 0.012, "\n".join(textwrap.wrap(note, 165)), ha="left", va="bottom", fontsize=8,
             color=INK_2, linespacing=1.4)
    for ax in axes:                                   # headroom so the peak labels stay inside the panel
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.905))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path
