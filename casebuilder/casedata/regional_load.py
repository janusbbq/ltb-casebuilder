"""Regional hourly demand: read, check coverage, slice a calendar period, scale.

Pure data layer: depends on numpy / pandas only. It must not import ``ams`` or
anything from LLM_LTB's ``agent`` package (enforced by a test), so it can move
to its own package unchanged.

Source: the EIA Hourly Electric Grid Monitor exports shipped in
ENLITEN-Grid-Econ-Data (``Source Data/Load/Source_Load_<REGION>.csv``; UTC, MW).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DATA_REPO_URL = "https://github.com/enliten/ENLITEN-Grid-Econ-Data"
DATA_ENV = "ENLITEN_DATA_DIR"
LOAD_SUBDIR = os.path.join("Source Data", "Load")


class CaseDataError(ValueError):
    """A request the data cannot satisfy (missing files, uncovered period, bad values).
    The message is written for the user."""


@dataclass(frozen=True)
class RegionInfo:
    code: str
    label: str
    filename: str
    tz: str            # IANA zone used for 'local' slicing
    tz_note: str = ""


REGIONS: Dict[str, RegionInfo] = {
    "NE": RegionInfo("NE", "New England (ISO-NE)", "Source_Load_NE.csv", "America/New_York"),
    "NY": RegionInfo("NY", "New York (NYISO)", "Source_Load_NY.csv", "America/New_York"),
    "CA": RegionInfo("CA", "California", "Source_Load_CA.csv", "America/Los_Angeles"),
    "NW": RegionInfo("NW", "Northwest", "Source_Load_NW.csv", "America/Los_Angeles",
                     "region spans Pacific and Mountain time; Pacific is used"),
    "SW": RegionInfo("SW", "Southwest", "Source_Load_SW.csv", "America/Phoenix",
                     "region spans several zones; Arizona (no DST) is used"),
}

# Free-text names -> region code (used by the agent node's deterministic parser).
REGION_ALIASES: Dict[str, str] = {
    "new england": "NE", "newengland": "NE", "iso-ne": "NE", "isone": "NE", "iso ne": "NE", "新英格兰": "NE",
    "new york": "NY", "nyiso": "NY", "纽约": "NY",
    "california": "CA", "caiso": "CA", "加州": "CA",
    "northwest": "NW", "西北": "NW",
    "southwest": "SW", "西南": "SW",
}


# --------------------------------------------------------------------------- #
# locating and reading the source data
# --------------------------------------------------------------------------- #
def _package_repo_root() -> str:
    # .../ltb-casebuilder/casebuilder/casedata/regional_load.py -> .../ltb-casebuilder
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_data_dir(data_dir: Optional[str] = None) -> str:
    """Return the root of an ENLITEN-Grid-Econ-Data checkout.

    Order: explicit argument, ``$ENLITEN_DATA_DIR``, ``./ENLITEN-Grid-Econ-Data``,
    ``./data/ENLITEN-Grid-Econ-Data``, a checkout next to this package's repository.
    """
    candidates = [
        data_dir,
        os.environ.get(DATA_ENV),
        os.path.join(os.getcwd(), "ENLITEN-Grid-Econ-Data"),
        os.path.join(os.getcwd(), "data", "ENLITEN-Grid-Econ-Data"),
        os.path.join(os.path.dirname(_package_repo_root()), "ENLITEN-Grid-Econ-Data"),
    ]
    tried = []
    for c in candidates:
        if not c:
            continue
        c = os.path.abspath(os.path.expanduser(c))
        tried.append(c)
        if os.path.isdir(os.path.join(c, LOAD_SUBDIR)):
            return c
    raise CaseDataError(
        "The regional load data was not found. Clone the dataset and point to it:\n"
        f"  git clone {DATA_REPO_URL}.git\n"
        f"  export {DATA_ENV}=/path/to/ENLITEN-Grid-Econ-Data\n"
        f"(looked for '{LOAD_SUBDIR}' under: {tried or ['<no candidate>']})"
    )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_ROW_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2})H,(.*)$")
_SERIES_RE = re.compile(r"Series ID:\s*([A-Za-z0-9_.\-]+)\s*(\w+)?")


@dataclass
class RegionalLoad:
    region: RegionInfo
    series: pd.Series                 # hourly MW, UTC tz-aware index, ascending
    path: str
    sha256: str
    title: str = ""
    url: str = ""
    series_id: str = ""
    unit: str = ""
    source: str = ""
    retrieved: str = ""               # the export's own timestamp line (time of day only)


@lru_cache(maxsize=8)
def _read_csv_cached(path: str, mtime: float) -> Tuple[pd.Series, Dict[str, str]]:
    header: List[str] = []
    stamps: List[str] = []
    values: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = _ROW_RE.match(line)
            if m:
                mo, d, y, h, v = m.groups()
                stamps.append(f"{y}-{int(mo):02d}-{int(d):02d} {int(h):02d}:00")
                values.append(v.strip())
            elif not stamps and line.strip():
                header.append(line.strip())
    if not stamps:
        raise CaseDataError(f"{path}: no hourly rows of the form 'MM/DD/YYYY HHH,<MW>' found")
    s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float),
                  index=pd.to_datetime(stamps, format="%Y-%m-%d %H:%M").tz_localize("UTC"))
    if s.index.has_duplicates:
        raise CaseDataError(f"{path}: duplicate hourly timestamps in the source file")
    s = s.sort_index()
    meta = {"title": header[0] if header else "", "url": "", "series_id": "", "unit": "",
            "source": "", "retrieved": ""}
    for h in header:
        if h.startswith("http"):
            meta["url"] = h
        elif h.lower().startswith("source:"):
            meta["source"] = h.split(":", 1)[1].strip()
        elif "GMT" in h:
            meta["retrieved"] = h
        m = _SERIES_RE.search(h)
        if m:
            meta["series_id"], meta["unit"] = m.group(1), (m.group(2) or "")
    if "utc" not in meta["title"].lower():
        raise CaseDataError(f"{path}: header does not say the series is in UTC ({meta['title']!r}); "
                            "refusing to guess the time base")
    return s, meta


def read_regional_load(region: str, data_dir: Optional[str] = None) -> RegionalLoad:
    """Read one region's hourly demand (MW, UTC) and its header metadata."""
    if region not in REGIONS:
        raise CaseDataError(f"unknown region {region!r}; choose one of {sorted(REGIONS)}")
    info = REGIONS[region]
    path = os.path.join(find_data_dir(data_dir), LOAD_SUBDIR, info.filename)
    if not os.path.isfile(path):
        raise CaseDataError(f"load file for region {region} not found: {path}")
    s, meta = _read_csv_cached(path, os.path.getmtime(path))
    return RegionalLoad(region=info, series=s, path=path, sha256=sha256_file(path), **meta)


def localized(rl: RegionalLoad, timezone: str) -> pd.Series:
    """The series indexed in the region's local time ('local') or UTC ('utc')."""
    tz = rl.region.tz if timezone == "local" else "UTC"
    s = rl.series.copy()
    s.index = s.index.tz_convert(tz)
    return s


def _hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int(round((end - start) / pd.Timedelta(hours=1)))


def _count(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int(((s.index >= start) & (s.index < end)).sum())


def coverage_hint(rl: RegionalLoad, timezone: str) -> str:
    s = localized(rl, timezone)
    return f"the {rl.region.code} record covers {s.index[0]:%Y-%m-%d %H:%M} to {s.index[-1]:%Y-%m-%d %H:%M}"


def available_periods(region: str, data_dir: Optional[str] = None,
                      timezone: str = "local") -> Dict[str, object]:
    """Coverage of a region's record: first/last hour and the calendar months / years that are
    complete (every hour present and finite) in the chosen time base."""
    rl = read_regional_load(region, data_dir)
    s = localized(rl, timezone)
    ok = s.dropna()
    months, years = [], []
    for y in range(s.index[0].year, s.index[-1].year + 1):
        y0 = pd.Timestamp(year=y, month=1, day=1, tz=s.index.tz)
        y1 = y0 + pd.DateOffset(years=1)
        if _count(ok, y0, y1) == _hours(y0, y1):
            years.append(y)
        for mth in range(1, 13):
            m0 = pd.Timestamp(year=y, month=mth, day=1, tz=s.index.tz)
            m1 = m0 + pd.DateOffset(months=1)
            if _count(ok, m0, m1) == _hours(m0, m1):
                months.append(f"{y}-{mth:02d}")
    return {"region": region, "first": str(s.index[0]), "last": str(s.index[-1]),
            "complete_months": months, "complete_years": years}


# --------------------------------------------------------------------------- #
# periods
# --------------------------------------------------------------------------- #
def period_label(year: int, month: Optional[int] = None, start_day: int = 1,
                 days: Optional[int] = None) -> str:
    """Human-readable period, e.g. '2020-07', '2020-07 (7 days from day 1)', '2020'."""
    base = f"{year}-{month:02d}" if month else f"{year}"
    return f"{base} ({days} days from day {start_day})" if days else base


def slice_period(rl: RegionalLoad, year: int, month: Optional[int] = None, start_day: int = 1,
                 days: Optional[int] = None, timezone: str = "local") -> pd.Series:
    """Hourly MW for the period; raises unless every hour is present, finite and positive."""
    label = period_label(year, month, start_day, days)
    s = localized(rl, timezone)
    tz = s.index.tz
    try:
        start = pd.Timestamp(year=year, month=month or 1, day=start_day, tz=tz)
    except ValueError as exc:
        raise CaseDataError(f"invalid start date {year}-{month or 1}-{start_day}: {exc}")
    if days:
        end = start + pd.DateOffset(days=days)
    elif month:
        end = pd.Timestamp(year=year, month=month, day=1, tz=tz) + pd.DateOffset(months=1)
    else:
        end = start + pd.DateOffset(years=1)
    if end <= start:
        raise CaseDataError(f"empty period: start {start} is not before end {end}")
    sel = s[(s.index >= start) & (s.index < end)]
    expected = _hours(start, end)
    n_nan = int(sel.isna().sum())
    if len(sel) != expected or n_nan:
        raise CaseDataError(
            f"Period {label} is not fully covered: need {expected} hourly values "
            f"from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} ({tz}), found {len(sel) - n_nan}. "
            f"Note: {coverage_hint(rl, timezone)}.")
    if (sel <= 0).any():
        bad = sel[sel <= 0]
        raise CaseDataError(f"Period {label} has {len(bad)} non-positive demand values "
                            f"(first at {bad.index[0]}); the source record is not usable for this period.")
    return sel


def reference_peak(rl: RegionalLoad, year: int, ref_year: Optional[int] = None,
                   timezone: str = "local") -> Tuple[float, int, str]:
    """Regional peak (MW) of the reference year, the year actually used, and a note.

    The reference year is ``ref_year`` or ``year``. If that calendar year is not fully covered
    and ``ref_year`` was not pinned, the closest fully covered year is used and reported.
    """
    s = localized(rl, timezone).dropna()
    tz = s.index.tz

    def full(y: int) -> bool:
        y0 = pd.Timestamp(year=y, month=1, day=1, tz=tz)
        return _count(s, y0, y0 + pd.DateOffset(years=1)) == _hours(y0, y0 + pd.DateOffset(years=1))

    def peak(y: int) -> float:
        y0 = pd.Timestamp(year=y, month=1, day=1, tz=tz)
        return float(s[(s.index >= y0) & (s.index < y0 + pd.DateOffset(years=1))].max())

    want = ref_year or year
    if full(want):
        return peak(want), want, ""
    if ref_year is not None:
        raise CaseDataError(f"ref_year {want} is not fully covered; {coverage_hint(rl, timezone)}.")
    years = [y for y in range(s.index[0].year, s.index[-1].year + 1) if full(y)]
    if not years:
        raise CaseDataError(f"no calendar year is fully covered; {coverage_hint(rl, timezone)}.")
    used = min(years, key=lambda y: (abs(y - want), -y))
    return peak(used), used, (f"{want} is not a fully covered calendar year, so the scale reference "
                              f"is the {used} regional peak instead")


# --------------------------------------------------------------------------- #
# scaling and checks
# --------------------------------------------------------------------------- #
def scale_and_distribute(regional_mw: Sequence[float], ref_peak_mw: float, peak_mw: float,
                         weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """``total(t) = peak_mw * R(t) / ref_peak`` and ``load_i(t) = total(t) * w_i / sum(w)``.

    Returns ``(total, per_load)`` in MW with shapes ``(n,)`` and ``(n, len(weights))``.
    """
    w = np.asarray(weights, dtype=float)
    if (w <= 0).any():
        raise CaseDataError("participation weights must be positive")
    total = float(peak_mw) * np.asarray(regional_mw, dtype=float) / float(ref_peak_mw)
    return total, np.outer(total, w / w.sum())


def check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def source_checks(sel: pd.Series) -> List[Dict[str, str]]:
    """Data-side checks of a sliced period: coverage (already enforced) and outliers."""
    out = [check("coverage", "ok", f"{len(sel)} of {len(sel)} hourly values present "
                                   f"({sel.index[0]:%Y-%m-%d %H:%M} to {sel.index[-1]:%Y-%m-%d %H:%M} {sel.index.tz})")]
    med = float(np.median(sel))
    odd = sel[(sel < 0.3 * med) | (sel > 3.0 * med)]
    out.append(check("source_values", "warn" if len(odd) else "ok",
                     (f"{len(odd)} hourly values are outside 0.3x-3x the period median "
                      f"({med:.0f} MW); first at {odd.index[0]}" if len(odd)
                      else f"all values within 0.3x-3x the period median ({med:.0f} MW)")))
    return out


def provenance(rl: RegionalLoad, timezone: str) -> Dict[str, object]:
    """Provenance block for a manifest (no case-specific fields)."""
    return {
        "parameter": "load_p (per load, per hour)",
        "source_organization": rl.source or "U.S. Energy Information Administration",
        "source_title": rl.title, "source_url": rl.url, "series_id": rl.series_id,
        "source_unit": f"{rl.unit} per hour (= MW average)" if rl.unit else "MW",
        "source_time_base": "UTC",
        "source_file": {"path": rl.path, "sha256": rl.sha256, "export_time_of_day": rl.retrieved},
        "obtained_via": DATA_REPO_URL,
        "geographic_coverage": rl.region.label,
        "slicing_time_base": rl.region.tz if timezone == "local" else "UTC",
    }
