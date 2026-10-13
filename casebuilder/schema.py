"""Spec for building a period-specific (month / year) multi-period case.

The LLM fills :class:`PeriodCaseSpec`; deterministic Python
(``casebuilder.period_case``) reads the regional hourly load record, scales and
distributes it, and writes the case through ``week_case.build_week_case``. Nothing here
imports ams: this module is pure data validation and is safe to hand to the LLM's
structured-output layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from casebuilder.casedata import period_label

# Regions with an hourly demand file in ENLITEN-Grid-Econ-Data/Source Data/Load.
RegionCode = Literal["NE", "NY", "CA", "NW", "SW"]


class PeriodCaseSpec(BaseModel):
    """One period-specific case: which base case, which region's load shape, which period."""

    model_config = ConfigDict(extra="forbid")

    base_case: str = Field(
        "ieee39_uced",
        description="Case catalog key or path of the base case. It must carry ED/UC slot tables "
                    "(a '*_uced' case or pjm5bus_demo). Default: ieee39_uced.")
    region: RegionCode = Field(
        "NE", description="Region whose hourly demand shapes the load: NE (New England), NY, CA, "
                          "NW (Northwest), SW (Southwest). Default NE, which matches IEEE 39-bus.")
    year: int = Field(..., ge=2000, le=2100, description="Calendar year of the period, e.g. 2020")
    month: Optional[int] = Field(
        None, ge=1, le=12, description="Calendar month 1-12. Omit to build the whole year.")
    start_day: int = Field(1, ge=1, le=31, description="First day of the month to include (default 1)")
    days: Optional[int] = Field(
        None, ge=1, le=366, description="Number of days to include. Omit for the whole month/year.")

    peak_mw: Optional[float] = Field(
        None, gt=0, allow_inf_nan=False,
        description="System load (MW) that the reference year's regional peak is mapped to. "
                    "Omit to use the base case's total load, which keeps the case feasible.")
    ref_year: Optional[int] = Field(
        None, ge=2000, le=2100,
        description="Year whose regional peak defines the scale. Omit to use `year` itself. Fix it "
                    "(e.g. 2020) when comparing years so that year-to-year differences survive.")
    timezone: Literal["local", "utc"] = Field(
        "local", description="'local' slices the period in the region's local calendar (default); "
                             "'utc' slices it in UTC, the time base of the source file.")

    @model_validator(mode="after")
    def _check(self) -> "PeriodCaseSpec":
        if self.month is None and self.start_day != 1:
            raise ValueError("start_day needs a month")
        return self

    def period_label(self) -> str:
        """Human-readable period, e.g. '2020-07' or '2020-07 (7 days from day 1)' or '2020'."""
        return period_label(self.year, self.month, self.start_day, self.days)

    def case_id(self, base_stem: str) -> str:
        """File stem of the built case, e.g. ``ieee39_uced_NE_2020_07``."""
        cid = f"{base_stem}_{self.region}_{self.year}"
        if self.month:
            cid += f"_{self.month:02d}"
        if self.days:
            cid += f"_d{self.start_day:02d}x{self.days}"
        if self.timezone == "utc":
            cid += "_utc"
        return cid
