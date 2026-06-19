"""Structured I/O contracts for the Procurement & Sourcing Agent (Agent 3).

Agent 1 (Material Theory) proposes a material. Agent 2 (Virtual Lab) simulates
its synthesis + manufacturing cost and emits a `VirtualLabReport`. Agent 3
consumes that report, checks real-world supply-chain availability for the
feedstocks, computes ROI, and emits an `ExecutiveProductProposal` — the final,
human-facing deliverable of the pipeline.

Keeping this as an explicit schema is what lets the three agents exchange
context reliably instead of passing free-form prose around.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeedstockSourcing(BaseModel):
    """Supply-chain assessment for a single raw material / feedstock."""

    material: str = Field(..., description="Feedstock / reagent / element being sourced")
    availability: Literal[
        "ABUNDANT", "AVAILABLE", "CONSTRAINED", "SCARCE", "UNAVAILABLE"
    ] = Field(..., description="Current real-world market availability")
    market_price_usd_per_kg: float = Field(
        ..., ge=0, description="Indicative spot price per kg from the sourcing lookup"
    )
    estimated_lead_time_weeks: float = Field(
        ..., ge=0, description="Typical procurement lead time in weeks"
    )
    suppliers_identified: int = Field(
        ..., ge=0, description="Number of qualified suppliers found (supplier diversity)"
    )
    supply_risk: Literal["LOW", "MEDIUM", "HIGH"]
    notes: str = Field(..., description="Concentration, geopolitics, substitutes, etc.")


class SupplyChainAssessment(BaseModel):
    """Aggregate sourcing picture across every feedstock the material needs."""

    feedstocks: list[FeedstockSourcing]
    overall_supply_risk: Literal["LOW", "MEDIUM", "HIGH"]
    single_points_of_failure: list[str] = Field(
        default_factory=list,
        description="Feedstocks with one/no supplier or that are scarce/unavailable",
    )
    geographic_concentration: str = Field(
        ..., description="Where supply is concentrated and the geopolitical exposure"
    )


class ROIAnalysis(BaseModel):
    """Grounded, reproducible product economics (numbers come from the ROI tool)."""

    manufacturing_cost_usd_per_kg: float = Field(..., ge=0)
    target_selling_price_usd_per_kg: float = Field(..., ge=0)
    gross_margin_percent: float = Field(..., description="(price - cost) / price * 100")
    annual_production_volume_kg: float = Field(..., ge=0)
    annual_revenue_usd: float = Field(..., ge=0)
    annual_gross_profit_usd: float
    upfront_investment_usd: float = Field(..., ge=0)
    payback_period_years: float = Field(
        ..., description="upfront_investment / annual_gross_profit (-1 if never)"
    )
    roi_percent: float = Field(..., description="ROI over the analysis horizon")
    horizon_years: int = Field(..., ge=1, description="Horizon used for the ROI figure")
    assumptions: list[str] = Field(default_factory=list)


class ExecutiveProductProposal(BaseModel):
    """The full Agent 3 deliverable — the final pipeline output for executives."""

    status: Literal["SUCCESS", "FAILURE"] = "SUCCESS"
    material_formula: str
    recommendation: Literal["GREENLIGHT", "PILOT", "HOLD", "REJECT"] = Field(
        ...,
        description=(
            "GREENLIGHT = commercialize; PILOT = fund a pilot first; "
            "HOLD = revisit when conditions change; REJECT = do not pursue"
        ),
    )
    executive_summary: str = Field(
        ..., description="2-4 sentence summary a non-technical executive can act on"
    )
    supply_chain: SupplyChainAssessment
    roi: ROIAnalysis
    strategic_risks: list[str] = Field(
        default_factory=list, description="Top business/technical/supply risks"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Concrete recommended actions"
    )
    confidence: float = Field(..., ge=0, le=1)
