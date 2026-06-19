"""Structured I/O contracts for the Virtual Lab Agent (Agent 2).

Agent 1 (Material Theory) emits a material proposal. Agent 2 consumes it,
simulates the synthesis reaction + yield and estimates manufacturing cost,
then emits a `VirtualLabReport` that Agent 3 (Procurement & Sourcing)
consumes. Keeping this as an explicit schema is what lets the three agents
exchange context reliably instead of passing free-form prose around.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReactionCondition(BaseModel):
    """A single controllable process parameter for the simulated synthesis."""

    parameter: str = Field(..., description="e.g. 'temperature', 'pressure', 'catalyst loading'")
    value: str = Field(..., description="Simulated set-point, e.g. '150 °C', '2 atm', '0.5 mol% Pt'")
    rationale: str = Field(..., description="Why this set-point was chosen for the reaction")


class ReactionSimulation(BaseModel):
    """Outcome of the virtual reaction / process simulation."""

    synthesis_route: str = Field(..., description="The reaction route taken from the proposal")
    feasibility: Literal["FEASIBLE", "CHALLENGING", "INFEASIBLE"]
    predicted_yield_percent: float = Field(
        ..., ge=0, le=100, description="Estimated isolated yield of the target material"
    )
    reaction_conditions: list[ReactionCondition]
    byproducts: list[str] = Field(default_factory=list, description="Expected side-products / waste streams")
    hazards: list[str] = Field(default_factory=list, description="Process safety / handling concerns")
    scale_up_risk: Literal["LOW", "MEDIUM", "HIGH"]
    reasoning: str = Field(..., description="Mechanistic justification for yield and feasibility call")


class CostBreakdown(BaseModel):
    """Per-kilogram manufacturing economics at the assessed production scale."""

    production_scale: Literal["lab", "pilot", "industrial"]
    raw_materials_usd_per_kg: float = Field(..., ge=0)
    energy_usd_per_kg: float = Field(..., ge=0)
    labor_overhead_usd_per_kg: float = Field(..., ge=0)
    total_cost_usd_per_kg: float = Field(..., ge=0)
    cost_tier: Literal["very_low", "low", "moderate", "high", "very_high"]
    key_cost_drivers: list[str] = Field(default_factory=list)


class VirtualLabReport(BaseModel):
    """The full Agent 2 deliverable — handed to Agent 3 (Procurement)."""

    status: Literal["SUCCESS", "FAILURE"] = "SUCCESS"
    material_formula: str
    reaction_simulation: ReactionSimulation
    manufacturing_cost: CostBreakdown
    manufacturability_verdict: Literal["GO", "CONDITIONAL", "NO_GO"]
    confidence: float = Field(..., ge=0, le=1)
    recommendations_for_procurement: list[str] = Field(
        default_factory=list,
        description="Concrete next-steps / sourcing flags for the downstream procurement agent",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Simulation assumptions a reviewer should know about"
    )
