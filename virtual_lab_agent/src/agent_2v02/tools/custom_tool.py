"""Deterministic tools for the Virtual Lab Agent.

The chemistry reasoning is done by the LLM, but the *economics* should not be
hallucinated. `ManufacturingCostCalculatorTool` turns the agent's estimated
inputs (raw-material cost, yield, energy, labor) into a grounded per-kg cost
breakdown with a fixed formula, so the financial numbers in the report are
reproducible arithmetic rather than guesses.
"""

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ManufacturingCostInput(BaseModel):
    """Inputs for the per-kilogram manufacturing-cost estimate."""

    raw_materials_usd_per_kg: float = Field(
        ..., ge=0, description="Cost of feedstock/reagents per kg of *product*, before yield losses."
    )
    yield_percent: float = Field(
        ..., gt=0, le=100, description="Isolated reaction yield (%). Lower yield => more feedstock per kg of product."
    )
    energy_usd_per_kg: float = Field(
        0.0, ge=0, description="Energy/utilities cost per kg of product (heating, vacuum, cooling)."
    )
    labor_overhead_usd_per_kg: float = Field(
        0.0, ge=0, description="Labor, equipment depreciation, and overhead per kg of product."
    )


class ManufacturingCostCalculatorTool(BaseTool):
    name: str = "manufacturing_cost_calculator"
    description: str = (
        "Compute a grounded, reproducible per-kilogram manufacturing cost for a material. "
        "Call this when you have estimated the feedstock cost, reaction yield, energy and "
        "labor/overhead. It scales raw-material cost by the yield loss and returns a full "
        "breakdown plus a cost tier (very_low / low / moderate / high / very_high). "
        "Use the returned numbers verbatim in the report — do not recompute them yourself."
    )
    args_schema: Type[BaseModel] = ManufacturingCostInput

    def _run(
        self,
        raw_materials_usd_per_kg: float,
        yield_percent: float,
        energy_usd_per_kg: float = 0.0,
        labor_overhead_usd_per_kg: float = 0.0,
    ) -> str:
        yield_fraction = yield_percent / 100.0
        # Feedstock actually consumed per kg of *isolated* product grows as yield drops.
        effective_raw = raw_materials_usd_per_kg / yield_fraction
        total = effective_raw + energy_usd_per_kg + labor_overhead_usd_per_kg

        if total < 5:
            tier = "very_low"
        elif total < 25:
            tier = "low"
        elif total < 100:
            tier = "moderate"
        elif total < 500:
            tier = "high"
        else:
            tier = "very_high"

        return (
            "Manufacturing cost estimate (per kg of product):\n"
            f"  raw_materials (yield-adjusted): ${effective_raw:.2f}\n"
            f"  energy:                         ${energy_usd_per_kg:.2f}\n"
            f"  labor_overhead:                 ${labor_overhead_usd_per_kg:.2f}\n"
            f"  TOTAL:                          ${total:.2f}/kg\n"
            f"  cost_tier:                      {tier}\n"
            f"  (assumed yield: {yield_percent:.0f}% -> feedstock multiplier {1/yield_fraction:.2f}x)"
        )
