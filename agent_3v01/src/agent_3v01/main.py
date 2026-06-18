#!/usr/bin/env python
"""Standalone entry point for the Procurement & Sourcing Agent (Agent 3).

Reads a VirtualLabReport (Agent 2's output) and emits an
ExecutiveProductProposal to output/product_proposal.json.

Usage:
    uv run agent_3v01 [path/to/virtual_lab_report.json]

With no argument it runs against a built-in sample VirtualLabReport so the
pipeline works end-to-end out of the box.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_3v01.pipeline import run_procurement_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output" / "product_proposal.json"

# A representative Agent 2 (Virtual Lab Agent) output. Used as the default input
# so the pipeline runs end-to-end without a real upstream report.
SAMPLE_VIRTUAL_LAB_REPORT = {
    "status": "SUCCESS",
    "material_formula": "Poly(dimethylsiloxane-co-diphenylsiloxane)",
    "reaction_simulation": {
        "synthesis_route": "Hydrosilylation crosslinking at 150°C under Pt catalyst",
        "feasibility": "FEASIBLE",
        "predicted_yield_percent": 78.0,
        "reaction_conditions": [
            {"parameter": "temperature", "value": "150 °C", "rationale": "Activates Pt-catalyzed hydrosilylation"},
            {"parameter": "catalyst loading", "value": "0.5 mol% Pt", "rationale": "Balances rate vs. PGM cost"},
        ],
        "byproducts": ["trace cyclic siloxanes"],
        "hazards": ["exotherm during crosslinking"],
        "scale_up_risk": "MEDIUM",
        "reasoning": "Well-established siloxane chemistry; Pt catalyst is the main cost/scale concern.",
    },
    "manufacturing_cost": {
        "production_scale": "pilot",
        "raw_materials_usd_per_kg": 14.0,
        "energy_usd_per_kg": 3.0,
        "labor_overhead_usd_per_kg": 6.0,
        "total_cost_usd_per_kg": 23.0,
        "cost_tier": "low",
        "key_cost_drivers": ["platinum catalyst", "siloxane precursor", "energy for crosslinking"],
    },
    "manufacturability_verdict": "GO",
    "confidence": 0.81,
    "recommendations_for_procurement": [
        "Secure a stable platinum catalyst supply; PGM price volatility is the key risk.",
        "Qualify at least two siloxane precursor suppliers.",
    ],
    "assumptions": ["Pilot-scale economics", "No solvent recovery credit applied"],
}


def _load_report() -> dict:
    """Resolve the Agent 2 report: a JSON file path argv, else the sample."""
    if len(sys.argv) > 1 and sys.argv[1]:
        path = Path(sys.argv[1])
        if path.exists():
            return json.loads(path.read_text())
        # Treat the argument as a raw JSON string if it isn't a file.
        return json.loads(sys.argv[1])
    return SAMPLE_VIRTUAL_LAB_REPORT


def run():
    """Run the procurement pipeline on a VirtualLabReport."""
    load_dotenv()  # MODEL + OPENAI_API_KEY (or ANTHROPIC_API_KEY)

    report = _load_report()
    proposal = run_procurement_analysis(json.dumps(report, indent=2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(proposal.model_dump_json(indent=2))

    print("\n===== EXECUTIVE PRODUCT PROPOSAL (Agent 3) =====")
    print(proposal.model_dump_json(indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")
    return proposal


if __name__ == "__main__":
    run()
