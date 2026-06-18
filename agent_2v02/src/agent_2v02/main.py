#!/usr/bin/env python
import json
import sys
import warnings
from pathlib import Path

from agent_2v02.crew import Agent2V02

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# A representative Agent 1 (Material Theory Agent) output. Used as the default
# input when no real proposal is supplied, so the crew runs end-to-end out of
# the box. Replace at runtime by passing a JSON file path:  run_crew path.json
SAMPLE_MATERIAL_PROPOSAL = {
    "status": "SUCCESS",
    "material": {
        "formula": "Poly(dimethylsiloxane-co-diphenylsiloxane)",
        "iupac_name": "Poly[oxy(dimethylsilylene)] crosslinked with diphenyl groups",
        "material_class": "polymer",
        "composition": {"Si": "38%", "O": "22%", "C": "32%", "H": "8%"},
        "synthesis_route": "Hydrosilylation crosslinking at 150°C under Pt catalyst",
        "pubchem_verified": False,
    },
    "predicted_properties": {
        "tensile_strength_MPa": 35,
        "density_g_cm3": 1.12,
        "melting_point_C": None,
        "thermal_conductivity_W_mK": 0.18,
        "electrical_resistivity_ohm_m": 1e14,
        "notes": "Decomposes above 300°C rather than melting",
    },
    "scientific_reasoning": {
        "theories_used": ["glass transition temperature", "siloxane backbone chemistry", "crosslink density"],
        "design_principles": ["aromatic rings for thermal stability", "siloxane for flexibility"],
        "hypotheses": [
            {
                "id": 1,
                "reasoning": "High aromatic content improves thermal stability via pi-pi stacking",
                "key_features": ["crosslinked siloxane backbone", "diphenyl co-monomer"],
                "predicted_strengths": ["heat resistance > 200°C", "low toxicity"],
                "predicted_weaknesses": ["moderate tensile strength"],
            }
        ],
    },
    "validation": {
        "overall": "PASS",
        "checks": {
            "oxidation_states": "PASS",
            "toxicity": "PASS",
            "cost": "PASS",
            "meets_strength": "PASS",
            "meets_temperature": "PASS",
        },
        "confidence": 0.82,
        "cost_analysis": {
            "overall_tier": "low",
            "breakdown": {"Si": "low", "C": "very_low", "O": "very_low"},
        },
    },
    "original_requirement": "Need a heat-resistant polymer adhesive",
    "parsed_requirement": {"material_type": "polymer", "temperature_resistance": 200},
    "total_iterations": 1,
    "rejection_history": [],
    "tool_evidence": {
        "wikipedia": "Polydimethylsiloxane (PDMS) is a silicon-based organic polymer",
        "literature": ["[Wikipedia] PDMS is widely used", "[PubChem] Formula: C2H6OSi"],
    },
}


def _load_proposal() -> dict:
    """Resolve the Agent 1 proposal: a JSON file path argv, else the sample."""
    if len(sys.argv) > 1 and sys.argv[1]:
        path = Path(sys.argv[1])
        if path.exists():
            return json.loads(path.read_text())
        # Treat the argument as a raw JSON string if it isn't a file.
        return json.loads(sys.argv[1])
    return SAMPLE_MATERIAL_PROPOSAL


def _inputs(proposal: dict) -> dict:
    # The proposal is passed as a compact JSON string into the task templates'
    # {material_proposal} placeholder.
    return {"material_proposal": json.dumps(proposal, indent=2)}


def run():
    """Run the Virtual Lab crew on a material proposal."""
    try:
        result = Agent2V02().crew().kickoff(inputs=_inputs(_load_proposal()))
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

    print("\n===== VIRTUAL LAB REPORT (for Procurement Agent) =====")
    if result.pydantic is not None:
        print(result.pydantic.model_dump_json(indent=2))
    else:
        print(result.raw)
    return result


def train():
    """Train the crew for a given number of iterations."""
    inputs = _inputs(SAMPLE_MATERIAL_PROPOSAL)
    try:
        Agent2V02().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Agent2V02().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return the results."""
    inputs = _inputs(SAMPLE_MATERIAL_PROPOSAL)
    try:
        Agent2V02().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run the crew with a trigger payload — e.g. Agent 1's emitted proposal.

    The payload may be the proposal directly, or an object that wraps it under
    a 'material' / 'material_proposal' / 'proposal' key.
    """
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Pass the Agent 1 proposal as a JSON argument.")

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    proposal = payload
    if isinstance(payload, dict):
        for key in ("material_proposal", "proposal", "crewai_trigger_payload"):
            if key in payload:
                proposal = payload[key]
                break

    try:
        result = Agent2V02().crew().kickoff(inputs=_inputs(proposal))
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


if __name__ == "__main__":
    run()
