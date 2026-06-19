"""
nodes/material_generator.py — Node 4: Material Generator

Takes the best hypothesis from the Scientific Reasoner and converts it
into a concrete material specification: chemical formula, composition,
predicted properties, and synthesis route.

Uses PubChem to verify the formula exists or is closely related to
known compounds, grounding the output in reality.
"""

from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from state import MaterialTheoryState
from utils import get_rate_limited_llm, safe_json_parse
from tools import pubchem_lookup

# ── Prompt ────────────────────────────────────────────────────────────────────

GENERATOR_PROMPT = PromptTemplate(
    input_variables=["hypothesis", "requirements", "plan"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a materials design specialist. Convert a scientific hypothesis into
a concrete, realizable material specification with a valid chemical formula.

Rules:
- The formula must be chemically valid (correct valences, charge neutrality)
- Use standard chemical notation (e.g. Al2O3, SiC, PDMS)
- Polymers can use IUPAC-style names (e.g. Poly(dimethylsiloxane))
- Predicted properties must be physically plausible
- Synthesis route must be industrially feasible
<|eot_id|><|start_header_id|>user<|end_header_id|>

=== BEST HYPOTHESIS ===
{hypothesis}

=== ENGINEERING REQUIREMENTS ===
{requirements}

=== SCIENTIFIC PLAN ===
{plan}

Output ONLY valid JSON. No preamble. No markdown.

JSON schema:
{{
  "formula": "<chemical formula or IUPAC polymer name>",
  "iupac_name": "<full IUPAC name>",
  "material_class": "<ceramic|polymer|metal|alloy|composite|semiconductor>",
  "composition": {{
    "<element symbol>": "<percentage or ratio>"
  }},
  "predicted_properties": {{
    "tensile_strength_MPa": <number or null>,
    "density_g_cm3": <number or null>,
    "melting_point_C": <number or null>,
    "thermal_conductivity_W_mK": <number or null>,
    "electrical_resistivity_ohm_m": <number or null>,
    "notes": "<additional property notes>"
  }},
  "synthesis_route": "<brief synthesis description>",
  "hypothesis_id": <id of the hypothesis used>
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)

# ── Node function ─────────────────────────────────────────────────────────────

def material_generator(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Node 4: Convert the best hypothesis into a concrete material specification.
    Attempts PubChem verification of the generated formula.
    Updates state["candidate_material"].
    """
    print("\n[Node 4] Material Generator running...")

    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        print("[Node 4] WARNING: No hypotheses found, using placeholder.")
        return {
            **state,
            "candidate_material": {
                "formula": "Unknown",
                "iupac_name": "Unknown",
                "material_class": "unknown",
                "composition": {},
                "predicted_properties": {},
                "synthesis_route": "Not determined",
                "hypothesis_id": 0,
            },
        }

    # Use the first (best) hypothesis
    best_hypothesis = hypotheses[0]

    llm = get_rate_limited_llm()
    chain = GENERATOR_PROMPT | llm

    raw_output = chain.invoke({
        "hypothesis": json.dumps(best_hypothesis, indent=2),
        "requirements": json.dumps(state.get("parsed_requirement", {}), indent=2),
        "plan": json.dumps(state.get("scientific_plan", {}), indent=2),
    })

    fallback = {
        "formula": "Al2O3",
        "iupac_name": "Aluminium oxide",
        "material_class": "ceramic",
        "composition": {"Al": "52.9%", "O": "47.1%"},
        "predicted_properties": {
            "tensile_strength_MPa": 300,
            "density_g_cm3": 3.95,
            "melting_point_C": 2072,
            "notes": "Fallback material — review hypotheses.",
        },
        "synthesis_route": "Conventional sintering of alumina powder",
        "hypothesis_id": 1,
    }

    candidate = safe_json_parse(raw_output.content, fallback)
    formula = candidate.get("formula", "")

    # ── PubChem verification (best-effort) ──────────────────────────────────
    if formula and formula != "Unknown":
        print(f"[Node 4] Verifying formula '{formula}' on PubChem...")
        pubchem_result = pubchem_lookup(formula)
        if pubchem_result.get("found"):
            candidate["pubchem_verified"] = True
            candidate["pubchem_cid"] = pubchem_result.get("cid")
            candidate["pubchem_mw"] = pubchem_result.get("molecular_weight")
            print(f"[Node 4] ✅ PubChem found: CID={pubchem_result.get('cid')}")
        else:
            candidate["pubchem_verified"] = False
            print(f"[Node 4] ℹ️  Not in PubChem (novel material or polymer — expected)")

    print(f"[Node 4] Candidate material: {formula}")
    return {**state, "candidate_material": candidate}

