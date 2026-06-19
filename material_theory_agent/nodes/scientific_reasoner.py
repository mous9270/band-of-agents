"""
nodes/scientific_reasoner.py — Node 3: Scientific Reasoner

This is the core innovation node. It:
  1. Queries real scientific APIs (Wikipedia, PubChem, Periodic Table)
     to ground its reasoning in factual data.
  2. Uses an LLM to reason from first principles — atomic interactions,
     bonding, thermodynamics, mechanical theory.
  3. Incorporates the rejection history to avoid repeating failed ideas.
  4. Generates 1–3 scientifically grounded hypotheses.

The tool data is injected into the prompt so the LLM reasons from
real facts, not hallucinated data.
"""

from __future__ import annotations

import json
from typing import List

from langchain_core.prompts import PromptTemplate
from state import MaterialTheoryState, Hypothesis
from utils import get_rate_limited_llm, safe_json_parse
from tools import (
    periodic_table_lookup,
    literature_search,
)

# ── Prompt ────────────────────────────────────────────────────────────────────

REASONER_PROMPT = PromptTemplate(
    input_variables=[
        "requirements",
        "scientific_plan",
        "periodic_data",
        "literature_data",
        "rejection_history",
        "iteration",
    ],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior theoretical materials scientist with 30 years of experience.
Reason from first principles using atomic interactions, bonding theory,
electronic structure, crystal chemistry, thermodynamics, and mechanical theory.

NEVER hallucinate impossible compounds. Base every decision on real chemistry.
Explain EVERY design choice with scientific reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>

=== ENGINEERING REQUIREMENTS ===
{requirements}

=== SCIENTIFIC PLAN ===
{scientific_plan}

=== REAL ELEMENT DATA (from Periodic Table) ===
{periodic_data}

=== LITERATURE / DATABASE EVIDENCE ===
{literature_data}

=== REJECTION HISTORY (avoid these) ===
Iteration: {iteration}
{rejection_history}

=== YOUR TASK ===
Generate 1-3 scientifically grounded material hypotheses.
Each hypothesis must:
1. Be based on real chemistry (consistent with the element data above)
2. Avoid anything in the rejection history
3. Include clear first-principles reasoning
4. Predict strengths AND weaknesses honestly

Output ONLY valid JSON. No preamble. No markdown.

JSON schema:
{{
  "hypotheses": [
    {{
      "id": 1,
      "reasoning": "<detailed first-principles scientific reasoning>",
      "key_features": ["<feature 1>", "<feature 2>"],
      "predicted_strengths": ["<strength 1>", ...],
      "predicted_weaknesses": ["<weakness 1>", ...]
    }}
  ]
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)

# ── Node function ─────────────────────────────────────────────────────────────

def scientific_reasoner(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Node 3: Reason from first principles and generate material hypotheses.
    Uses real API data; incorporates rejection history to avoid repeating failures.
    Updates state["hypotheses"] and state["tool_results"].
    """
    print(f"\n[Node 3] Scientific Reasoner running (iteration {state.get('iteration', 0)})...")

    plan = state.get("scientific_plan", {})
    req = state.get("parsed_requirement", {})
    rejection_history = state.get("rejection_history", [])
    iteration = state.get("iteration", 0)

    # ── Step 1: Call real scientific tools ──────────────────────────────────
    key_elements: List[str] = plan.get("key_elements", ["C", "Si", "O", "Al"])

    print(f"[Node 3] Fetching periodic table data for: {key_elements}")
    pt_data = periodic_table_lookup(key_elements)

    # Literature search on the material type
    material_type = req.get("material_type", "advanced material")
    application = req.get("application", "general")
    search_query = f"{material_type} {application} material science"

    print(f"[Node 3] Searching literature: '{search_query}'")
    lit_data = literature_search(search_query, material_type)

    # ── Step 2: Format tool data for prompt ─────────────────────────────────
    periodic_str = json.dumps(pt_data, indent=2)

    snippets = lit_data.get("snippets", [])
    lit_str = "\n".join(snippets) if snippets else "No literature found."

    rejection_str = (
        json.dumps(rejection_history, indent=2)
        if rejection_history
        else "No rejections yet — this is the first attempt."
    )

    # ── Step 3: Run LLM reasoning ───────────────────────────────────────────
    llm = get_rate_limited_llm()
    chain = REASONER_PROMPT | llm

    raw_output = chain.invoke({
        "requirements": json.dumps(req, indent=2),
        "scientific_plan": json.dumps(plan, indent=2),
        "periodic_data": periodic_str[:2000],    # trim to fit context
        "literature_data": lit_str[:1000],
        "rejection_history": rejection_str[:1000],
        "iteration": iteration,
    })

    fallback = {
        "hypotheses": [
            {
                "id": 1,
                "reasoning": (
                    f"Based on requirements for {material_type}, "
                    "a conventional approach using abundant elements is recommended."
                ),
                "key_features": ["stability", "manufacturability"],
                "predicted_strengths": ["cost-effective", "well-understood chemistry"],
                "predicted_weaknesses": ["may not meet all performance targets"],
            }
        ]
    }

    result = safe_json_parse(raw_output.content, fallback)
    hypotheses: List[Hypothesis] = result.get("hypotheses", fallback["hypotheses"])

    print(f"[Node 3] Generated {len(hypotheses)} hypothesis/hypotheses")

    tool_results = {
        "periodic_table_data": pt_data,
        "wikipedia_summary": lit_data.get("wikipedia", {}).get("summary", ""),
        "pubchem_data": lit_data.get("pubchem", {}),
        "literature_snippets": snippets,
    }

    return {
        **state,
        "hypotheses": hypotheses,
        "tool_results": tool_results,
    }

