"""
nodes/theory_planner.py — Node 2: Theory Planner

Decides which scientific domains and theories are relevant, which
elements to explore, and which to avoid — grounded in the parsed
engineering requirements.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from state import MaterialTheoryState
from utils import get_rate_limited_llm, safe_json_parse

# ── Prompt ────────────────────────────────────────────────────────────────────

PLANNER_PROMPT = PromptTemplate(
    input_variables=["parsed_requirement"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a materials science theory planner.
Given structured engineering requirements, you decide:
1. Which scientific theories and disciplines are relevant.
2. Which design principles should guide material selection.
3. Which elements or structural motifs are promising candidates.
4. Which elements or structural patterns to avoid.

Output ONLY valid JSON. No explanations. No preamble. No markdown.

JSON schema:
{{
  "theories": ["<theory 1>", "<theory 2>", ...],
  "design_principles": ["<principle 1>", ...],
  "key_elements": ["<element symbol>", ...],
  "exclusions": ["<element or pattern to avoid>", ...],
  "material_families": ["<family 1>", ...]
}}

Examples of theories: "polymer chain flexibility", "glass transition temperature",
"crosslink density", "band gap engineering", "defect chemistry", "grain boundary strengthening",
"thermodynamic stability", "crystal field theory", "coordination chemistry"

Examples of design principles: "high aromatic content for thermal stability",
"siloxane backbone for flexibility", "crosslinking for mechanical strength"
<|eot_id|><|start_header_id|>user<|end_header_id|>
Engineering requirements:
{parsed_requirement}

Plan the scientific approach.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)

# ── Node function ─────────────────────────────────────────────────────────────

def theory_planner(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Node 2: Map requirements to relevant scientific theories and candidate elements.
    Updates state["scientific_plan"].
    """
    print("\n[Node 2] Theory Planner running...")

    llm = get_rate_limited_llm()
    chain = PLANNER_PROMPT | llm

    req_str = str(state.get("parsed_requirement", {}))
    raw_output = chain.invoke({"parsed_requirement": req_str})

    fallback = {
        "theories": ["materials science", "thermodynamics", "mechanical theory"],
        "design_principles": ["stability", "manufacturability"],
        "key_elements": ["C", "Si", "O", "Al"],
        "exclusions": ["radioactive elements", "extremely rare elements"],
        "material_families": ["unknown"],
    }

    plan = safe_json_parse(raw_output.content, fallback)
    print(f"[Node 2] Scientific plan: theories={plan.get('theories', [])}")

    return {**state, "scientific_plan": plan}

