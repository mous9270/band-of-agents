"""
state.py — LangGraph state definition for Material Theory Agent.

IMPORTANT: When invoked via band's LangGraphAdapter, the graph receives
{"messages": [("user", "text")]} as input. Our state includes both
`messages` (for band compatibility) and `user_requirement` (our internal key).
Node 1 (requirement_parser) bridges between them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ParsedRequirement(TypedDict, total=False):
    material_type: str
    application: str
    strength: str
    temperature_resistance: float
    toxicity: str
    cost: str
    electrical: str
    optical: str
    flexibility: str
    extra_constraints: List[str]


class ScientificPlan(TypedDict, total=False):
    theories: List[str]
    design_principles: List[str]
    key_elements: List[str]
    exclusions: List[str]
    material_families: List[str]


class Hypothesis(TypedDict, total=False):
    id: int
    reasoning: str
    key_features: List[str]
    predicted_strengths: List[str]
    predicted_weaknesses: List[str]


class CandidateMaterial(TypedDict, total=False):
    formula: str
    iupac_name: str
    composition: Dict[str, Any]
    material_class: str
    predicted_properties: Dict[str, Any]
    synthesis_route: str
    hypothesis_id: int


class ConstraintResult(TypedDict, total=False):
    status: str
    checks: Dict[str, str]
    reason: str
    suggestions: List[str]


class ToolResults(TypedDict, total=False):
    pubchem_data: Dict[str, Any]
    wikipedia_summary: str
    periodic_table_data: Dict[str, Any]
    literature_snippets: List[str]
    cost_estimate: str


class MaterialTheoryState(TypedDict, total=False):
    # ── Band compatibility: band passes input here, we send output here ──
    # `add_messages` is a LangGraph reducer that appends rather than overwrites
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Our internal pipeline fields ─────────────────────────────────────
    user_requirement: str           # extracted from messages[0] by Node 1

    parsed_requirement: ParsedRequirement
    scientific_plan: ScientificPlan
    hypotheses: List[Hypothesis]
    tool_results: ToolResults
    candidate_material: CandidateMaterial
    constraint_result: ConstraintResult

    iteration: int
    rejection_history: List[Dict[str, Any]]

    final_output: Dict[str, Any]
    error: Optional[str]
