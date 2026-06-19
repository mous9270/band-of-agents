"""
graph.py — LangGraph workflow definition for Material Theory Agent (Agent 1).

Graph structure:
  parser → planner → reasoner → generator → checker
                        ↑                      │
                        └──────── (FAIL) ───────┘
                                    │
                                  (PASS)
                                    │
                                   END

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY is set in your .env file.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from state import MaterialTheoryState
from nodes import (
    requirement_parser,
    theory_planner,
    scientific_reasoner,
    material_generator,
    constraint_checker,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MAX_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "1"))


# ── Conditional edge: should we refine or finish? ─────────────────────────────

def should_continue(state: MaterialTheoryState) -> Literal["refine", "finish"]:
    """
    Routing function called after the Constraint Checker node.

    - PASS → "finish" → go to END
    - FAIL + iterations remaining → "refine" → loop back to Scientific Reasoner
    - FAIL + max iterations reached → "finish" anyway (best-effort result)
    """
    result = state.get("constraint_result", {})
    iteration = state.get("iteration", 0)

    if result.get("status") == "PASS":
        print(f"\n[Router] ✅ PASS — Proceeding to final output (iteration {iteration})")
        return "finish"

    if iteration >= MAX_ITERATIONS:
        print(f"\n[Router] ⚠️  Max iterations ({MAX_ITERATIONS}) reached. Using best result.")
        return "finish"

    print(f"\n[Router] 🔄 FAIL — Refining (iteration {iteration}/{MAX_ITERATIONS})")
    return "refine"


# ── Final output assembler ────────────────────────────────────────────────────

def assemble_final_output(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Terminal node: package all state into a clean final_output dict.
    This is what the caller receives as the agent's result.
    """
    candidate = state.get("candidate_material", {})
    constraint = state.get("constraint_result", {})
    req = state.get("parsed_requirement", {})
    plan = state.get("scientific_plan", {})
    hypotheses = state.get("hypotheses", [])

    final_output = {
        "status": "SUCCESS" if constraint.get("status") == "PASS" else "BEST_EFFORT",
        "material": {
            "formula": candidate.get("formula", "N/A"),
            "iupac_name": candidate.get("iupac_name", "N/A"),
            "material_class": candidate.get("material_class", "N/A"),
            "composition": candidate.get("composition", {}),
            "synthesis_route": candidate.get("synthesis_route", "N/A"),
            "pubchem_verified": candidate.get("pubchem_verified", False),
        },
        "predicted_properties": candidate.get("predicted_properties", {}),
        "scientific_reasoning": {
            "theories_used": plan.get("theories", []),
            "design_principles": plan.get("design_principles", []),
            "hypotheses": hypotheses,
        },
        "validation": {
            "overall": constraint.get("status", "UNKNOWN"),
            "checks": constraint.get("checks", {}),
            "confidence": constraint.get("confidence", 0.0),
            "cost_analysis": constraint.get("cost_analysis", {}),
        },
        "original_requirement": state.get("user_requirement", ""),
        "parsed_requirement": req,
        "total_iterations": state.get("iteration", 0),
        "rejection_history": state.get("rejection_history", []),
        "tool_evidence": {
            "literature": state.get("tool_results", {}).get("literature_snippets", []),
            "wikipedia": state.get("tool_results", {}).get("wikipedia_summary", ""),
        },
    }

    print("\n" + "=" * 60)
    print("  MATERIAL THEORY AGENT — FINAL OUTPUT")
    print("=" * 60)
    print(f"  Status        : {final_output['status']}")
    print(f"  Formula       : {final_output['material']['formula']}")
    print(f"  IUPAC Name    : {final_output['material']['iupac_name']}")
    print(f"  Material Class: {final_output['material']['material_class']}")
    print(f"  PubChem ✓     : {final_output['material']['pubchem_verified']}")
    print(f"  Confidence    : {final_output['validation']['confidence']:.2f}")
    print(f"  Iterations    : {final_output['total_iterations']}")
    print("=" * 60)

    return {**state, "final_output": final_output}


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Compile and return the Material Theory Agent LangGraph.

    Nodes:
      parser    → Node 1: Requirement Parser
      planner   → Node 2: Theory Planner
      reasoner  → Node 3: Scientific Reasoner  (re-entry point on FAIL)
      generator → Node 4: Material Generator
      checker   → Node 5: Constraint Checker
      output    → Final output assembler

    Edges:
      parser → planner → reasoner → generator → checker
      checker --[PASS]-→ output → END
      checker --[FAIL]-→ reasoner (loop with rejection memory)
    """
    builder = StateGraph(MaterialTheoryState)

    # Register nodes
    builder.add_node("parser",    requirement_parser)
    builder.add_node("planner",   theory_planner)
    builder.add_node("reasoner",  scientific_reasoner)
    builder.add_node("generator", material_generator)
    builder.add_node("checker",   constraint_checker)
    builder.add_node("output",    assemble_final_output)

    # Entry point
    builder.set_entry_point("parser")

    # Linear edges
    builder.add_edge("parser",    "planner")
    builder.add_edge("planner",   "reasoner")
    builder.add_edge("reasoner",  "generator")
    builder.add_edge("generator", "checker")

    # Conditional edge: refine loop or finish
    builder.add_conditional_edges(
        "checker",
        should_continue,
        {
            "refine":  "reasoner",   # loop back — rejection history included
            "finish":  "output",     # proceed to final packaging
        },
    )

    # Terminal
    builder.add_edge("output", END)

    return builder.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def run_material_theory_agent(user_requirement: str) -> dict:
    """
    Run the full Material Theory Agent and return the final output dict.

    Args:
        user_requirement: Free-form text describing what material is needed.

    Returns:
        final_output dict with formula, properties, reasoning, and validation.
    """
    graph = build_graph()

    initial_state: MaterialTheoryState = {
        "user_requirement": user_requirement,
        "iteration": 0,
        "rejection_history": [],
    }

    print("\n" + "=" * 60)
    print("  MATERIAL THEORY AGENT — STARTING")
    print("=" * 60)
    print(f"  Requirement: {user_requirement[:80]}...")

    # LangSmith tracing happens automatically via env vars — no extra code needed
    final_state = graph.invoke(initial_state)

    return final_state.get("final_output", {})


compiled_graph = build_graph()  # module-level compiled graph for band server
