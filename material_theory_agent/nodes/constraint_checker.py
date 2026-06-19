"""
nodes/constraint_checker.py — Node 5: Constraint Checker

Performs multi-layer theoretical validation of the candidate material:

  Layer 1 — Chemistry checks (oxidation states, charge neutrality, toxicity)
             using real periodic table data + custom rules.
  Layer 2 — Requirements checks (does it meet the original spec?)
             using the LLM as a scientific judge.
  Layer 3 — Manufacturability / cost checks
             using the cost estimator tool.

Returns PASS or FAIL with detailed reasons and refinement suggestions.
On FAIL, the failure is added to rejection_history to prevent repetition.
"""

from __future__ import annotations

import json
import re
from typing import List

from langchain_core.prompts import PromptTemplate

from state import MaterialTheoryState
from utils import get_rate_limited_llm, safe_json_parse
from tools import check_oxidation_states, estimate_cost

# ── LLM Prompt for requirements check ────────────────────────────────────────

CHECKER_PROMPT = PromptTemplate(
    input_variables=["candidate", "requirements", "layer1_results"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a materials validation expert. Assess whether a candidate material
meets engineering requirements from a purely theoretical standpoint.
Be strict but fair. Base decisions on known materials science.
<|eot_id|><|start_header_id|>user<|end_header_id|>

=== CANDIDATE MATERIAL ===
{candidate}

=== ENGINEERING REQUIREMENTS ===
{requirements}

=== CHEMISTRY VALIDATION (Layer 1) ===
{layer1_results}

Check these requirements and output ONLY valid JSON:

JSON schema:
{{
  "meets_strength": true/false,
  "meets_temperature": true/false,
  "meets_toxicity": true/false,
  "meets_cost": true/false,
  "meets_electrical": true/false,
  "meets_flexibility": true/false,
  "overall_assessment": "PASS" or "FAIL",
  "confidence": <0.0 to 1.0>,
  "failure_reasons": ["<reason 1>", ...],
  "refinement_suggestions": ["<suggestion 1>", ...]
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)


# ── Helper: extract elements from formula ─────────────────────────────────────

def _extract_elements(formula: str) -> List[str]:
    """Pull unique element symbols from a chemical formula string."""
    if not formula:
        return []
    return list(set(re.findall(r"[A-Z][a-z]?", formula)))


# ── Node function ─────────────────────────────────────────────────────────────

def constraint_checker(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Node 5: Validate the candidate material against all constraints.
    On FAIL: logs to rejection_history, increments iteration.
    On PASS: marks state ready for final output.
    Updates state["constraint_result"].
    """
    print(f"\n[Node 5] Constraint Checker running (iteration {state.get('iteration', 0)})...")

    candidate = state.get("candidate_material", {})
    requirements = state.get("parsed_requirement", {})
    rejection_history: list = state.get("rejection_history", [])
    iteration = state.get("iteration", 0)

    formula = candidate.get("formula", "")
    elements = _extract_elements(formula)

    # ── Layer 1: Chemistry checks (deterministic, tool-based) ────────────────
    print("[Node 5] Layer 1: Chemistry validation...")
    oxidation_check = check_oxidation_states(formula)
    cost_check = estimate_cost(elements)

    chemistry_issues = []
    chemistry_checks = {}

    # Charge / oxidation check
    if oxidation_check["valid"]:
        chemistry_checks["oxidation_states"] = "PASS"
    else:
        chemistry_checks["oxidation_states"] = "FAIL"
        chemistry_issues.extend(oxidation_check["issues"])

    # Toxicity check
    toxic = oxidation_check.get("toxic_elements", [])
    required_toxicity = requirements.get("toxicity", "any")
    if required_toxicity == "low" and toxic:
        chemistry_checks["toxicity"] = "FAIL"
        chemistry_issues.append(f"Toxic elements present: {', '.join(toxic)}")
    else:
        chemistry_checks["toxicity"] = "PASS"

    # Cost check
    required_cost = requirements.get("cost", "any")
    actual_cost = cost_check.get("overall_tier", "medium")
    cost_score_map = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
    cost_req_map = {"low": 2, "medium": 3, "high": 5}
    if required_cost != "any":
        allowed_score = cost_req_map.get(required_cost, 5)
        actual_score = cost_score_map.get(actual_cost, 3)
        if actual_score <= allowed_score:
            chemistry_checks["cost"] = "PASS"
        else:
            chemistry_checks["cost"] = "FAIL"
            chemistry_issues.append(
                f"Cost tier '{actual_cost}' exceeds required '{required_cost}'"
            )
    else:
        chemistry_checks["cost"] = "PASS"

    layer1_summary = {
        "chemistry_checks": chemistry_checks,
        "issues": chemistry_issues,
        "cost_tier": actual_cost,
        "toxic_elements": toxic,
    }

    # ── Layer 2: LLM requirements validation ─────────────────────────────────
    print("[Node 5] Layer 2: LLM requirements validation...")
    llm = get_rate_limited_llm()
    chain = CHECKER_PROMPT | llm

    raw_output = chain.invoke({
        "candidate": json.dumps(candidate, indent=2),
        "requirements": json.dumps(requirements, indent=2),
        "layer1_results": json.dumps(layer1_summary, indent=2),
    })

    fallback = {
        "meets_strength": True,
        "meets_temperature": True,
        "meets_toxicity": len(toxic) == 0,
        "meets_cost": chemistry_checks.get("cost") == "PASS",
        "meets_electrical": True,
        "meets_flexibility": True,
        "overall_assessment": "FAIL" if chemistry_issues else "PASS",
        "confidence": 0.5,
        "failure_reasons": chemistry_issues,
        "refinement_suggestions": ["Review element selection"],
    }

    llm_result = safe_json_parse(raw_output.content, fallback)

    # ── Merge Layer 1 + Layer 2 into final result ────────────────────────────
    all_failures = chemistry_issues + llm_result.get("failure_reasons", [])

    # Final status: PASS only if BOTH layers pass
    if chemistry_issues:
        final_status = "FAIL"
    else:
        final_status = llm_result.get("overall_assessment", "FAIL")

    constraint_result = {
        "status": final_status,
        "checks": {
            **chemistry_checks,
            "meets_strength": "PASS" if llm_result.get("meets_strength") else "FAIL",
            "meets_temperature": "PASS" if llm_result.get("meets_temperature") else "FAIL",
            "meets_electrical": "PASS" if llm_result.get("meets_electrical") else "FAIL",
            "meets_flexibility": "PASS" if llm_result.get("meets_flexibility") else "FAIL",
        },
        "reason": "; ".join(all_failures) if all_failures else "All constraints satisfied",
        "confidence": llm_result.get("confidence", 0.5),
        "suggestions": llm_result.get("refinement_suggestions", []),
        "cost_analysis": cost_check,
    }

    print(f"[Node 5] Result: {final_status} | Confidence: {constraint_result['confidence']}")
    if final_status == "FAIL":
        print(f"[Node 5] Reason: {constraint_result['reason']}")

    # ── Update rejection history on FAIL ─────────────────────────────────────
    new_history = list(rejection_history)
    if final_status == "FAIL":
        new_history.append({
            "iteration": iteration,
            "formula": candidate.get("formula", "unknown"),
            "reason": constraint_result["reason"],
            "suggestions": constraint_result["suggestions"],
        })

    return {
        **state,
        "constraint_result": constraint_result,
        "rejection_history": new_history,
        "iteration": iteration + 1,
    }

