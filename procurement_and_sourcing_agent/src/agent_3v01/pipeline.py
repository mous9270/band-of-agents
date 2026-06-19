"""The Procurement & Sourcing pipeline (Agent 3) — built with pydantic-ai.

This is Agent 3's *real capability*, independent of the Band platform. A single
pydantic-ai Agent:

  1. reads Agent 2's VirtualLabReport,
  2. calls ``lookup_supply_chain`` for each key feedstock (grounded sourcing),
  3. calls ``compute_roi`` to ground the product economics,
  4. returns a structured ``ExecutiveProductProposal``.

The LLM does the judgement (which feedstocks matter, GO/NO-GO, the executive
narrative); the deterministic tools supply the prices, lead times and ROI math
so the financial story in the proposal is reproducible arithmetic, not guesses.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent

from agent_3v01.models import ExecutiveProductProposal
from agent_3v01.tools.sourcing_tools import compute_roi, lookup_supply_chain

# pydantic-ai model string, e.g. "openai:gpt-4o" or "anthropic:claude-opus-4-8".
# Read from MODEL (set in .env); defaults to OpenAI to match the rest of the
# pipeline (Agent 2 ships with OPENAI_API_KEY).
MODEL = os.getenv("MODEL", "openai:gpt-4o")

SYSTEM_PROMPT = """\
You are the Procurement & Product Sourcing Agent (Agent 3) in an automated
materials-R&D pipeline. The Material Theory Agent (Agent 1) proposes a material;
the Virtual Lab Agent (Agent 2) simulates its synthesis and manufacturing cost
and hands you a VirtualLabReport. Your job is to decide whether the enterprise
should actually make and sell this material.

Given a VirtualLabReport you MUST:
1. Identify the key feedstocks / raw materials needed (from the formula,
   composition, synthesis route, catalysts and cost drivers in the report).
2. Call `lookup_supply_chain` ONCE for EACH key feedstock to get grounded
   availability, price, lead time and supplier counts. Never invent these.
3. Call `compute_roi` to ground the economics. Use the report's
   total_cost_usd_per_kg as the manufacturing cost. Choose a realistic target
   selling price (typically a sensible markup over manufacturing cost, adjusted
   for the material class and demand), an annual production volume, and an
   upfront investment appropriate to the production scale (lab/pilot/industrial).
   State these choices as assumptions.
4. Synthesize an ExecutiveProductProposal:
   - recommendation: GREENLIGHT / PILOT / HOLD / REJECT. Be disciplined — a
     NO_GO manufacturability verdict, negative margin, or a SCARCE/UNAVAILABLE
     single-source feedstock should pull you toward HOLD/REJECT or at most PILOT.
   - a crisp executive_summary a non-technical executive can act on,
   - the supply_chain assessment (roll feedstock risks up into an overall risk
     and flag single points of failure),
   - the roi block populated from `compute_roi`'s returned numbers verbatim,
   - strategic_risks and concrete next_steps,
   - a calibrated confidence in [0,1].

Use tool outputs verbatim for all numbers. Be rigorous and never fabricate
prices, lead times or ROI figures.
"""


def build_agent() -> Agent[None, ExecutiveProductProposal]:
    """Construct the procurement pydantic-ai agent with its deterministic tools."""
    agent: Agent[None, ExecutiveProductProposal] = Agent(
        MODEL,
        output_type=ExecutiveProductProposal,
        system_prompt=SYSTEM_PROMPT,
    )
    # Tools take no RunContext — register them as "plain" tools.
    agent.tool_plain(lookup_supply_chain)
    agent.tool_plain(compute_roi)
    return agent


def run_procurement_analysis(virtual_lab_report_json: str) -> ExecutiveProductProposal:
    """Run the full procurement analysis on a VirtualLabReport JSON string.

    Accepts either Agent 2's VirtualLabReport JSON or a plain-language
    description; returns a validated ExecutiveProductProposal.
    """
    prompt = (
        "Here is the VirtualLabReport from the Virtual Lab Agent (Agent 2). "
        "Source the feedstocks, compute the ROI, and draft the executive "
        "product proposal.\n\n"
        f"{virtual_lab_report_json}"
    )
    result = build_agent().run_sync(prompt)
    return result.output
