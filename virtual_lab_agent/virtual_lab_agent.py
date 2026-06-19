"""Step 5 — wire the Virtual Lab crew (Agent 2) into the Band platform.

This is the only file where our agent meets Band. The Band-facing agent is a
CrewAI conversational agent (via CrewAIAdapter). Its real capability — the
two-stage reaction-simulation + manufacturing-cost crew we built in
src/agent_2v02 — is exposed as a single custom tool, so when another agent or
a human asks it to evaluate a material, it runs the full pipeline and returns
the structured VirtualLabReport.

Run:  uv run python virtual_lab_agent.py
"""

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from thenvoi import Agent
from thenvoi.adapters.crewai import CrewAIAdapter
from thenvoi.config import load_agent_config

from agent_2v02.crew import MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "output" / "virtual_lab_report.json"

# Handles of the other agents in the pipeline, used for autonomous handoff.
# Source of truth is the `pipeline:` block in agent_config.yaml; the defaults
# below are a fallback so the agent still runs if that block is absent.
DEFAULT_PIPELINE = {
    # The human who starts the pipeline. Final results are delivered here.
    "requester": "@anonymous9270222",
    "material_theory": "@anonymous9270222/material-theory-agent",
    "virtual_lab": "@anonymous9270222/virtual-lab-agent",
    "procurement": "@anonymous9270222/the-procurement-sourcing",
}


def load_pipeline() -> dict[str, str]:
    """Read the pipeline handle map from agent_config.yaml, with safe defaults."""
    handles = dict(DEFAULT_PIPELINE)
    try:
        data = yaml.safe_load((PROJECT_ROOT / "agent_config.yaml").read_text()) or {}
        handles.update(data.get("pipeline") or {})
    except FileNotFoundError:
        logger.warning("agent_config.yaml not found; using default pipeline handles.")
    return handles


class VirtualLabSimulationInput(BaseModel):
    """Run the full Virtual Lab simulation on a candidate material.

    Use this whenever you are asked to evaluate, simulate, test, or cost a
    material. Pass the Material Theory Agent's (Agent 1) proposal as a JSON
    string, or a plain-language description of the material if no JSON is
    available. Returns a structured VirtualLabReport: reaction feasibility,
    predicted yield, a per-kg manufacturing-cost breakdown, a GO / CONDITIONAL
    / NO_GO manufacturability verdict, and sourcing flags for the Procurement
    Agent.
    """

    material_proposal: str = Field(
        ...,
        description=(
            "Agent 1's material proposal as a JSON string, or a natural-language "
            "description of the material to evaluate."
        ),
    )


def run_virtual_lab(args: VirtualLabSimulationInput) -> dict:
    """Execute the Virtual Lab crew and return the VirtualLabReport as a dict.

    The crew is run in an isolated subprocess. It cannot run in-process: a
    nested crew.kickoff() would start a second CrewAI flow runtime / event bus
    inside this Band agent's own flow loop, which corrupts the outer agent's
    LLM calls (empty-response errors, event-pairing mismatches). The subprocess
    gives the inner crew its own clean runtime.
    """
    raw = args.material_proposal
    try:
        proposal = json.loads(raw)  # accept a JSON proposal from Agent 1
    except json.JSONDecodeError:
        # Not JSON — wrap a free-text description into a minimal proposal so the
        # crew (and main._load_proposal's json.loads) still work.
        proposal = {"original_requirement": raw}

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=PROJECT_ROOT, delete=False, encoding="utf-8"
    ) as fh:
        json.dump(proposal, fh)
        proposal_path = fh.name

    try:
        completed = subprocess.run(
            [sys.executable, "-c", "from agent_2v02.main import run; run()", proposal_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            logger.error("Virtual Lab subprocess failed:\n%s", completed.stderr[-2000:])
            return {
                "status": "FAILURE",
                "error": "Virtual Lab crew failed to run.",
                "detail": completed.stderr[-1000:],
            }
        return json.loads(REPORT_PATH.read_text())
    except subprocess.TimeoutExpired:
        return {"status": "FAILURE", "error": "Virtual Lab simulation timed out."}
    finally:
        Path(proposal_path).unlink(missing_ok=True)


async def main():
    load_dotenv()  # loads OPENAI_API_KEY (and MODEL) for the crew + adapter

    handles = load_pipeline()

    adapter = CrewAIAdapter(
        model=MODEL,
        role="Virtual Lab Agent",
        goal=(
            "Simulate the synthesis of candidate materials, predict reaction "
            "yield, estimate manufacturing cost, and deliver a manufacturability "
            "verdict for the Procurement Agent."
        ),
        backstory=(
            "You are Agent 2 in an automated materials-R&D pipeline. The Material "
            "Theory Agent proposes materials; you run the virtual lab; the "
            "Procurement Agent sources what you approve. You are rigorous and "
            "never fabricate numbers."
        ),
        custom_section=(
            "You are Agent 2, the Virtual Lab Agent, in an AUTONOMOUS materials-R&D "
            "pipeline:\n"
            "  Agent 1 (Material Theory) -> Agent 2 (you) -> Agent 3 (Procurement).\n"
            "A human starts the pipeline with ONE request; the agents then converse "
            "and finish the whole job themselves, without the human re-prompting at "
            "each step.\n\n"
            "MENTIONS — READ CAREFULLY. Every mention must be an EXACT Band handle, "
            "never a display name. A person's handle looks like \"@username\"; an "
            "agent's looks like \"@username/agent-name\". Display names you see in "
            "chat (e.g. [A Mous]) are NOT handles — never turn \"A Mous\" into "
            "\"@A Mous\". If you are unsure of a participant's exact handle, call "
            "thenvoi_get_participants() (or thenvoi_lookup_peers()) and copy the "
            "`handle` field verbatim. The handles you need are fixed:\n"
            f"  - Human requester: {handles['requester']}\n"
            f"  - Procurement Agent: {handles['procurement']}\n\n"
            "WHEN YOU RECEIVE A MATERIAL TO EVALUATE (from the human, or handed to "
            f"you by the Material Theory Agent {handles['material_theory']}):\n"
            "1. Send a brief thenvoi_send_event(..., message_type=\"thought\") saying "
            "you are running the simulation.\n"
            "2. Call the virtuallabsimulation tool with the material proposal — if an "
            "upstream agent gave you JSON, pass it straight through.\n"
            "3. AUTONOMOUSLY HAND THE RESULT TO THE PROCUREMENT AGENT. Do NOT wait for "
            "the human to ask. To do this:\n"
            f"   a. Ensure the Procurement Agent {handles['procurement']} is in the "
            "room: call thenvoi_get_participants(); if it is missing, call "
            "thenvoi_lookup_peers() then thenvoi_add_participant(...) to bring it in.\n"
            "   b. Send it the FULL VirtualLabReport as JSON plus a one-line summary "
            "via thenvoi_send_message(<report-json + summary>, mentions=["
            f"\"{handles['procurement']}\", \"{handles['requester']}\"]) — i.e. "
            "mention BOTH the Procurement Agent and the human requester, each by its "
            "exact handle above, so the human can follow along.\n"
            "4. Keep every number exactly as the tool returned it — never fabricate or "
            "round away the verdict, predicted yield, or total $/kg.\n"
            "You do NOT write the final business proposal yourself: the Procurement "
            "Agent closes the loop with the human. Your job is to simulate, cost, and "
            "forward."
        ),
        additional_tools=[(VirtualLabSimulationInput, run_virtual_lab)],
        verbose=True,
    )

    agent_id, api_key = load_agent_config("virtual_lab")
    agent = Agent.create(adapter=adapter, agent_id=agent_id, api_key=api_key)

    logger.info("Virtual Lab Agent is running! Press Ctrl+C to stop.")
    await agent.run()  # opens a persistent WebSocket and listens forever


if __name__ == "__main__":
    asyncio.run(main())
