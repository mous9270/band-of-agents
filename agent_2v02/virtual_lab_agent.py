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
            "When asked to evaluate, simulate, test, or cost any material, call the "
            "virtuallabsimulation tool with the material proposal (JSON if you have "
            "it). Report the verdict, predicted yield, total $/kg, and the "
            "recommendations for procurement. If another agent provides a material "
            "proposal, pass it straight to the tool."
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
