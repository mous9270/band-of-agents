"""Wire the Procurement & Sourcing pipeline (Agent 3) into the Band platform.

This is the only file where our agent meets Band. The Band-facing agent is a
pydantic-ai conversational agent (via PydanticAIAdapter). Its real capability —
the supply-chain + ROI + executive-proposal pipeline we built in
src/agent_3v01 — is exposed as a single custom tool, so when another agent
(Agent 2) or a human hands it a VirtualLabReport, it runs the full pipeline and
returns the structured ExecutiveProductProposal.

Run:  uv run python procurement_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import RunContext

from band import Agent
from band.adapters.pydantic_ai import PydanticAIAdapter
from band.config import load_agent_config
from band.core.protocols import AgentToolsProtocol

from agent_3v01.pipeline import MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROPOSAL_PATH = PROJECT_ROOT / "output" / "product_proposal.json"


def _run_pipeline_subprocess(report_path: str) -> dict:
    """Run the procurement pipeline in an isolated subprocess and read its output.

    A subprocess gives the inner pydantic-ai run its own clean event loop and
    process state, instead of nesting a second agent run inside this Band
    agent's own async run loop. It also yields the on-disk product_proposal.json
    artifact, matching how Agent 2 runs its crew.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "from agent_3v01.main import run; run()", report_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        logger.error("Procurement subprocess failed:\n%s", completed.stderr[-2000:])
        return {
            "status": "FAILURE",
            "error": "Procurement pipeline failed to run.",
            "detail": completed.stderr[-1000:],
        }
    return json.loads(PROPOSAL_PATH.read_text())


async def draft_product_proposal(
    ctx: RunContext[AgentToolsProtocol],
    virtual_lab_report: str,
) -> dict:
    """Run the full procurement analysis and draft an executive product proposal.

    Use this whenever you are asked to source a material, check supply-chain
    availability, compute ROI, or produce a product/business proposal. Pass the
    Virtual Lab Agent's (Agent 2) VirtualLabReport as a JSON string, or a
    plain-language description of the material if no JSON is available. Returns a
    structured ExecutiveProductProposal: per-feedstock supply-chain assessment,
    grounded ROI (margin, payback, ROI %), a GREENLIGHT / PILOT / HOLD / REJECT
    recommendation, strategic risks, and next steps.
    """
    raw = virtual_lab_report
    try:
        report = json.loads(raw)  # accept a VirtualLabReport JSON from Agent 2
    except json.JSONDecodeError:
        # Not JSON — wrap free text so the pipeline still has something to read.
        report = {"material_formula": raw, "description": raw}

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=PROJECT_ROOT, delete=False, encoding="utf-8"
    ) as fh:
        json.dump(report, fh)
        report_path = fh.name

    try:
        # Run the blocking subprocess off the event loop.
        return await asyncio.to_thread(_run_pipeline_subprocess, report_path)
    except subprocess.TimeoutExpired:
        return {"status": "FAILURE", "error": "Procurement analysis timed out."}
    except Exception as e:  # noqa: BLE001 - surface failures to the LLM
        return {"status": "FAILURE", "error": f"Procurement analysis failed: {e}"}
    finally:
        Path(report_path).unlink(missing_ok=True)


async def main():
    load_dotenv()  # MODEL + OPENAI_API_KEY for the pipeline + adapter

    adapter = PydanticAIAdapter(
        model=MODEL,
        custom_section=(
            "You are Agent 3, the Procurement & Product Sourcing Agent, in an "
            "automated materials-R&D pipeline. The Material Theory Agent proposes "
            "materials; the Virtual Lab Agent simulates and costs them; you check "
            "real-world supply-chain availability, calculate ROI, and draft the "
            "executive product proposal.\n\n"
            "When asked to source, cost, evaluate the business case for, compute "
            "ROI on, or write a proposal for any material, call the "
            "`draft_product_proposal` tool with the VirtualLabReport (JSON if you "
            "have it). If another agent provides a report, pass it straight "
            "through. Then report the recommendation "
            "(GREENLIGHT/PILOT/HOLD/REJECT), the ROI (gross margin, payback, "
            "ROI %), the overall supply risk, and a short executive summary. "
            "Never fabricate prices, lead times or ROI numbers — they come from "
            "the tool."
        ),
        additional_tools=[draft_product_proposal],
    )

    agent_id, api_key = load_agent_config("procurement")
    agent = Agent.create(adapter=adapter, agent_id=agent_id, api_key=api_key)

    logger.info("Procurement Agent is running! Press Ctrl+C to stop.")
    await agent.run()  # opens a persistent WebSocket and listens forever


if __name__ == "__main__":
    asyncio.run(main())
