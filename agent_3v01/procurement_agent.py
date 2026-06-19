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

import yaml
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

    handles = load_pipeline()

    adapter = PydanticAIAdapter(
        model=MODEL,
        custom_section=(
            "You are Agent 3, the Procurement & Product Sourcing Agent — the FINAL "
            "stage of an AUTONOMOUS materials-R&D pipeline:\n"
            "  Agent 1 (Material Theory) -> Agent 2 (Virtual Lab) -> Agent 3 (you).\n"
            "A human starts the pipeline with ONE request; the agents converse and "
            "finish the job themselves. You normally receive a VirtualLabReport "
            f"handed to you by the Virtual Lab Agent {handles['virtual_lab']} (it "
            "will also @mention the original human requester).\n\n"
            "MENTIONS — READ CAREFULLY. Every mention must be an EXACT Band handle, "
            "never a display name. A person's handle looks like \"@username\"; an "
            "agent's looks like \"@username/agent-name\". Display names you see in "
            "chat (e.g. [A Mous]) are NOT handles — never turn \"A Mous\" into "
            "\"@A Mous\", and never glue a person's name onto an agent name (e.g. "
            "\"@A Mous/virtual-lab-agent\" is invalid). If unsure of a participant's "
            "exact handle, call band_get_participants() (or band_lookup_peers()) and "
            "copy the `handle` field verbatim. The human requester's exact handle "
            f"is {handles['requester']}.\n\n"
            "WHEN YOU RECEIVE A REPORT / ARE ASKED FOR A PROPOSAL:\n"
            "1. Send a brief band_send_event(..., message_type=\"thought\") saying you "
            "are running the procurement analysis.\n"
            "2. Call the `draft_product_proposal` tool with the VirtualLabReport — if "
            "an upstream agent gave you JSON, pass it straight through.\n"
            "3. CLOSE THE LOOP WITH THE HUMAN. You are the last agent, so you deliver "
            "the final answer: call band_send_message(..., mentions=["
            f"\"{handles['requester']}\"]) addressed to the human requester. Do NOT "
            "mention or hand back to the upstream agents — the human is the final "
            "recipient. Include the recommendation (GREENLIGHT/PILOT/HOLD/REJECT), "
            "the ROI (gross margin, payback, ROI %), the overall supply risk, and a "
            "short executive summary.\n"
            "Never fabricate prices, lead times or ROI numbers — they come from the "
            "tool."
        ),
        additional_tools=[draft_product_proposal],
    )

    agent_id, api_key = load_agent_config("procurement")
    agent = Agent.create(adapter=adapter, agent_id=agent_id, api_key=api_key)

    logger.info("Procurement Agent is running! Press Ctrl+C to stop.")
    await agent.run()  # opens a persistent WebSocket and listens forever


if __name__ == "__main__":
    asyncio.run(main())
