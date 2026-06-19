"""
main.py — CLI entry point for the Material Theory Agent.

How band works:
  `python main.py` starts a long-running process that:
    1. Registers this agent with the band platform using your agent_id + api_key
    2. Opens a persistent connection (WebSocket/SSE) to band's servers
    3. Sits and LISTENS — when a user sends a message to your agent in band chat,
       band pushes it here, agent.run() fires, LangGraph executes, result goes back.

The process must stay running in your terminal for band to reach it.
"""

from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from band_graph import create_graph

import argparse
import asyncio
import json

from dotenv import load_dotenv
load_dotenv()

# ── FIX 2: Import compiled_graph (not `graph`) — the old name shadowed
#           the local variable inside run_material_theory_agent()
from graph import compiled_graph, run_material_theory_agent

from band import Agent
from band.config import load_agent_config
from band.adapters import LangGraphAdapter


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MATERIAL THEORY AGENT  (Agent 1 / 3)               ║
║          LangGraph + LangSmith + Groq (llama-3.1-8b)        ║
║          Real APIs: PubChem · Wikipedia · Periodic Table     ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_result(result: dict):
    print("\n" + "═" * 65)
    print("  📋  ENTERPRISE MATERIAL SPECIFICATION")
    print("═" * 65)

    mat = result.get("material", {})
    print(f"\n  🔬  Formula        : {mat.get('formula', 'N/A')}")
    print(f"  📛  IUPAC Name     : {mat.get('iupac_name', 'N/A')}")
    print(f"  🏷️   Material Class  : {mat.get('material_class', 'N/A')}")
    print(f"  ✅  PubChem Verified: {mat.get('pubchem_verified', False)}")
    print(f"\n  🛠️   Synthesis: {mat.get('synthesis_route', 'N/A')}")

    props = result.get("predicted_properties", {})
    if props:
        print("\n  ⚙️   Predicted Properties:")
        for k, v in props.items():
            if v is not None:
                print(f"       {k}: {v}")

    val = result.get("validation", {})
    print(f"\n  🎯  Validation : {val.get('overall', 'N/A')}")
    print(f"  📊  Confidence : {val.get('confidence', 0):.0%}")

    cost = val.get("cost_analysis", {})
    if cost:
        print(f"  💰  Cost Tier  : {cost.get('overall_tier', 'N/A')}")

    theories = result.get("scientific_reasoning", {}).get("theories_used", [])
    if theories:
        print("\n  📚  Theories Used:")
        for t in theories[:5]:
            print(f"       • {t}")

    print(f"\n  🔁  Total Iterations: {result.get('total_iterations', 0)}")
    history = result.get("rejection_history", [])
    if history:
        print(f"  ❌  Rejected Candidates: {len(history)}")
        for h in history:
            print(f"       Iter {h['iteration']}: {h['formula']} → {h['reason'][:60]}")

    wiki = result.get("tool_evidence", {}).get("wikipedia", "")
    if wiki:
        print(f"\n  📖  Wikipedia Evidence:")
        print(f"       {wiki[:200]}...")

    print("\n" + "═" * 65)
    print("  LangSmith traces: https://smith.langchain.com")
    print("═" * 65)


# ── FIX 3: band needs agent.serve(), not agent.run()
#           agent.run(msg) sends ONE message and exits.
#           agent.serve() starts the persistent listener loop that band
#           calls into every time a user sends a message in band chat.

async def start_band_server():
    """
    Start the band agent server — keeps running and handles all
    incoming messages from band chat automatically.
    """
    agent_id, api_key = load_agent_config("material_theory_agent")

    # FIX 4: Pass compiled_graph (the module-level compiled instance),
    #         not the `graph` module itself.
    adapter = LangGraphAdapter(
        graph_factory=create_graph
    )
    
    from custom_preprocessor import CustomPreprocessor
    preprocessor = CustomPreprocessor()
    
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        preprocessor=preprocessor,
    )

    print(f"  🔗  Agent ID  : {agent_id}")
    print(f"  🌐  Connected to band — waiting for messages...")
    print(f"  💬  Go to band chat and send your requirement now.\n")
    print(f"  (Press Ctrl+C to stop)\n")

    # serve() blocks forever, handling each incoming band message by:
    #   1. Wrapping it into LangGraph initial state
    #   2. Running the full graph
    #   3. Returning final_output back to band chat
    await agent.run()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Material Theory Agent — AI-powered material discovery"
    )
    parser.add_argument(
        "--requirement", type=str, default=None,
        help="Run directly with this requirement (no band, no server)",
    )
    parser.add_argument(
        "--output", type=str, default="output.json",
        help="Path to save JSON result when running directly",
    )
    args = parser.parse_args()

    print_banner()

    if args.requirement:
        # ── Direct mode: run once locally, print result, exit ───────────────
        print("  📌  Running in direct mode (no band)...\n")
        result = run_material_theory_agent(args.requirement)
        print_result(result)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  ✅  Full result saved to {args.output}")

    else:
        # ── Band server mode: start listener, stay alive ─────────────────────
        print("  🚀  Starting band agent server...\n")
        await start_band_server()


if __name__ == "__main__":
    asyncio.run(main())