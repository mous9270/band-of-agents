# Agent 3 — Procurement & Product Sourcing Agent

The third agent in the **Automated Semiconductor / Material R&D Pipeline**, built
with **pydantic-ai** and wired into the **Band** multi-agent platform.

```
Agent 1  ─►  Agent 2  ─►  Agent 3
Material      Virtual      Procurement & Sourcing  (this repo)
Theory        Lab          • supply-chain availability
(proposes)    (simulates   • ROI calculation
              & costs)     • executive product proposal
```

Agent 3 consumes Agent 2's **`VirtualLabReport`**, checks real-world supply-chain
availability for each feedstock, computes grounded ROI, and emits an
**`ExecutiveProductProposal`** with a `GREENLIGHT / PILOT / HOLD / REJECT`
recommendation — the final, human-facing deliverable of the pipeline.

## How it works

The strategic judgement (which feedstocks matter, GO/NO-GO, the executive
narrative) is done by the LLM. The **facts and the money are grounded** by two
deterministic tools so they're reproducible, not hallucinated:

- `lookup_supply_chain(material)` — a procurement market-data feed: availability,
  price, lead time, supplier count, and supply risk per feedstock.
- `compute_roi(...)` — margin, annual revenue/profit, payback period, ROI %.

## Layout

```
procurement_agent.py            # Band wiring (entry point) — PydanticAIAdapter
src/agent_3v01/
  models.py                     # ExecutiveProductProposal schema (+ sub-models)
  pipeline.py                   # the pydantic-ai agent (real capability)
  main.py                       # standalone runner → output/product_proposal.json
  tools/sourcing_tools.py       # deterministic supply-chain + ROI tools
```

## Setup

```bash
uv sync
cp .env.example .env                       # set MODEL + OPENAI_API_KEY
cp agent_config.yaml.example agent_config.yaml   # paste Band agent_id + api_key
```

## Run standalone (no Band)

Runs the full pipeline against a built-in sample VirtualLabReport, or a file you
pass:

```bash
uv run agent_3v01                          # uses the built-in sample
uv run agent_3v01 path/to/virtual_lab_report.json
```

Output is printed and written to `output/product_proposal.json`.

## Run on the Band platform

Register a Band **external agent** named "procurement", put its credentials in
`agent_config.yaml`, then:

```bash
uv run python procurement_agent.py
```

The agent listens on a WebSocket; when Agent 2 (or a human) hands it a
`VirtualLabReport`, it calls `draft_product_proposal`, runs the pipeline, and
replies with the structured proposal.

> Model: defaults to `openai:gpt-4o` to match the existing pipeline. Set
> `MODEL=anthropic:claude-opus-4-8` (and `ANTHROPIC_API_KEY`) in `.env` to use
> Claude instead.
