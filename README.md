# 🧪 Band of Agents — Automated Materials R&D Pipeline

An autonomous, 3-agent AI pipeline for end-to-end materials discovery — from a plain-English requirement all the way to a board-ready product proposal — wired into the **[Band](https://band.ai)** multi-agent platform.

```
User Requirement
      │
      ▼
┌─────────────────────┐
│  Agent 1            │  Proposes material (formula, properties, synthesis route)
│  Material Theory    │  LangGraph · Groq (llama-3.1-8b) · PubChem · Wikipedia
└────────┬────────────┘
         │  MaterialSpec JSON
         ▼
┌─────────────────────┐
│  Agent 2            │  Simulates reaction, predicts yield, computes $/kg
│  Virtual Lab        │  CrewAI · GPT-4o (or Claude)
└────────┬────────────┘
         │  VirtualLabReport JSON
         ▼
┌─────────────────────┐
│  Agent 3            │  Sources feedstocks, computes ROI, issues GO/NO-GO
│  Procurement &      │  pydantic-ai · GPT-4o (or Claude)
│  Sourcing           │
└─────────────────────┘
         │
         ▼
Executive Product Proposal (GREENLIGHT / PILOT / HOLD / REJECT)
```

All three agents communicate **autonomously** over Band's WebSocket messaging platform. A human sends the initial material requirement; the agents handle every downstream step themselves, passing structured JSON schemas between each other — no re-prompting required.
---

## 🤖 Agent Details

### Agent 1 — Material Theory Agent

**Framework:** LangGraph + LangChain + Groq (llama-3.1-8b-instant)  
**Directory:** `material_theory_agent/`

Converts a free-form material requirement into a validated material specification using a 5-node LangGraph pipeline with a self-correction loop:

```
parser → planner → reasoner → generator → checker
                      ↑                       │
                      └──── (FAIL, retry) ────┘
                                    │
                                 (PASS)
                                    ▼
                                  output
```

| Node | Role |
|---|---|
| **parser** | Converts natural language → structured JSON spec (material type, strength, temp, toxicity, cost, etc.) |
| **planner** | Selects relevant scientific theories and design principles |
| **reasoner** | Generates material hypotheses grounded in the plan |
| **generator** | Proposes a specific candidate material formula via real API lookups (PubChem, Wikipedia) |
| **checker** | Validates oxidation states, toxicity, cost tier, and constraint compliance — routes to `PASS` or `FAIL` |

**Scientific Tools (real APIs, no hallucination):**
- 🔬 **PubChem** — molecular formula, MW, IUPAC name, CID verification
- 📖 **Wikipedia** — background summaries with search fallback
- ⚛️ **Periodic Table** — hardcoded element properties (electronegativity, oxidation states)
- 💰 **Cost Estimator** — element-level cost tier aggregation
- ✅ **Oxidation State Checker** — formula validity + toxicity flagging

**LLM:** Groq free tier with automatic rate-limit retry, exponential backoff, and a 3-model fallback chain (`llama-3.1-8b-instant` → `llama3-8b-8192` → `gemma2-9b-it`).

---

### Agent 2 — Virtual Lab Agent

**Framework:** CrewAI (sequential crew, 2 agents)  
**Directory:** `virtual_lab_agent/`

Simulates the synthesis of the proposed material and estimates per-kg manufacturing economics. The agent runs the inner CrewAI crew in a **subprocess** to avoid event-loop conflicts with Band's WebSocket runtime.

**CrewAI Agents:**

| Agent | Role |
|---|---|
| **Process Simulation Scientist** | Judges reaction feasibility, predicts isolated yield, defines process conditions, flags byproducts and hazards |
| **Manufacturing Cost Analyst** | Converts simulation results into a per-kg cost breakdown using the deterministic `ManufacturingCostCalculatorTool`; issues a `GO / CONDITIONAL / NO_GO` verdict |

**Output schema (`VirtualLabReport`):**

```python
VirtualLabReport:
  material_formula       str
  reaction_simulation:
    feasibility          FEASIBLE | CHALLENGING | INFEASIBLE
    predicted_yield_%    float
    reaction_conditions  list[ReactionCondition]
    scale_up_risk        LOW | MEDIUM | HIGH
  manufacturing_cost:
    total_cost_usd_per_kg  float
    cost_tier              very_low → very_high
  manufacturability_verdict  GO | CONDITIONAL | NO_GO
  confidence             float [0, 1]
```

---

### Agent 3 — Procurement & Sourcing Agent

**Framework:** pydantic-ai  
**Directory:** `procurement_and_sourcing_agent/`

Takes the Virtual Lab's report, grounds the business case with real supply-chain data and deterministic ROI math, and produces an executive-ready product proposal.

**Tools (deterministic — no hallucinated numbers):**
- `lookup_supply_chain(material)` — availability (`ABUNDANT → UNAVAILABLE`), spot price, lead time, supplier count, supply risk
- `compute_roi(...)` — gross margin %, annual revenue/profit, payback period, ROI % over a configurable horizon

**Output schema (`ExecutiveProductProposal`):**

```python
ExecutiveProductProposal:
  recommendation         GREENLIGHT | PILOT | HOLD | REJECT
  executive_summary      str  # 2-4 sentences for non-technical executives
  supply_chain:
    feedstocks           list[FeedstockSourcing]
    overall_supply_risk  LOW | MEDIUM | HIGH
    single_points_of_failure  list[str]
  roi:
    gross_margin_%       float
    payback_period_years float
    roi_%                float
  strategic_risks        list[str]
  next_steps             list[str]
  confidence             float [0, 1]
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) package manager (`pip install uv`)
- A [Band](https://band.ai) account (for multi-agent mode)

---

### Agent 1 — Material Theory Agent

```bash
cd material_theory_agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — fill in GROQ_API_KEY (free at https://console.groq.com)
# and LANGCHAIN_API_KEY (free at https://smith.langchain.com)

# Run directly (no Band)
python main.py --requirement "I need a high-strength, heat-resistant ceramic for aerospace turbine blades"

# Run as a Band agent server (keeps listening for chat messages)
python main.py
```

**Required `.env` keys:**

| Key | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (free tier: 14,400 req/day) |
| `LANGCHAIN_API_KEY` | LangSmith tracing key |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable tracing |
| `MAX_REFINEMENT_ITERATIONS` | How many FAIL→refine loops to allow (default: `5`) |

---

### Agent 2 — Virtual Lab Agent

```bash
cd virtual_lab_agent

# Install dependencies
uv sync

# Configure environment
# Add OPENAI_API_KEY (or MODEL + ANTHROPIC_API_KEY for Claude) to .env

# Run standalone (uses built-in sample proposal)
uv run python -c "from agent_2v02.main import run; run()"

# Run as a Band agent server
uv run python virtual_lab_agent.py
```

---

### Agent 3 — Procurement & Sourcing Agent

```bash
cd procurement_and_sourcing_agent

# Install dependencies
uv sync

# Configure environment
cp .env.example .env                        # add OPENAI_API_KEY (or MODEL + provider key)
cp agent_config.yaml.example agent_config.yaml  # add Band agent_id + api_key

# Run standalone
uv run agent_3v01                           # uses built-in sample VirtualLabReport
uv run agent_3v01 path/to/virtual_lab_report.json

# Run as a Band agent server
uv run python procurement_agent.py
```

---

## 🔗 Running the Full Autonomous Pipeline

To run all three agents as a live, autonomous pipeline on the Band platform:

1. **Register** three external agents on Band and note their `agent_id` and `api_key`.
2. **Configure** each agent's credentials in its `agent_config.yaml`.
3. **Set** the pipeline handle map in each `agent_config.yaml` (see `.yaml.example` files):
   ```yaml
   pipeline:
     requester: "@<your-username>"
     material_theory: "@<username>/material-theory-agent"
     virtual_lab: "@<username>/virtual-lab-agent"
     procurement: "@<username>/the-procurement-sourcing"
   ```
4. **Start** all three agents in separate terminals.
5. **Send** your material requirement in Band chat to the Material Theory Agent.

The pipeline runs fully autonomously:
- Agent 1 designs the material and hands off to Agent 2
- Agent 2 simulates, costs, and hands off to Agent 3
- Agent 3 sources, computes ROI, and sends the final proposal back to you
- On rejection by Agent 2, Agent 1 retries up to 3 times before declaring failure

---

## 🏗️ Architecture & Data Flow

```
User
 │  "I need a heat-resistant polymer adhesive"
 ▼
Agent 1 (LangGraph, Groq)
 │  Runs 5-node pipeline:
 │  parse → plan → reason → generate → validate
 │
 │  MaterialSpec:
 │  { formula: "PDMS-co-diphenyl", class: "polymer",
 │    synthesis_route: "...", pubchem_verified: true,
 │    validation: { status: "PASS", confidence: 0.82 } }
 ▼
Agent 2 (CrewAI, GPT-4o)
 │  Process Simulation Scientist → reaction feasibility, yield
 │  Manufacturing Cost Analyst   → $/kg breakdown, GO/NO-GO
 │
 │  VirtualLabReport:
 │  { feasibility: "FEASIBLE", yield: 78%, cost: $12/kg,
 │    verdict: "GO", confidence: 0.88 }
 ▼
Agent 3 (pydantic-ai, GPT-4o)
 │  lookup_supply_chain() × each feedstock
 │  compute_roi()
 │
 │  ExecutiveProductProposal:
 │  { recommendation: "GREENLIGHT",
 │    roi_percent: 142, payback_years: 1.8,
 │    supply_risk: "LOW" }
 ▼
User receives final executive proposal
```

---

## 🔑 Environment Variables Summary

| Agent | Key | Required | Source |
|---|---|---|---|
| Agent 1 | `GROQ_API_KEY` | ✅ | [console.groq.com](https://console.groq.com) |
| Agent 1 | `LANGCHAIN_API_KEY` | ✅ | [smith.langchain.com](https://smith.langchain.com) |
| Agent 1 | `GROQ_MODEL_ID` | optional | default: `llama-3.1-8b-instant` |
| Agent 1 | `MAX_REFINEMENT_ITERATIONS` | optional | default: `5` |
| Agent 2 | `OPENAI_API_KEY` | ✅ | [platform.openai.com](https://platform.openai.com) |
| Agent 2 | `MODEL` | optional | default: `gpt-4o` |
| Agent 3 | `OPENAI_API_KEY` | ✅ | [platform.openai.com](https://platform.openai.com) |
| Agent 3 | `MODEL` | optional | default: `openai:gpt-4o` |

> **Claude alternative (Agents 2 & 3):** Set `MODEL=anthropic:claude-opus-4-8` and `ANTHROPIC_API_KEY=sk-ant-...` to use Anthropic Claude instead of OpenAI.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration platform | [Band](https://band.ai) |
| Agent 1 framework | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://github.com/langchain-ai/langchain) |
| Agent 1 LLM | [Groq](https://console.groq.com) — `llama-3.1-8b-instant` |
| Agent 1 observability | [LangSmith](https://smith.langchain.com) |
| Agent 2 framework | [CrewAI](https://crewai.com) |
| Agent 2 LLM | OpenAI `gpt-4o` (or Anthropic Claude) |
| Agent 3 framework | [pydantic-ai](https://ai.pydantic.dev) |
| Agent 3 LLM | OpenAI `gpt-4o` (or Anthropic Claude) |
| Structured schemas | [Pydantic v2](https://docs.pydantic.dev) |
| Package manager (Agents 2 & 3) | [uv](https://docs.astral.sh/uv/) |
| External APIs | PubChem REST, Wikipedia API |

---

## 📊 LangSmith Tracing (Agent 1)

Agent 1 emits full LangGraph traces to LangSmith automatically when `LANGCHAIN_TRACING_V2=true`. View them at [smith.langchain.com](https://smith.langchain.com) — every node invocation, LLM call, tool call, and routing decision is captured.

---

## 📄 License

This project is open-source. See individual agent directories for any additional license details.
