# Material Theory Agent (Agent 1 of 3)

A production-ready LangGraph agent that discovers and validates novel materials
from natural language requirements using:

- **Meta-Llama-3-8B-Instruct** via HuggingFace Inference API
- **LangGraph** for stateful multi-node workflow with refinement loop
- **LangSmith** for full LLM call tracing and observability
- **Real scientific APIs**: PubChem, Wikipedia, Periodic Table (local)

---

## Architecture

```
[User Requirement]
        │
        ▼
┌─────────────────────────┐
│  Node 1: Req. Parser    │  LLM → structured JSON spec
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Node 2: Theory Planner │  LLM → theories, elements, principles
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐◄──── (FAIL loop with rejection memory)
│  Node 3: Sci. Reasoner  │  Tools: Periodic Table, Wikipedia, PubChem
│  (Core Innovation)      │  LLM: first-principles hypothesis generation
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Node 4: Mat. Generator │  LLM → formula + properties + synthesis
│                         │  Tool: PubChem verification
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Node 5: Constraint     │  Layer 1: Oxidation/toxicity/cost (deterministic)
│          Checker        │  Layer 2: LLM requirements judge
└─────────────────────────┘
        │
   PASS │   FAIL → loop back to Node 3 (up to 5 iterations)
        ▼
┌─────────────────────────┐
│  Final Output Assembler │  → Enterprise material proposal
└─────────────────────────┘
```

---

## Setup

### 1. Clone / copy the project

```bash
cd material_theory_agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
# Required: HuggingFace token
# Get from https://huggingface.co/settings/tokens
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# Required: LangSmith (for tracing)
# Get from https://smith.langchain.com → Settings → API Keys
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=material-theory-agent
```

> **HuggingFace token**: Must have access to `meta-llama/Meta-Llama-3-8B-Instruct`.
> Accept the model license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct first.

### 5. Run the agent

**Interactive mode** (recommended first run):
```bash
python main.py
```

**With a requirement directly**:
```bash
python main.py --requirement "Need a heat-resistant polymer adhesive, non-toxic, suitable for aerospace bonding, low cost"
```

**Visualize the graph structure** (no API calls):
```bash
python main.py --visualize
```

---

## Viewing Traces in LangSmith

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Open the **material-theory-agent** project
3. Every run appears as a trace with:
   - All 5 node executions
   - Each LLM call (prompt + response)
   - Token counts and latency per node
   - The full state at each step
   - Refinement loops (if any) shown as nested traces

---

## Project Structure

```
material_theory_agent/
├── main.py                    # CLI entry point
├── graph.py                   # LangGraph workflow + conditional edges
├── state.py                   # MaterialTheoryState TypedDict
├── requirements.txt
├── .env.example
│
├── nodes/
│   ├── requirement_parser.py  # Node 1: NL → JSON spec
│   ├── theory_planner.py      # Node 2: theories + elements
│   ├── scientific_reasoner.py # Node 3: first-principles reasoning (+ tools)
│   ├── material_generator.py  # Node 4: formula + properties
│   └── constraint_checker.py  # Node 5: multi-layer validation
│
├── tools/
│   └── scientific_tools.py    # PubChem, Wikipedia, Periodic Table, Cost
│
└── utils/
    ├── llm.py                 # HuggingFace LLM factory (cached)
    └── json_parser.py         # Robust JSON extraction from LLM output
```

---

## Example Run

**Input:**
```
Need a polymer adhesive with high tensile strength, heat resistance above 200°C,
non-toxic, and low manufacturing cost.
```

**Output (example):**
```
╔════════════════════════════════════════════════╗
║        ENTERPRISE MATERIAL SPECIFICATION       ║
╚════════════════════════════════════════════════╝

Formula       : Poly(dimethylsiloxane-co-diphenylsiloxane)
IUPAC Name    : Poly[oxy(dimethylsilylene)] crosslinked with diphenyl groups
Material Class: polymer
PubChem Verified: False  (novel polymer — expected)

Synthesis     : Hydrosilylation crosslinking of PDMS with diphenyl co-monomer
                at 150°C under Pt catalyst

Predicted Properties:
  tensile_strength_MPa : 35
  density_g_cm3        : 1.12
  melting_point_C      : >250 (decomposes, not melts)
  notes                : Aromatic rings improve thermal stability; siloxane gives flexibility

Validation    : PASS
Confidence    : 82%
Cost Tier     : low

Theories Used :
  • glass transition temperature
  • siloxane backbone chemistry
  • crosslink density and mechanical strength
  • aromatic thermal stabilization

Total Iterations: 2
```

---

## Connecting to Agent 2 (Virtual Lab)

The `final_output` dict from `run_material_theory_agent()` is designed to
feed directly into Agent 2 (Virtual Lab Agent):

```python
from graph import run_material_theory_agent

result = run_material_theory_agent("Your requirement here")

# Pass to Agent 2
# agent2_input = {
#     "formula": result["material"]["formula"],
#     "predicted_properties": result["predicted_properties"],
#     "synthesis_route": result["material"]["synthesis_route"],
# }
```

---

## Tuning

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `MAX_REFINEMENT_ITERATIONS` | `.env` | 5 | Max refinement loops |
| `HF_TEMPERATURE` | `.env` | 0.2 | LLM creativity (lower = more consistent) |
| `HF_MAX_NEW_TOKENS` | `.env` | 1024 | Max tokens per LLM call |
| `HF_MODEL_ID` | `.env` | `meta-llama/Meta-Llama-3-8B-Instruct` | Model to use |
