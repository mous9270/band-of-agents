"""
nodes/requirement_parser.py — Node 1: Requirement Parser

Converts free-form user text into a structured engineering specification.

Input sources (in priority order):
  1. state["user_requirement"]  — set when running directly / locally
  2. state["messages"][-1]      — set by band's LangGraphAdapter when invoked
                                   from band chat (passes HumanMessage objects)
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage

from state import MaterialTheoryState
from utils import get_rate_limited_llm, safe_json_parse


def _extract_requirement(state: MaterialTheoryState) -> str:
    """
    Extract the user's requirement text from state.

    band passes the user's chat message via state["messages"] as a list of
    LangChain BaseMessage objects. We pull the last human message from there.
    When running locally (python main.py --requirement "..."), the text is
    already in state["user_requirement"].
    """
    # Priority 1: already set directly (local / direct mode)
    req = state.get("user_requirement", "")
    if isinstance(req, dict):
        if "user_requirement" in req:
            req = req["user_requirement"]
        elif req:
            first_val = next(iter(req.values()))
            req = first_val if first_val is not None else ""
        else:
            req = ""
            
    if not isinstance(req, str):
        req = str(req) if req is not None else ""

    if req.strip():
        return req.strip()

    # Priority 2: extract from messages (band mode)
    messages = state.get("messages", [])
    if messages:
        # Walk backwards to find the last human message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                # Strip [System]: prefixes band sometimes injects
                content = msg.content
                if isinstance(content, str) and not content.startswith("[System]:"):
                    return content.strip()
            elif isinstance(msg, tuple) and msg[0] == "user":
                content = msg[1]
                if not content.startswith("[System]:"):
                    return content.strip()
        # Fallback: use last message content regardless of type
        last = messages[-1]
        if hasattr(last, "content"):
            return str(last.content).strip()
        if isinstance(last, tuple):
            return str(last[1]).strip()

    return ""

# ── Prompt ────────────────────────────────────────────────────────────────────

PARSER_PROMPT = PromptTemplate(
    input_variables=["user_requirement"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precision material requirements parser.
Your ONLY job is to convert natural language into a JSON engineering specification.
Output ONLY valid JSON. No explanations. No preamble. No markdown.

JSON schema you MUST follow:
{{
  "material_type": "<polymer|ceramic|metal|alloy|composite|semiconductor|other>",
  "application": "<brief description of intended use>",
  "strength": "<low|medium|high|very_high>",
  "temperature_resistance": <number in Celsius or null if not specified>,
  "toxicity": "<low|medium|high> (desired toxicity level of the material)",
  "cost": "<low|medium|high>",
  "electrical": "<conductor|insulator|semiconductor|any>",
  "optical": "<transparent|opaque|translucent|any>",
  "flexibility": "<rigid|flexible|any>",
  "extra_constraints": ["<constraint 1>", "<constraint 2>"]
}}

Rules:
- Use null for numeric fields not mentioned
- "extra_constraints" lists any requirements not captured by other fields
- If a property is not mentioned, use "any" for enum fields
<|eot_id|><|start_header_id|>user<|end_header_id|>
Parse this material requirement:

{user_requirement}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)

# ── Node function ─────────────────────────────────────────────────────────────

def requirement_parser(state: MaterialTheoryState) -> MaterialTheoryState:
    """
    Node 1: Parse the user's natural language requirement into structured JSON.
    - Extracts requirement text from messages (band) or user_requirement (direct)
    - Updates state["parsed_requirement"] and state["user_requirement"]
    """
    print("\n[Node 1] Requirement Parser running...")

    # ── Extract requirement from whichever source is available ──────────────
    requirement = _extract_requirement(state)

    if not requirement:
        print("[Node 1] WARNING: Could not extract requirement from state.")
        requirement = "unknown material requirement"

    print(f"[Node 1] Requirement: {requirement[:100]}")

    llm = get_rate_limited_llm()
    chain = PARSER_PROMPT | llm

    raw_output = chain.invoke({"user_requirement": requirement})

    fallback = {
        "material_type": "unknown",
        "application": requirement,
        "strength": "any",
        "temperature_resistance": None,
        "toxicity": "low",
        "cost": "any",
        "electrical": "any",
        "optical": "any",
        "flexibility": "any",
        "extra_constraints": [],
    }

    parsed = safe_json_parse(raw_output.content, fallback)
    print(f"[Node 1] Parsed requirement: {parsed}")

    return {
        **state,
        "user_requirement": requirement,   # ensure it's set for all downstream nodes
        "parsed_requirement": parsed,
    }

