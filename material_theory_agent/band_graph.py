from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from typing import List

from band.integrations.langgraph import graph_as_tool
from utils import get_rate_limited_llm

from graph import compiled_graph



checkpointer = InMemorySaver()

llm = get_rate_limited_llm()


def _clean_orphaned_tool_calls(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Remove AIMessages whose tool_calls have no corresponding ToolMessage response.

    This can happen when a previous session failed mid-tool-execution (e.g. due to
    quota errors), leaving orphaned tool call IDs in the band message history.
    Without this cleanup, LangGraph throws INVALID_CHAT_HISTORY on restart.
    """
    # Collect all tool_call_ids that actually have a ToolMessage response
    responded_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id")
    }

    cleaned: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Only keep tool calls that have a matching ToolMessage
            valid_calls = [tc for tc in msg.tool_calls if tc["id"] in responded_ids]
            if valid_calls:
                cleaned.append(
                    AIMessage(
                        content=msg.content,
                        tool_calls=valid_calls,
                        id=msg.id,
                        additional_kwargs=msg.additional_kwargs,
                    )
                )
            elif msg.content:
                # Drop orphaned tool calls but preserve any text content
                cleaned.append(AIMessage(content=msg.content, id=msg.id))
            # else: skip entirely — empty AI message with only orphaned calls
        else:
            cleaned.append(msg)
    return cleaned


def _format_material_result(state: dict) -> str:
    """
    Return a concise material spec string to keep token usage low.
    The full final_output dict can exceed 1000 tokens; this targets ~250.
    """
    out = state.get("final_output", {})
    mat = out.get("material", {})
    props = out.get("predicted_properties", {})
    val = out.get("validation", {})

    lines = [
        "=== MATERIAL SPECIFICATION ===",
        f"Formula      : {mat.get('formula', 'N/A')}",
        f"IUPAC Name   : {mat.get('iupac_name', 'N/A')}",
        f"Class        : {mat.get('material_class', 'N/A')}",
        f"Synthesis    : {mat.get('synthesis_route', 'N/A')}",
        f"PubChem OK   : {mat.get('pubchem_verified', False)}",
        f"Status       : {out.get('status', 'N/A')}",
        f"Confidence   : {val.get('confidence', 0):.0%}",
        f"Validation   : {val.get('overall', 'N/A')}",
    ]
    # At most 5 predicted properties to keep tokens low
    if props:
        lines.append("Key Properties:")
        for k, v in list(props.items())[:5]:
            if v is not None:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ── Rebuild the tool with the concise formatter ───────────────────────────────

material_tool = graph_as_tool(
    compiled_graph,
    name="material_theory",
    description=(
        "Use this tool to design, analyze, validate, or find any material, chemical, "
        "substance, rock, mineral, metal, or compound based on a user requirement. "
        "Returns a concise material specification."
    ),
    result_formatter=_format_material_result,
    input_schema={
        "user_requirement": "Natural language description of the required material."
    },
)


def create_graph(band_tools):

    tools = [
        material_tool,
        *band_tools,
    ]

    # ── Short system prompt (~300 tokens) to stay inside 6000 TPM limit ──────
    _system_prompt = (
        "You are the Material Theory Agent (Agent 1/3) in an autonomous 3-agent pipeline.\n\n"

        "PIPELINE AGENTS:\n"
        "- Agent 2 (Virtual Lab)  : @anonymous9270222/virtual-lab-agent  "
        "(UUID 971e2183-be3b-424a-9347-7dd06fc898d3)\n"
        "- Agent 3 (Procurement)  : @anonymous9270222/the-procurement-sourcing  "
        "(UUID c66092c5-75eb-410c-a4e8-0bc3f853ac2a)\n\n"

        "DETECT YOUR PHASE from who sent the latest message:\n"
        "- Regular user            → PHASE 1\n"
        "- virtual-lab-agent       → PHASE 2\n"
        "- the-procurement-sourcing→ PHASE 3\n\n"

        "PHASE 1 (user message):\n"
        "1. Run `material_theory` tool.\n"
        "2. `band_send_message` the result to @anonymous9270222/virtual-lab-agent for review.\n"
        "3. STOP — do NOT message the user.\n\n"

        "PHASE 2 (virtual-lab-agent reply):\n"
        "- APPROVED → `band_send_message` spec + approval to @anonymous9270222/the-procurement-sourcing.\n"
        "- REJECTED, attempts < 3 → re-run `material_theory` with rejection notes, resend to Agent 2.\n"
        "- REJECTED, attempts >= 3 → `band_send_message` failure notice to original user.\n\n"

        "PHASE 3 (the-procurement-sourcing reply):\n"
        "- APPROVED → `band_send_message` final summary to user "
        "(material spec + lab validation + procurement assessment).\n"
        "- NOT APPROVED → `band_send_message` failure notice with concerns to user.\n\n"

        "RULES (never break):\n"
        "1. ALWAYS use `band_send_message` — never reply with plain text.\n"
        "2. NEVER mention yourself (causes 422 error).\n"
        "3. NEVER message the user until procurement approves (or pipeline fails after 3 attempts).\n"
        "4. ALWAYS use `material_theory` tool — never answer material questions from memory.\n"
    )

    # ── Trim history to keep token usage under the 6000 TPM limit ────────────
    def _prompt_with_trim(state: dict):
        """
        Inject the system prompt and keep only the last 4 non-system messages.
        This prevents the accumulated material spec + conversation from blowing
        past llama-3.1-8b-instant's 6000 TPM cap.
        """
        from langchain_core.messages import SystemMessage, BaseMessage

        all_msgs: list[BaseMessage] = state.get("messages", [])
        non_system = [m for m in all_msgs if not isinstance(m, SystemMessage)]
        # 4 messages = last 2 full exchange pairs — enough for phase detection
        trimmed = non_system[-4:] if len(non_system) > 4 else non_system
        return [SystemMessage(content=_system_prompt)] + trimmed

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=_prompt_with_trim,
        checkpointer=checkpointer,
    )