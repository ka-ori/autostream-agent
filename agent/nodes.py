from __future__ import annotations
import json
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import AgentState
from agent.llm import get_llm
from agent.tools import mock_lead_capture
from rag.retriever import get_retriever

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

_INTENT_SYSTEM = """\
You are an intent classifier for AutoStream, an automated video editing SaaS.

Classify the user's latest message into EXACTLY one of these labels:
- greeting    : Hello, hi, how are you, casual small talk
- inquiry     : Questions about features, pricing, plans, policies, or how AutoStream works
- high_intent : User clearly wants to sign up, purchase a plan, start a trial, or get started
- unknown     : Does not fit any category above

Reply with ONLY the label. No punctuation, no explanation.\
"""

_AGENT_SYSTEM = """\
You are AutoStream's friendly and knowledgeable AI sales assistant.
AutoStream provides automated video editing tools for content creators.

Guidelines:
- Answer product questions using ONLY the knowledge base context provided below.
- Be concise, warm, and helpful.
- Never make up features or prices not in the context.

{kb_context}\
{lead_section}\
"""

_EXTRACT_PROMPT = """\
Extract lead qualification data from the conversation below.
Return a JSON object with exactly these keys: name, email, platform.
- name     : user's full name (string or null)
- email    : user's email address (string or null)
- platform : creator platform, e.g. YouTube, Instagram, TikTok (string or null)

Rules:
- Only extract values the USER explicitly stated — never guess.
- Use null for any field not yet provided.
- Return ONLY valid JSON, nothing else.

Conversation:
{conversation}
"""


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _last_human(state: AgentState) -> str:
    return state["messages"][-1].content


def _conversation_text(state: AgentState, window: int = 12) -> str:
    lines = []
    for m in state["messages"][-window:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def classify_intent(state: AgentState) -> dict:
    """Classify intent of the latest user message.

    If lead collection is already in progress, short-circuits to 'collecting'
    so the flow stays in the extraction branch.
    """
    if state.get("collecting_lead"):
        return {"intent": "collecting"}

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=_last_human(state)),
    ])
    raw = response.content.strip().lower().strip("\"'")
    intent = raw if raw in ("greeting", "inquiry", "high_intent") else "unknown"
    return {"intent": intent}


def retrieve_rag(state: AgentState) -> dict:
    """BM25 retrieval from the knowledge base."""
    context = get_retriever().search(_last_human(state))
    return {"rag_context": context}


def extract_lead_fields(state: AgentState) -> dict:
    """Use the LLM to extract name / email / platform from conversation history."""
    llm = get_llm()
    prompt = _EXTRACT_PROMPT.format(conversation=_conversation_text(state))
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        match = re.search(r"\{[^}]+\}", response.content, re.DOTALL)
        data: dict = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    def _pick(key: str, state_key: str) -> str | None:
        val = data.get(key)
        return val if (val and val != "null") else state.get(state_key)

    return {
        "lead_name": _pick("name", "lead_name"),
        "lead_email": _pick("email", "lead_email"),
        "lead_platform": _pick("platform", "lead_platform"),
    }


def capture_lead(state: AgentState) -> dict:
    """Call mock_lead_capture — only reached when all three fields are present."""
    mock_lead_capture(
        state["lead_name"],
        state["lead_email"],
        state["lead_platform"],
    )
    return {"lead_captured": True}


def generate_response(state: AgentState) -> dict:
    """Generate the agent's conversational reply."""
    llm = get_llm()
    intent = state.get("intent", "unknown")
    collecting = state.get("collecting_lead", False)

    # Build KB context block
    ctx = state.get("rag_context", "")
    kb_context = (
        f"\nKnowledge Base Context:\n{ctx}\n"
        if ctx
        else "\n[No specific context retrieved — answer from general knowledge about AutoStream.]\n"
    )

    # Build lead collection status block
    if state.get("lead_captured"):
        name = state.get("lead_name", "")
        email = state.get("lead_email", "")
        platform = state.get("lead_platform", "")
        lead_section = (
            f"\n\nLEAD JUST CAPTURED: {name} | {email} | {platform}\n"
            "Write a warm confirmation message. Thank the user by name, confirm their details, "
            "and tell them the AutoStream team will be in touch soon. Mention they can start "
            "their free trial at autostream.io/signup"
        )
    elif collecting:
        missing = [
            f for f, k in [("name", "lead_name"), ("email address", "lead_email"), ("creator platform", "lead_platform")]
            if not state.get(k)
        ]
        collected = {
            "name": state.get("lead_name"),
            "email": state.get("lead_email"),
            "platform": state.get("lead_platform"),
        }
        lead_section = (
            f"\n\nLEAD COLLECTION IN PROGRESS.\n"
            f"Collected so far: {collected}\n"
            f"Still needed: {missing}\n"
            "Ask ONLY for the first missing field in a friendly, natural way. "
            "Do not ask for multiple fields at once."
        )
    elif intent == "high_intent":
        lead_section = (
            "\n\nThe user wants to sign up — great!\n"
            "Express enthusiasm, then ask for their full name to get started."
        )
    else:
        lead_section = ""

    system = _AGENT_SYSTEM.format(kb_context=kb_context, lead_section=lead_section)
    messages = [SystemMessage(content=system)] + list(state["messages"])
    response = llm.invoke(messages)

    updates: dict = {"messages": [response]}
    if intent == "high_intent" and not collecting:
        updates["collecting_lead"] = True
    return updates
