from __future__ import annotations
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    classify_intent,
    retrieve_rag,
    extract_lead_fields,
    capture_lead,
    generate_response,
)


def _route_after_classify(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "inquiry":
        return "retrieve_rag"
    if intent == "collecting":
        return "extract_lead_fields"
    return "generate_response"


def _route_after_extract(state: AgentState) -> str:
    if state.get("lead_name") and state.get("lead_email") and state.get("lead_platform"):
        return "capture_lead"
    return "generate_response"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_rag", retrieve_rag)
    g.add_node("extract_lead_fields", extract_lead_fields)
    g.add_node("capture_lead", capture_lead)
    g.add_node("generate_response", generate_response)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "retrieve_rag": "retrieve_rag",
            "extract_lead_fields": "extract_lead_fields",
            "generate_response": "generate_response",
        },
    )

    # After RAG retrieval, generate the response
    g.add_edge("retrieve_rag", "generate_response")

    # After field extraction, either capture or ask for more
    g.add_conditional_edges(
        "extract_lead_fields",
        _route_after_extract,
        {
            "capture_lead": "capture_lead",
            "generate_response": "generate_response",
        },
    )

    # After capture, generate the confirmation message
    g.add_edge("capture_lead", "generate_response")

    g.add_edge("generate_response", END)

    return g.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
