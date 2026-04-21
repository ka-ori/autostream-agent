"""AutoStream AI Sales Agent — Streamlit UI."""
from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import get_app
from agent.state import AgentState

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Session state init
# --------------------------------------------------------------------------- #

def _initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": "unknown",
        "collecting_lead": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "rag_context": "",
    }

if "agent_state" not in st.session_state:
    st.session_state.agent_state = _initial_state()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/video-editing.png", width=64)
    st.title("AutoStream")
    st.caption("AI-powered video editing for creators")
    st.divider()

    state = st.session_state.agent_state
    intent = state.get("intent", "unknown")

    intent_color = {
        "greeting": "🟢",
        "inquiry": "🔵",
        "high_intent": "🟡",
        "collecting": "🟠",
        "unknown": "⚪",
    }
    st.markdown(f"**Intent:** {intent_color.get(intent, '⚪')} `{intent}`")

    if state.get("collecting_lead"):
        st.divider()
        st.markdown("**Lead Collection**")
        st.markdown(f"{'✅' if state.get('lead_name') else '⬜'} Name: `{state.get('lead_name') or '—'}`")
        st.markdown(f"{'✅' if state.get('lead_email') else '⬜'} Email: `{state.get('lead_email') or '—'}`")
        st.markdown(f"{'✅' if state.get('lead_platform') else '⬜'} Platform: `{state.get('lead_platform') or '—'}`")

    if state.get("lead_captured"):
        st.success("🎉 Lead captured!")

    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.agent_state = _initial_state()
        st.session_state.display_messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat UI
# --------------------------------------------------------------------------- #

st.title("🎬 AutoStream Assistant")
st.caption("Ask about pricing, features, or sign up for a plan.")

# Render existing messages
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type a message…", disabled=state.get("lead_captured", False)):
    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            app = get_app()
            current = st.session_state.agent_state
            invoke_state = {
                **current,
                "messages": list(current["messages"]) + [HumanMessage(content=prompt)],
            }
            result = app.invoke(invoke_state)
            st.session_state.agent_state = result

        # Get latest AI response
        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                reply = msg.content
                break

        st.markdown(reply)
        st.session_state.display_messages.append({"role": "assistant", "content": reply})

    st.rerun()
