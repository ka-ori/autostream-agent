"""AutoStream AI Sales Agent — Streamlit UI."""
from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import get_app
from agent.state import AgentState

st.set_page_config(
    page_title="AutoStream",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme — Claude-inspired dark with orange accent
# --------------------------------------------------------------------------- #

_CSS = """
<style>
:root {
    --bg: #1a1a1a;
    --bg-2: #141414;
    --bg-3: #242424;
    --border: #2e2e2e;
    --fg: #ebe9e4;
    --fg-dim: #8a8680;
    --accent: #d97757;
    --accent-dim: #b35f43;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--fg) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--fg) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

h1, h2, h3, h4 { color: var(--fg) !important; font-weight: 600 !important; letter-spacing: -0.01em; }

.block-container { padding-top: 2.5rem; max-width: 780px; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: var(--bg-3) !important;
    color: var(--fg) !important;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem !important;
    line-height: 1.55;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--accent) !important;
    color: #1a1a1a !important;
    border-color: var(--accent);
}
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--bg-3) !important;
    color: var(--fg) !important;
    border: 1px solid var(--border);
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--bg-2) !important;
    border-top: 1px solid var(--border);
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-3) !important;
    color: var(--fg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    color: var(--fg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 120ms ease;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Caption / secondary text */
[data-testid="stCaptionContainer"], .caption { color: var(--fg-dim) !important; }

code, pre {
    background: var(--bg-2) !important;
    color: var(--accent) !important;
    border-radius: 4px;
    padding: 0.1em 0.35em;
    font-size: 0.9em;
}

/* Custom elements */
.as-logo {
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.02em;
}
.as-logo .dot { color: var(--accent); }

.as-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--fg-dim);
    margin-bottom: 0.35rem;
}

.as-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 500;
    background: var(--bg-3);
    color: var(--fg);
    border: 1px solid var(--border);
}
.as-badge.accent { background: rgba(217, 119, 87, 0.12); color: var(--accent); border-color: var(--accent-dim); }

.as-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    font-size: 0.88rem;
    border-bottom: 1px solid var(--border);
}
.as-row:last-child { border-bottom: none; }
.as-row .key { color: var(--fg-dim); }
.as-row .val { color: var(--fg); font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.82rem; }
.as-row .val.pending { color: var(--fg-dim); }
.as-row .val.done { color: var(--accent); }

.as-banner {
    background: rgba(217, 119, 87, 0.1);
    border: 1px solid var(--accent-dim);
    color: var(--accent);
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

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
    st.markdown(
        '<div class="as-logo">AutoStream<span class="dot">.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#8a8680; font-size:0.85rem; margin-top:0.25rem;">'
        'AI sales assistant</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    state = st.session_state.agent_state
    intent = state.get("intent", "unknown")

    st.markdown('<div class="as-label">Current intent</div>', unsafe_allow_html=True)
    badge_class = "as-badge accent" if intent in ("high_intent", "collecting") else "as-badge"
    st.markdown(f'<span class="{badge_class}">{intent}</span>', unsafe_allow_html=True)

    if state.get("collecting_lead"):
        st.markdown(
            '<div class="as-label" style="margin-top:1.25rem;">Lead collection</div>',
            unsafe_allow_html=True,
        )
        fields = [
            ("Name", state.get("lead_name")),
            ("Email", state.get("lead_email")),
            ("Platform", state.get("lead_platform")),
        ]
        rows = []
        for key, val in fields:
            cls = "val done" if val else "val pending"
            display = val or "—"
            rows.append(
                f'<div class="as-row"><span class="key">{key}</span>'
                f'<span class="{cls}">{display}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    if state.get("lead_captured"):
        st.markdown(
            '<div class="as-banner">Lead captured</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.agent_state = _initial_state()
        st.session_state.display_messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat UI
# --------------------------------------------------------------------------- #

st.markdown(
    '<h1 style="margin-bottom:0.25rem;">AutoStream</h1>'
    '<div style="color:#8a8680; margin-bottom:1.5rem;">'
    'Ask about pricing, features, or start a plan.</div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message AutoStream…", disabled=state.get("lead_captured", False)):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            app = get_app()
            current = st.session_state.agent_state
            invoke_state = {
                **current,
                "messages": list(current["messages"]) + [HumanMessage(content=prompt)],
            }
            result = app.invoke(invoke_state)
            st.session_state.agent_state = result

        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                reply = msg.content
                break

        st.markdown(reply)
        st.session_state.display_messages.append({"role": "assistant", "content": reply})

    st.rerun()
