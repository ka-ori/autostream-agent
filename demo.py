"""Scripted demo — runs a full conversation automatically.

Useful for recording the required demo video.
Run: python demo.py
"""
from __future__ import annotations
import time
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import get_app
from agent.state import AgentState

SCRIPT = [
    "Hi there!",
    "Tell me about your pricing plans.",
    "What's included in the Pro plan specifically?",
    "That sounds great — I want to sign up for the Pro plan for my YouTube channel.",
    "My name is Alex Johnson",
    "alex.johnson@gmail.com",
    "YouTube",
]


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


def run_demo() -> None:
    print("\n" + "═" * 56)
    print("  AutoStream Agent — Scripted Demo")
    print("═" * 56 + "\n")

    app = get_app()
    state = _initial_state()

    for user_msg in SCRIPT:
        print(f"You:   {user_msg}")
        time.sleep(0.3)

        invoke_state = {
            **state,
            "messages": list(state["messages"]) + [HumanMessage(content=user_msg)],
        }
        state = app.invoke(invoke_state)

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"Agent: {msg.content}\n")
                break

        time.sleep(0.3)

        if state.get("lead_captured"):
            print("═" * 56)
            print("  Demo complete — lead captured successfully!")
            print("═" * 56)
            break


if __name__ == "__main__":
    run_demo()
