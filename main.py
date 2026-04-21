"""AutoStream AI Sales Agent — interactive CLI."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import get_app
from agent.state import AgentState

_BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream  AI  Sales  Assistant              ║
║        Automated Video Editing for Creators          ║
╚══════════════════════════════════════════════════════╝
  Type 'quit' or press Ctrl-C to exit.
  Set DEBUG=1 in .env to see intent + lead state.
"""


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


def run() -> None:
    print(_BANNER)
    app = get_app()
    state = _initial_state()
    debug = bool(os.getenv("DEBUG"))

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAgent: Thanks for chatting — bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("Agent: Thanks for chatting — bye!")
            break

        # Append user message then invoke
        invoke_state = {
            **state,
            "messages": list(state["messages"]) + [HumanMessage(content=user_input)],
        }
        state = app.invoke(invoke_state)

        # Print latest AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nAgent: {msg.content}\n")
                break

        if debug:
            print(
                f"  [DEBUG] intent={state['intent']} | collecting={state['collecting_lead']} | "
                f"name={state['lead_name']} | email={state['lead_email']} | "
                f"platform={state['lead_platform']} | captured={state['lead_captured']}"
            )

        if state.get("lead_captured"):
            print("─" * 54)
            print("  Lead capture complete. Session ended.")
            print("─" * 54)
            break


if __name__ == "__main__":
    run()
