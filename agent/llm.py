from __future__ import annotations
import os
from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    """Return a chat model based on available API keys.

    Priority order: GROQ → ANTHROPIC → OPENAI → GOOGLE
    Override with LLM_PROVIDER env var.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()

    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
        )

    if provider == "anthropic" or (provider == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            temperature=0.3,
        )

    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
        )

    if provider == "google" or (provider == "auto" and os.getenv("GOOGLE_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=0.3,
        )

    raise ValueError(
        "No LLM provider configured.\n"
        "Set one of: GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
    )
