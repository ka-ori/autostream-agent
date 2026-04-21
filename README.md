---
title: AutoStream AI Sales Agent
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AutoStream AI Sales Agent

A **LangGraph**-powered conversational AI agent that acts as an intelligent sales assistant for **AutoStream** — an automated video editing SaaS for content creators.

Built for the ServiceHive / Inflx ML Intern assignment: _Social-to-Lead Agentic Workflow_.

---

## Features

| Capability | Implementation |
|---|---|
| Intent detection | LLM classifier — 4 labels: `greeting`, `inquiry`, `high_intent`, `collecting` |
| RAG knowledge retrieval | BM25 search over structured JSON knowledge base |
| Lead qualification | Progressive field collection (name → email → platform) |
| Tool execution guard | `mock_lead_capture` fires **only** after all 3 fields collected |
| State persistence | Full conversation history via LangGraph `AgentState` across turns |
| Multi-LLM support | Claude 3 Haiku · GPT-4o-mini · Gemini 1.5 Flash |

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url> autostream-agent
cd autostream-agent
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env — add ONE of:
#   ANTHROPIC_API_KEY=sk-ant-...   ← Claude 3 Haiku (default)
#   OPENAI_API_KEY=sk-...          ← GPT-4o-mini
#   GOOGLE_API_KEY=...             ← Gemini 1.5 Flash
```

### 3. Run

```bash
# Interactive chat
python main.py

# Scripted demo (for recording the demo video)
python demo.py

# Debug mode — shows intent label + lead state after every turn
DEBUG=1 python main.py
```

---

## Example Conversation

```
You:   Hi, tell me about your pricing.
Agent: AutoStream has two plans:
       • Basic ($29/mo) — 10 videos/month, 720p, email support
       • Pro ($79/mo)   — Unlimited videos, 4K, AI captions, 24/7 support

You:   I want to sign up for Pro for my YouTube channel.
Agent: That's fantastic! Let's get you set up. What's your full name?

You:   Alex Johnson
Agent: Great to meet you, Alex! What's your email address?

You:   alex@example.com
Agent: Almost there! Which creator platform are you primarily on?

You:   YouTube
[LEAD CAPTURE] Lead captured successfully: Alex Johnson, alex@example.com, YouTube

Agent: You're all set, Alex! Our team will reach out to alex@example.com shortly.
       Start your free trial at autostream.io/signup
```

---

## Architecture

### Why LangGraph?

LangGraph was chosen over AutoGen because this workflow requires a **deterministic state machine** with clearly defined transitions. The lead qualification pipeline has discrete stages — greeting, inquiry, high-intent detection, field collection, and capture — that map directly onto LangGraph's node/edge model. LangGraph makes conditional routing explicit (no ambiguous agent-to-agent negotiation) and provides typed state that is immutable between graph invocations, preventing hidden side effects.

### How state is managed

State is a typed Python dict (`AgentState`) with these fields: conversation `messages`, classified `intent`, `collecting_lead` flag, individual lead fields (`lead_name`, `lead_email`, `lead_platform`), a `lead_captured` boolean, and `rag_context` for the current turn's retrieval result.

The `messages` field uses LangGraph's `add_messages` reducer — new messages are **appended** on each node invocation rather than replacing the list, preserving full conversation history across all turns. All other fields are plain values updated by returning a partial dict from each node. Between CLI turns, the full state dict is kept in memory and re-passed to `app.invoke()`.

### Graph topology

```
User message
     │
     ▼
classify_intent ──► (greeting / high_intent / unknown)
     │                        │
     │ inquiry                │
     ▼                        │
retrieve_rag                  │
     │                        │
     └──────────┬─────────────┘
                ▼
          generate_response ──► END
                ▲
     collecting │
     ▼          │
extract_lead_fields
     │  incomplete
     │──────────────► generate_response ──► END
     │  complete
     ▼
capture_lead ──► generate_response ──► END
```

### RAG pipeline

The knowledge base (`knowledge_base/autostream_kb.json`) contains company info, plan details, policies, and FAQs. On startup, `KnowledgeBaseRetriever` chunks each entry into a document and builds a **BM25Okapi** index (`rank-bm25`). On each inquiry turn, the top-3 highest-scoring chunks are injected into the LLM system prompt as grounded context. BM25 was chosen over vector embeddings to avoid heavy dependencies (no 500 MB model download) while still providing relevance-ranked retrieval suitable for this knowledge base size.

---

## WhatsApp Webhook Integration

To deploy this agent on WhatsApp Business:

### 1. Set up WhatsApp Business API

Register a Meta for Developers account, create a WhatsApp Business App, and obtain a `PHONE_NUMBER_ID` and `WHATSAPP_TOKEN`.

### 2. Build a webhook server (FastAPI example)

```python
from fastapi import FastAPI, Request
import httpx, redis, json
from agent.graph import get_app
from agent.state import AgentState
from langchain_core.messages import HumanMessage

app_server = FastAPI()
redis_client = redis.Redis()
agent = get_app()

@app_server.get("/webhook")
async def verify(hub_mode: str, hub_verify_token: str, hub_challenge: str):
    if hub_verify_token == "YOUR_VERIFY_TOKEN":
        return int(hub_challenge)

@app_server.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    msg_obj = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone   = msg_obj["from"]
    text    = msg_obj["text"]["body"]

    # Load or initialise session state from Redis
    raw = redis_client.get(f"session:{phone}")
    state: AgentState = json.loads(raw) if raw else _initial_state()

    # Run agent
    state = agent.invoke({
        **state,
        "messages": state["messages"] + [HumanMessage(content=text)],
    })

    # Persist updated state
    redis_client.setex(f"session:{phone}", 86400, json.dumps(state, default=str))

    # Send reply via WhatsApp Cloud API
    reply = next(m.content for m in reversed(state["messages"]) if hasattr(m, "content"))
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={"messaging_product": "whatsapp", "to": phone,
                  "text": {"body": reply}},
        )
    return {"status": "ok"}
```

### 3. Expose & register

Deploy the server (e.g. Railway, Fly.io) and register the public URL in the Meta Developer Console as your webhook endpoint.

---

## Project Structure

```
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.json      # Pricing, policies, FAQs
├── rag/
│   ├── __init__.py
│   └── retriever.py            # BM25 retriever
├── agent/
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict
│   ├── llm.py                  # Multi-provider LLM factory
│   ├── tools.py                # mock_lead_capture
│   ├── nodes.py                # Graph node functions
│   └── graph.py                # LangGraph assembly + compile
├── main.py                     # Interactive CLI
├── demo.py                     # Scripted demo for video recording
├── requirements.txt
├── .env.example
└── README.md
```

---

## Evaluation Mapping

| Criterion | Where |
|---|---|
| Agent reasoning & intent detection | `agent/nodes.py` → `classify_intent` |
| Correct use of RAG | `rag/retriever.py` + `agent/nodes.py` → `retrieve_rag` |
| Clean state management | `agent/state.py` + LangGraph `add_messages` |
| Proper tool calling logic | `agent/nodes.py` → `capture_lead` + `_route_after_extract` guard |
| Code clarity & structure | Modular single-responsibility files |
| Real-world deployability | WhatsApp webhook section above |
