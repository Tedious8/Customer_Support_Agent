# Support Agent — Multi-Step AI Customer Support System

A production-ready multi-step AI agent for customer support, built with
**LangGraph**, **FastAPI**, and the **Gemini API**.

---

## Architecture

```
User Request
    │
    ▼
FastAPI (main.py)
    │
    ▼
LangGraph Agent (agent.py)
    ├── Node 1: input_guard   ← Guardrails: legal, PII, billing disputes
    ├── Node 2: agent_think   ← Gemini 2.0 Flash reasons + plans tool calls
    ├── Node 3: execute_tools ← FAQ search, order lookup, escalation
    ├── Node 4: [loop back]   ← Up to 5 iterations
    └── Node 5: output_guard  ← Output safety + confidence scoring
    │
    ▼
SupportResponse
    ├── answer
    ├── resolution_status (resolved | escalated | unresolved)
    ├── reasoning_steps[]
    ├── tools_used[]
    └── confidence score
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Start the API server

```bash
python main.py
# or: uvicorn main:app --reload
```

### 4. Test a request

```bash
curl -X POST http://localhost:8000/support \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Where is my order ORD-1002?",
    "order_id": "ORD-1002",
    "user_tier": "premium"
  }'
```

---

## API Endpoints

| Method | Path       | Description               |
| ------ | ---------- | ------------------------- |
| POST   | `/support` | Submit a support question |
| GET    | `/health`  | Health check              |

### Request Body (`POST /support`)

```json
{
  "user_message": "My item arrived damaged",
  "order_id": "ORD-1001", // optional
  "user_tier": "standard", // standard | premium | vip
  "session_id": "my-session-1" // optional, auto-generated if omitted
}
```

---

## Guardrail Rules

| Rule  | Trigger                                     | Action                           |
| ----- | ------------------------------------------- | -------------------------------- |
| G-001 | Legal threats (`sue`, `lawyer`, `lawsuit`…) | Block + escalate (urgent)        |
| G-002 | Chargeback / billing dispute keywords       | Block + escalate (high)          |
| G-003 | PII in message (card numbers, SSN)          | Warn + security escalation       |
| G-004 | Out-of-scope topics                         | Redirect (no escalation)         |
| G-005 | Abusive language                            | Warn + log (continue processing) |
| G-010 | Agent uncertainty ≥ 2 phrases               | Auto-escalate (agent_unsure)     |
| G-011 | Promised specific compensation              | Block + billing escalation       |
| G-012 | Possible hallucinated order details         | Warn + log                       |
| G-HV  | Order total > $500 unresolved               | Route to VIP queue               |

---

## Simulated Orders

| Order ID | Status     | Total   | Notes                                 |
| -------- | ---------- | ------- | ------------------------------------- |
| ORD-1001 | Delivered  | $89.99  | Standard                              |
| ORD-1002 | In Transit | $199.97 | Premium customer                      |
| ORD-1003 | Processing | $499.00 | VIP customer                          |
| ORD-1004 | Cancelled  | $249.99 | Refunded                              |
| ORD-1005 | Delivered  | $779.98 | Dispute flag, triggers VIP escalation |

---

## Extending the Agent

**Add a new tool:**

1. Define an async function with `@tool` decorator in `tools.py`
2. Add it to `ALL_TOOLS` and `TOOL_MAP`
3. Update the system prompt in `agent.py`

**Add a guardrail:**

1. Add a regex or logic check in `guardrails.py`
2. Return a `GuardrailViolation` with severity and escalation config

---

## File Structure

```
support_agent/
├── main.py            # FastAPI app
├── agent.py           # LangGraph agent graph
├── tools.py           # Tool implementations
├── guardrails.py      # Rule-based safety checks
├── knowledge_base.py  # FAQ store + retrieval
├── models.py          # Pydantic schemas
├── logger.py          # Structured reasoning logger
├── requirements.txt
├── .env.example
└── README.md
```
