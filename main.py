"""
main.py — FastAPI application exposing the Support Agent via REST API.

Endpoints:
  POST /support          — Submit a support question
  GET  /health           — Health check
"""

from __future__ import annotations
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from agent import run_agent
from models import SupportRequest, SupportResponse

LOG_FILE = Path(os.getenv("AGENT_LOG_FILE", "agent_runs.jsonl"))

# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    LOG_FILE.touch(exist_ok=True)
    print(f"✅  Support Agent API ready. Log: {LOG_FILE}")
    yield

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Support Agent API",
    description="Multi-step AI customer support agent powered by Gemini + LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class SupportRequestBody(BaseModel):
    user_message: str
    order_id: Optional[str] = None
    user_tier: str = "standard"
    session_id: Optional[str] = None   # auto-generated if omitted

    model_config = {"json_schema_extra": {
        "examples": [{
            "user_message": "Where is my order ORD-1002?",
            "order_id": "ORD-1002",
            "user_tier": "premium",
        }]
    }}

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "log_file": str(LOG_FILE),
    }

@app.post("/support", response_model=SupportResponse, summary="Submit a support question")
async def handle_support(body: SupportRequestBody):
    """
    Run the multi-step support agent against the user's question.

    - Executes input guardrails (legal threats, billing disputes, PII)
    - Calls FAQ search and/or order lookup as needed
    - Escalates to a human queue when appropriate
    - Returns the final answer, reasoning trace, and resolution metadata
    """
    session_id = body.session_id or f"sess-{uuid.uuid4().hex[:8]}"
    request = SupportRequest(
        session_id=session_id,
        user_message=body.user_message,
        order_id=body.order_id,
        user_tier=body.user_tier,
    )

    try:
        response = await run_agent(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return response

# ─────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
