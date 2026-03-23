"""
main.py — FastAPI application exposing the Support Agent via REST API.

Endpoints:
  GET  /health           — Health check
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

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
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "log_file": str(LOG_FILE),
    }

# ─────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
