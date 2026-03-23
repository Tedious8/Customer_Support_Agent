"""
models.py — Shared Pydantic schemas for the Support Agent system.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class EscalationReason(str, Enum):
    BILLING_DISPUTE    = "billing_dispute"
    LEGAL_THREAT       = "legal_threat"
    REPEATED_FAILURE   = "repeated_failure"
    SENSITIVE_DATA     = "sensitive_data"
    AGENT_UNSURE       = "agent_unsure"
    HIGH_VALUE_ORDER   = "high_value_order"
    EXPLICIT_REQUEST   = "explicit_request"

class ToolName(str, Enum):
    FAQ_SEARCH    = "faq_search"
    ORDER_LOOKUP  = "order_lookup"
    ESCALATE      = "escalate"
    FINAL_ANSWER  = "final_answer"

class ResolutionStatus(str, Enum):
    RESOLVED   = "resolved"
    ESCALATED  = "escalated"
    UNRESOLVED = "unresolved"

# ─────────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────────

class SupportRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    user_message: str = Field(..., description="User's support question")
    order_id: Optional[str] = Field(None, description="Optional order ID provided by user")
    user_tier: str = Field("standard", description="Customer tier: standard | premium | vip")
    turn: int = Field(1, description="Conversation turn number")

class ToolCall(BaseModel):
    tool: ToolName
    args: dict[str, Any]
    result: Any
    latency_ms: float

class ReasoningStep(BaseModel):
    step: int
    thought: str
    tool_call: Optional[ToolCall] = None
    guardrail_triggered: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SupportResponse(BaseModel):
    session_id: str
    answer: str
    resolution_status: ResolutionStatus
    escalation_reason: Optional[EscalationReason] = None
    escalation_queue: Optional[str] = None
    reasoning_steps: list[ReasoningStep] = []
    tools_used: list[ToolName] = []
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)