"""
guardrails.py — Rule-based safety and policy checks applied before and after
the agent generates a response. These act as a deterministic safety net on top
of the LLM's probabilistic judgment.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum

class GuardrailSeverity(str, Enum):
    BLOCK     = "block"      # Hard stop — must escalate
    WARN      = "warn"       # Flag but allow, add disclaimer
    REDIRECT  = "redirect"   # Steer away from topic

@dataclass
class GuardrailViolation:
    rule_id: str
    severity: GuardrailSeverity
    message: str              # Internal description
    user_message: str         # What to tell the user
    escalate: bool = False
    escalation_reason: str = ""

# ─────────────────────────────────────────────
# Rule Definitions
# ─────────────────────────────────────────────

# Keywords that signal legal threats
_LEGAL_PATTERNS = re.compile(
    r"\b(sue|lawsuit|attorney|lawyer|legal action|court|litigation|"
    r"small claims|class action|attorney general|bbb complaint)\b",
    re.IGNORECASE,
)

# Keywords that signal billing disputes requiring human review
_DISPUTE_PATTERNS = re.compile(
    r"\b(chargeback|dispute|fraud|unauthorized charge|"
    r"report to bank|credit card dispute)\b",
    re.IGNORECASE,
)

# Sensitive PII patterns the agent should NEVER repeat back verbatim
_PII_PATTERNS = re.compile(
    r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|"  # credit card
    r"\d{3}[\s\-]?\d{2}[\s\-]?\d{4}|"                    # SSN
    r"password\s*[:=]\s*\S+)\b",
    re.IGNORECASE,
)

# Topics the agent should not engage with
_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(politics|religion|dating|medical advice|drug|illegal|hack|"
    r"bypass|jailbreak|ignore previous instructions)\b",
    re.IGNORECASE,
)

# Abusive language
_ABUSE_PATTERNS = re.compile(
    r"\b(fuck|shit|asshole|bitch|idiot|moron|stupid agent|"
    r"useless bot|piece of crap)\b",
    re.IGNORECASE,
)

# Low-confidence trigger words in LLM output that warrant a human review
_UNCERTAINTY_PHRASES = [
    "i'm not sure",
    "i don't know",
    "i cannot confirm",
    "you may want to verify",
    "i'm unable to determine",
    "i'm not certain",
    "might be",
    "could be",
    "possibly",
]

# ─────────────────────────────────────────────
# Input Guardrails (applied to user message)
# ─────────────────────────────────────────────

def check_input(user_message: str, order_total: float | None = None) -> list[GuardrailViolation]:
    """Run all input-side guardrails. Returns list of violations (empty = clean)."""
    violations: list[GuardrailViolation] = []

    # Rule G-001: Legal threat detection
    if _LEGAL_PATTERNS.search(user_message):
        violations.append(GuardrailViolation(
            rule_id="G-001",
            severity=GuardrailSeverity.BLOCK,
            message="User message contains legal threat keywords.",
            user_message=(
                "I understand you're frustrated, and I take your concern seriously. "
                "Given the nature of your message, I'm immediately routing you to our specialized "
                "escalations team who can best assist you. You'll hear from them within 1 hour."
            ),
            escalate=True,
            escalation_reason="legal_threat",
        ))

    # Rule G-002: Chargeback / fraud dispute
    if _DISPUTE_PATTERNS.search(user_message):
        violations.append(GuardrailViolation(
            rule_id="G-002",
            severity=GuardrailSeverity.BLOCK,
            message="User message contains billing dispute / chargeback keywords.",
            user_message=(
                "Billing disputes require review by our financial team. "
                "I'm escalating this to our Billing Specialists now. "
                "Please have your bank statement handy — they'll contact you within 4 hours."
            ),
            escalate=True,
            escalation_reason="billing_dispute",
        ))

    # Rule G-003: PII in message
    if _PII_PATTERNS.search(user_message):
        violations.append(GuardrailViolation(
            rule_id="G-003",
            severity=GuardrailSeverity.WARN,
            message="Potential PII (card number, SSN, password) detected in user message.",
            user_message=(
                "⚠️ For your security, please avoid sharing credit card numbers, "
                "passwords, or personal ID numbers in chat. Our team will never ask for these. "
                "I've flagged this conversation for security review."
            ),
            escalate=True,
            escalation_reason="sensitive_data",
        ))

    # Rule G-004: Out-of-scope topic
    if _OUT_OF_SCOPE_PATTERNS.search(user_message):
        violations.append(GuardrailViolation(
            rule_id="G-004",
            severity=GuardrailSeverity.REDIRECT,
            message="User message contains out-of-scope topic.",
            user_message=(
                "I'm a customer support assistant focused on helping with orders, "
                "shipping, returns, and account questions. "
                "I'm not able to help with that topic, but I'm happy to assist with anything support-related!"
            ),
            escalate=False,
        ))

    # Rule G-005: Abusive language — still help but log
    if _ABUSE_PATTERNS.search(user_message):
        violations.append(GuardrailViolation(
            rule_id="G-005",
            severity=GuardrailSeverity.WARN,
            message="Abusive language detected. Flagging conversation.",
            user_message=(
                "I'm here to help and I want to make sure we resolve your issue. "
                "Let's work through this together — what can I assist you with?"
            ),
            escalate=False,
        ))

    return violations

def check_output(
    agent_answer: str,
    resolution_status: str,
    order_total: float | None = None,
) -> list[GuardrailViolation]:
    """Run output-side guardrails on the agent's proposed answer."""
    violations: list[GuardrailViolation] = []

    # Rule G-010: Agent uncertainty — escalate if not already
    answer_lower = agent_answer.lower()
    uncertainty_count = sum(1 for phrase in _UNCERTAINTY_PHRASES if phrase in answer_lower)
    if uncertainty_count >= 2 and resolution_status != "escalated":
        violations.append(GuardrailViolation(
            rule_id="G-010",
            severity=GuardrailSeverity.BLOCK,
            message=f"Agent expressed uncertainty {uncertainty_count}× — auto-escalating.",
            user_message=(
                "I want to make sure you get the most accurate help possible. "
                "Let me connect you with a specialist who can give you a definitive answer."
            ),
            escalate=True,
            escalation_reason="agent_unsure",
        ))

    # Rule G-011: Never promise specific compensation amounts
    compensation_pattern = re.compile(
        r"\b(i'll give you|you will receive|we'll refund you|"
        r"compensation of \$|credit of \$|reimburse you \$)\b",
        re.IGNORECASE,
    )
    if compensation_pattern.search(agent_answer):
        violations.append(GuardrailViolation(
            rule_id="G-011",
            severity=GuardrailSeverity.WARN,
            message="Agent attempted to promise specific compensation amount.",
            user_message=(
                "I can see our team is reviewing your case. "
                "Any compensation decisions will be confirmed in writing by our billing team."
            ),
            escalate=True,
            escalation_reason="billing_dispute",
        ))

    # Rule G-012: Don't hallucinate tracking numbers or order details
    hallucination_pattern = re.compile(
        r"\b(your tracking number is [A-Z]{2}-\d+|"
        r"your order will arrive on [A-Z][a-z]+ \d+)\b",
        re.IGNORECASE,
    )
    # Only flag if we didn't actually look up an order
    if hallucination_pattern.search(agent_answer) and "order_lookup" not in agent_answer.lower():
        violations.append(GuardrailViolation(
            rule_id="G-012",
            severity=GuardrailSeverity.WARN,
            message="Possible hallucinated order detail in output.",
            user_message=agent_answer,  # Pass through but log
            escalate=False,
        ))

    return violations

# ─────────────────────────────────────────────
# High-Value Order Check
# ─────────────────────────────────────────────

def is_high_value_order(order_data: dict | None) -> bool:
    """Returns True if order total exceeds $500 — triggers VIP escalation path."""
    if not order_data:
        return False
    try:
        return float(order_data.get("total", 0)) > 500.0
    except (TypeError, ValueError):
        return False

def summarize_violations(violations: list[GuardrailViolation]) -> str:
    """Format violations for logging."""
    if not violations:
        return "none"
    return "; ".join(f"[{v.rule_id}:{v.severity.value}] {v.message}" for v in violations)