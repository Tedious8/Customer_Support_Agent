"""
tools.py — Tool implementations for the Support Agent.
Each tool is a plain async function wrapped with LangChain's @tool decorator.
"""

from __future__ import annotations
import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

from knowledge_base import search_faq


# ─────────────────────────────────────────────
# Simulated Order Database
# ─────────────────────────────────────────────

_ORDER_DB: dict[str, dict] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "status": "delivered",
        "items": [{"name": "Wireless Headphones", "qty": 1, "price": 89.99}],
        "total": 89.99,
        "placed_at": "2025-03-10",
        "delivered_at": "2025-03-15",
        "shipping_carrier": "FedEx",
        "tracking_number": "FX-7823941",
        "customer_tier": "standard",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "status": "in_transit",
        "items": [
            {"name": "USB-C Hub", "qty": 2, "price": 34.99},
            {"name": "Mechanical Keyboard", "qty": 1, "price": 129.99},
        ],
        "total": 199.97,
        "placed_at": "2025-03-17",
        "estimated_delivery": "2025-03-22",
        "shipping_carrier": "UPS",
        "tracking_number": "UP-4491028",
        "customer_tier": "premium",
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "status": "processing",
        "items": [{"name": "Ergonomic Chair", "qty": 1, "price": 499.00}],
        "total": 499.00,
        "placed_at": "2025-03-19",
        "customer_tier": "vip",
    },
    "ORD-1004": {
        "order_id": "ORD-1004",
        "status": "cancelled",
        "items": [{"name": "Smart Watch", "qty": 1, "price": 249.99}],
        "total": 249.99,
        "placed_at": "2025-03-05",
        "cancelled_at": "2025-03-06",
        "refund_status": "refunded",
        "customer_tier": "standard",
    },
    "ORD-1005": {
        "order_id": "ORD-1005",
        "status": "delivered",
        "items": [
            {"name": "4K Monitor", "qty": 1, "price": 699.99},
            {"name": "Monitor Stand", "qty": 1, "price": 79.99},
        ],
        "total": 779.98,
        "placed_at": "2025-03-01",
        "delivered_at": "2025-03-08",
        "shipping_carrier": "DHL",
        "tracking_number": "DH-9921033",
        "dispute_flag": True,          # triggers escalation guardrail
        "customer_tier": "vip",
    },
}

# Escalation queues by category
ESCALATION_QUEUES: dict[str, str] = {
    "billing_dispute":   "billing-team@support.com",
    "legal_threat":      "legal-escalations@company.com",
    "repeated_failure":  "tier2-support@support.com",
    "sensitive_data":    "security-team@company.com",
    "agent_unsure":      "tier2-support@support.com",
    "high_value_order":  "vip-support@support.com",
    "explicit_request":  "tier2-support@support.com",
}


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@tool
def faq_search(query: str) -> str:
    """
    Search the knowledge base for FAQ articles matching the user's question.
    Returns the top 3 relevant articles with their answers.
    Use this first for general product, shipping, returns, or account questions.

    Args:
        query: Natural language question from the user.
    """
    results = search_faq(query, top_k=3)
    if not results:
        return "No relevant FAQ articles found for this query."

    output_parts = []
    for r in results:
        output_parts.append(
            f"[{r['id']}] {r['question']}\n"
            f"Category: {r['category']}\n"
            f"Answer: {r['answer']}\n"
            f"Relevance Score: {r['score']}"
        )
    return "\n\n---\n\n".join(output_parts)


@tool
def order_lookup(order_id: str) -> str:
    """
    Look up detailed information about a specific order by its order ID.
    Use this when the user references an order number or asks about order status,
    delivery, refund, or cancellation for a specific order.

    Args:
        order_id: The order identifier, e.g. 'ORD-1002'.
    """
    # Normalize input (handle variations like "order 1002", "#1002")
    normalized = order_id.strip().upper()
    if not normalized.startswith("ORD-"):
        # Try to extract numeric part
        digits = "".join(filter(str.isdigit, normalized))
        if digits:
            normalized = f"ORD-{digits}"

    order = _ORDER_DB.get(normalized)
    if not order:
        return (
            f"Order '{order_id}' not found in our system. "
            "Please verify the order ID (format: ORD-XXXX) and try again. "
            "The customer can find their order ID in their confirmation email or account dashboard."
        )

    # Build a structured summary
    lines = [
        f"Order ID:     {order['order_id']}",
        f"Status:       {order['status'].replace('_', ' ').title()}",
        f"Customer Tier: {order.get('customer_tier', 'standard').title()}",
        f"Order Total:  ${order['total']:.2f}",
        f"Placed:       {order['placed_at']}",
    ]

    if order.get("delivered_at"):
        lines.append(f"Delivered:    {order['delivered_at']}")
    if order.get("estimated_delivery"):
        lines.append(f"Est. Delivery: {order['estimated_delivery']}")
    if order.get("tracking_number"):
        lines.append(
            f"Tracking:     {order['tracking_number']} via {order.get('shipping_carrier', 'carrier')}"
        )
    if order.get("cancelled_at"):
        lines.append(f"Cancelled:    {order['cancelled_at']}")
    if order.get("refund_status"):
        lines.append(f"Refund:       {order['refund_status'].title()}")
    if order.get("dispute_flag"):
        lines.append("⚠ DISPUTE FLAG: This order has an active dispute on record.")

    lines.append("\nItems:")
    for item in order.get("items", []):
        lines.append(f"  • {item['name']} × {item['qty']}  —  ${item['price']:.2f}")

    return "\n".join(lines)


@tool
def escalate(
    reason: str,
    summary: str,
    order_id: Optional[str] = None,
    priority: str = "normal",
) -> str:
    """
    Escalate the conversation to a human support agent or specialist queue.
    Use this when:
    - The user has a billing dispute or requests a refund outside policy
    - The user mentions legal action or uses threatening language
    - The issue has not been resolved after multiple attempts
    - The user explicitly asks for a human agent
    - You are unsure or lack confidence in your answer
    - The order value is high (>$500) and the issue is unresolved

    Args:
        reason: One of: billing_dispute | legal_threat | repeated_failure |
                sensitive_data | agent_unsure | high_value_order | explicit_request
        summary: Brief summary of the issue to hand off to the human agent.
        order_id: Related order ID if applicable.
        priority: 'urgent' | 'high' | 'normal' | 'low'
    """
    queue = ESCALATION_QUEUES.get(reason, "tier2-support@support.com")
    ticket_id = f"ESC-{random.randint(10000, 99999)}"

    result = (
        f"✅ Escalation created successfully.\n"
        f"Ticket ID:    {ticket_id}\n"
        f"Reason:       {reason.replace('_', ' ').title()}\n"
        f"Priority:     {priority.title()}\n"
        f"Routed to:    {queue}\n"
        f"Order ID:     {order_id or 'N/A'}\n"
        f"Summary:      {summary}\n"
        f"Est. Response: {'< 1 hour' if priority == 'urgent' else '< 4 hours' if priority == 'high' else '< 24 hours'}"
    )
    return result


# ─────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────

ALL_TOOLS = [faq_search, order_lookup, escalate]

TOOL_MAP: dict[str, any] = {
    "faq_search":   faq_search,
    "order_lookup": order_lookup,
    "escalate":     escalate,
}