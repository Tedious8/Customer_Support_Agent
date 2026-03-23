"""
agent.py — Multi-step Support Agent built with LangGraph.
"""

from __future__ import annotations
import os
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from guardrails import (
    GuardrailSeverity,
    GuardrailViolation,
    check_input,
    check_output,
    is_high_value_order,
    summarize_violations,
)
from logger import AgentLogger, print_session_header
from models import (
    EscalationReason,
    ResolutionStatus,
    SupportRequest,
    SupportResponse,
    ToolName,
)
from tools import ALL_TOOLS, TOOL_MAP, escalate, order_lookup

# ─────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages:              Annotated[list, add_messages]
    session_id:            str
    user_message:          str
    order_id:              Optional[str]
    user_tier:             str

    # Accumulated results
    faq_results:           list[dict]
    order_data:            Optional[dict]
    escalation_triggered:  bool
    escalation_reason:     Optional[str]
    escalation_ticket:     Optional[str]
    guardrail_violations:  list[str]

    # Final answer
    final_answer:          Optional[str]
    resolution_status:     str
    confidence:            float
    tools_used:            list[str]

    # Internal
    logger:                Any           # AgentLogger instance
    iteration:             int
    input_blocked:         bool

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful, professional customer support agent for an e-commerce platform.

TOOLS AVAILABLE:
- faq_search(query): Search the knowledge base for answers to general questions.
- order_lookup(order_id): Retrieve order status, tracking, and details.
- escalate(reason, summary, order_id, priority): Route to a human agent.

DECISION RULES:
1. ALWAYS call faq_search for general product, shipping, returns, or account questions.
2. ALWAYS call order_lookup if the user provides or implies an order ID.
3. Escalate immediately if:
   - The user mentions legal action, lawyers, or lawsuits → reason: legal_threat
   - There is a billing dispute or chargeback request → reason: billing_dispute
   - The order value > $500 and the issue is unresolved → reason: high_value_order
   - You are not confident in your answer → reason: agent_unsure
   - The user explicitly asks for a human → reason: explicit_request
4. NEVER guess order details, tracking numbers, or delivery dates — use order_lookup.
5. NEVER promise specific refund amounts — say a specialist will confirm.
6. Be concise, warm, and professional. Use the customer's tier to calibrate urgency.

After calling the necessary tools, synthesize a final answer that:
- Directly addresses the user's question
- References specific data from tool results
- Sets clear next steps if action is needed
- Is 2–4 sentences for simple issues, up to 6 for complex ones
"""

# ─────────────────────────────────────────────
# Node: Input Guardrails
# ─────────────────────────────────────────────

def input_guard_node(state: AgentState) -> dict:
    lg: AgentLogger = state["logger"]
    violations = check_input(state["user_message"])

    if not violations:
        lg.log_thought("Input guardrails passed — proceeding to agent reasoning.")
        return {"input_blocked": False, "guardrail_violations": []}

    block_violations = [v for v in violations if v.severity == GuardrailSeverity.BLOCK]
    warn_violations  = [v for v in violations if v.severity == GuardrailSeverity.WARN]
    redir_violations = [v for v in violations if v.severity == GuardrailSeverity.REDIRECT]

    violation_strs = [summarize_violations(violations)]

    # Redirect / warn — still proceed but note it
    if not block_violations:
        for v in warn_violations + redir_violations:
            lg.log_guardrail("Output guardrail (warn/redirect)", v.rule_id, v.message)
        # Return modified message as the answer but still process
        top = (warn_violations + redir_violations)[0]
        return {
            "final_answer":         top.user_message,
            "resolution_status":    "resolved",
            "confidence":           1.0,
            "input_blocked":        True,
            "guardrail_violations": violation_strs,
            "escalation_triggered": top.escalate,
            "escalation_reason":    top.escalation_reason if top.escalate else None,
        }

    # Hard block — escalate
    top = block_violations[0]
    lg.log_guardrail(f"Input blocked by rule {top.rule_id}", top.rule_id, top.message)

    # Call escalate tool programmatically
    esc_result = escalate.invoke({
        "reason":   top.escalation_reason,
        "summary":  f"Auto-escalated: {top.message}. User message: {state['user_message'][:200]}",
        "order_id": state.get("order_id"),
        "priority": "urgent" if top.escalation_reason == "legal_threat" else "high",
    })

    return {
        "final_answer":         top.user_message,
        "resolution_status":    "escalated",
        "confidence":           1.0,
        "input_blocked":        True,
        "escalation_triggered": True,
        "escalation_reason":    top.escalation_reason,
        "escalation_ticket":    esc_result,
        "guardrail_violations": violation_strs,
        "tools_used":           ["escalate"],
    }

def route_after_input_guard(state: AgentState) -> Literal["agent_think", "output_guard"]:
    return "output_guard" if state.get("input_blocked") else "agent_think"

# ─────────────────────────────────────────────
# Node: Agent Think (LLM + Tool Calls)
# ─────────────────────────────────────────────

def agent_think_node(state: AgentState) -> dict:
    lg: AgentLogger = state["logger"]

    # Build LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2,
    ).bind_tools(ALL_TOOLS)

    # Prepend order_id hint if provided
    user_msg = state["user_message"]
    if state.get("order_id"):
        user_msg = f"[Order ID: {state['order_id']}] {user_msg}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ] + state.get("messages", [])

    lg.log_thought(f"Calling LLM (iteration {state.get('iteration', 1)})…")
    t0 = time.perf_counter()
    response: AIMessage = llm.invoke(messages)
    latency = (time.perf_counter() - t0) * 1000

    lg.log_thought(f"LLM responded in {latency:.0f}ms. Tool calls: {len(response.tool_calls)}")

    return {
        "messages":  [response],
        "iteration": state.get("iteration", 0) + 1,
    }

def route_after_think(state: AgentState) -> Literal["execute_tools", "output_guard"]:
    """If the last AI message requested tool calls, execute them; else finalize."""
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "execute_tools"
    return "output_guard"

# ─────────────────────────────────────────────
# Node: Execute Tools
# ─────────────────────────────────────────────

def execute_tools_node(state: AgentState) -> dict:
    lg: AgentLogger = state["logger"]
    msgs  = state["messages"]
    last  = msgs[-1]

    tool_messages: list[ToolMessage] = []
    tools_used     = list(state.get("tools_used", []))
    order_data     = state.get("order_data")
    esc_triggered  = state.get("escalation_triggered", False)
    esc_reason     = state.get("escalation_reason")

    for tc in last.tool_calls:
        t_name = tc["name"]
        t_args = tc["args"]
        tool_fn = TOOL_MAP.get(t_name)

        if not tool_fn:
            result = f"Error: tool '{t_name}' not found."
        else:
            t0 = time.perf_counter()
            result = tool_fn.invoke(t_args)
            latency = (time.perf_counter() - t0) * 1000

            lg.log_tool_call(
                thought=f"Executing {t_name}",
                tool_name=t_name,
                args=t_args,
                result=result,
                latency_ms=latency,
            )

            if t_name not in tools_used:
                tools_used.append(t_name)

            # Cache order data for guardrail checks
            if t_name == "order_lookup" and "not found" not in str(result).lower():
                # Extract order info from result string for downstream checks
                order_data = {"raw": result, "total": _extract_total(result)}

            # Track escalation
            if t_name == "escalate":
                esc_triggered = True
                esc_reason = t_args.get("reason", "agent_unsure")

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {
        "messages":             tool_messages,
        "tools_used":           tools_used,
        "order_data":           order_data,
        "escalation_triggered": esc_triggered,
        "escalation_reason":    esc_reason,
    }


def route_after_tools(state: AgentState) -> Literal["agent_think", "output_guard"]:
    """Loop back to agent_think if under iteration limit, else finalize."""
    MAX_ITERATIONS = 5
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "output_guard"
    return "agent_think"


# ─────────────────────────────────────────────
# Node: Output Guardrails + Finalize
# ─────────────────────────────────────────────

def output_guard_node(state: AgentState) -> dict:
    lg: AgentLogger = state["logger"]

    # Extract final answer from last AIMessage (no tool calls)
    answer = state.get("final_answer")
    if not answer:
        msgs = state.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if not answer:
            answer = (
                "I wasn't able to fully resolve your issue. "
                "Let me connect you with a support specialist who can help."
            )

    # Determine resolution status
    esc_triggered = state.get("escalation_triggered", False)
    resolution = "escalated" if esc_triggered else "resolved"

    # Run output guardrails
    order_total = None
    if state.get("order_data"):
        order_total = state["order_data"].get("total")

    violations = check_output(answer, resolution, order_total)
    guardrail_violations = list(state.get("guardrail_violations", []))

    for v in violations:
        lg.log_guardrail("Output guardrail triggered", v.rule_id, v.message)
        guardrail_violations.append(f"{v.rule_id}: {v.message}")
        if v.escalate and not esc_triggered:
            # Auto-escalate
            esc_result = escalate.invoke({
                "reason":   v.escalation_reason,
                "summary":  f"Output guardrail {v.rule_id} triggered. Original answer truncated.",
                "order_id": state.get("order_id"),
                "priority": "normal",
            })
            answer        = v.user_message
            resolution    = "escalated"
            esc_triggered = True

    # High-value order check
    if (state.get("order_data") and
        is_high_value_order(state["order_data"]) and
        not esc_triggered and resolution != "escalated"):
        lg.log_guardrail("High-value order check", "G-HV", "Order > $500 — routing to VIP queue.")
        escalate.invoke({
            "reason":   "high_value_order",
            "summary":  f"High-value order query. User tier: {state.get('user_tier')}",
            "order_id": state.get("order_id"),
            "priority": "high",
        })
        resolution    = "escalated"
        esc_triggered = True

    # Confidence estimation (heuristic)
    confidence = _estimate_confidence(
        answer=answer,
        tools_used=state.get("tools_used", []),
        violations=violations,
        resolution=resolution,
    )

    lg.log_thought(
        f"Finalizing: status={resolution}, confidence={confidence:.2f}, "
        f"guardrails_triggered={len(violations)}"
    )

    return {
        "final_answer":         answer,
        "resolution_status":    resolution,
        "confidence":           confidence,
        "guardrail_violations": guardrail_violations,
        "escalation_triggered": esc_triggered,
        "escalation_reason":    state.get("escalation_reason"),
    }

# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("input_guard",   input_guard_node)
    graph.add_node("agent_think",   agent_think_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("output_guard",  output_guard_node)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"agent_think": "agent_think", "output_guard": "output_guard"},
    )
    graph.add_conditional_edges(
        "agent_think",
        route_after_think,
        {"execute_tools": "execute_tools", "output_guard": "output_guard"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"agent_think": "agent_think", "output_guard": "output_guard"},
    )
    graph.add_edge("output_guard", END)

    return graph.compile()

# ─────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────

compiled_graph = None  # Lazy init to avoid import-time API key check


def get_graph():
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph

async def run_agent(request: SupportRequest) -> SupportResponse:
    """Main entry point: run the support agent for a single request."""
    lg = AgentLogger(request.session_id)
    print_session_header(request.session_id, request.user_message, request.order_id)

    t0 = time.perf_counter()

    initial_state: AgentState = {
        "messages":             [],
        "session_id":           request.session_id,
        "user_message":         request.user_message,
        "order_id":             request.order_id,
        "user_tier":            request.user_tier,
        "faq_results":          [],
        "order_data":           None,
        "escalation_triggered": False,
        "escalation_reason":    None,
        "escalation_ticket":    None,
        "guardrail_violations": [],
        "final_answer":         None,
        "resolution_status":    "unresolved",
        "confidence":           0.0,
        "tools_used":           [],
        "logger":               lg,
        "iteration":            0,
        "input_blocked":        False,
    }

    graph  = get_graph()
    result = await graph.ainvoke(initial_state)

    latency = (time.perf_counter() - t0) * 1000

    response = SupportResponse(
        session_id       = request.session_id,
        answer           = result.get("final_answer", "Unable to process request."),
        resolution_status= ResolutionStatus(result.get("resolution_status", "unresolved")),
        escalation_reason= EscalationReason(result["escalation_reason"])
                           if result.get("escalation_reason") else None,
        escalation_queue = result.get("escalation_ticket"),
        reasoning_steps  = lg.steps,
        tools_used       = [ToolName(t) for t in result.get("tools_used", [])
                            if t in ToolName._value2member_map_],
        confidence       = result.get("confidence", 0.5),
        latency_ms       = latency,
    )

    lg.print_summary(response)
    lg.persist(response)
    return response

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_total(order_result: str) -> float:
    """Extract numeric total from order_lookup result string."""
    import re
    m = re.search(r"Order Total:\s*\$([0-9.]+)", order_result)
    return float(m.group(1)) if m else 0.0

def _estimate_confidence(
    answer: str,
    tools_used: list[str],
    violations: list[GuardrailViolation],
    resolution: str,
) -> float:
    """
    Heuristic confidence score based on:
    - Whether tools were used (grounded answers = higher confidence)
    - Uncertainty language in the answer
    - Guardrail violations
    - Resolution status
    """
    score = 0.75  # baseline

    if "faq_search" in tools_used or "order_lookup" in tools_used:
        score += 0.15

    uncertainty_terms = ["not sure", "i don't know", "might", "possibly", "i'm unable"]
    if any(t in answer.lower() for t in uncertainty_terms):
        score -= 0.25

    score -= len(violations) * 0.10

    if resolution == "escalated":
        score = min(score, 0.70)  # escalation signals partial resolution

    return round(max(0.1, min(1.0, score)), 2)