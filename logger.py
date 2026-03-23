"""
logger.py — Structured reasoning logger for agent debugging and audit trails.
Outputs to console (Rich) and optionally to a JSONL file for offline analysis.
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from models import ReasoningStep, ToolCall, SupportResponse

console = Console()
LOG_FILE = Path(os.getenv("AGENT_LOG_FILE", "agent_runs.jsonl"))

# ─────────────────────────────────────────────
# Session Logger
# ─────────────────────────────────────────────

class AgentLogger:
    """
    Tracks a single agent session: records reasoning steps, tool calls,
    guardrail events, and final resolution. Renders a Rich summary to console.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.steps: list[ReasoningStep] = []
        self._step_counter = 0
        self._start_time = time.perf_counter()

    # ── Recording ──────────────────────────────

    def log_thought(self, thought: str) -> ReasoningStep:
        self._step_counter += 1
        step = ReasoningStep(step=self._step_counter, thought=thought)
        self.steps.append(step)
        console.print(
            f"  [dim]→ Step {self._step_counter}:[/dim] [italic]{thought[:120]}[/italic]"
        )
        return step

    def log_tool_call(
        self,
        thought: str,
        tool_name: str,
        args: dict,
        result: Any,
        latency_ms: float,
    ) -> ReasoningStep:
        self._step_counter += 1
        tc = ToolCall(
            tool=tool_name,
            args=args,
            result=result,
            latency_ms=latency_ms,
        )
        step = ReasoningStep(
            step=self._step_counter,
            thought=thought,
            tool_call=tc,
        )
        self.steps.append(step)

        # Rich console output
        tool_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
        tool_table.add_row("[bold cyan]Tool[/bold cyan]", tool_name)
        tool_table.add_row("[bold cyan]Args[/bold cyan]", json.dumps(args, default=str))
        tool_table.add_row("[bold cyan]Latency[/bold cyan]", f"{latency_ms:.1f}ms")
        result_preview = str(result)[:200].replace("\n", " ")
        tool_table.add_row("[bold cyan]Result[/bold cyan]", result_preview)
        console.print(
            Panel(tool_table, title=f"[yellow]⚙ Tool Call[/yellow] — Step {self._step_counter}", expand=False)
        )
        return step

    def log_guardrail(self, thought: str, rule_id: str, description: str) -> ReasoningStep:
        self._step_counter += 1
        step = ReasoningStep(
            step=self._step_counter,
            thought=thought,
            guardrail_triggered=f"{rule_id}: {description}",
        )
        self.steps.append(step)
        console.print(
            f"  [bold red]🛡 Guardrail {rule_id}[/bold red] — {description}"
        )
        return step

    # ── Final Summary ───────────────────────────

    def print_summary(self, response: SupportResponse) -> None:
        elapsed = (time.perf_counter() - self._start_time) * 1000

        status_color = {
            "resolved":   "green",
            "escalated":  "yellow",
            "unresolved": "red",
        }.get(response.resolution_status, "white")

        summary = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
        summary.add_row("Session",    self.session_id)
        summary.add_row("Status",     f"[{status_color}]{response.resolution_status.upper()}[/{status_color}]")
        summary.add_row("Confidence", f"{response.confidence:.0%}")
        summary.add_row("Steps",      str(len(self.steps)))
        summary.add_row("Tools Used", ", ".join(t for t in response.tools_used) or "none")
        summary.add_row("Latency",    f"{elapsed:.0f}ms")
        if response.escalation_reason:
            summary.add_row("Escalation", response.escalation_reason)

        console.print(Panel(summary, title="[bold]📋 Agent Run Summary[/bold]", border_style="blue"))
        console.print(
            Panel(
                Text(response.answer[:500], overflow="fold"),
                title="[bold green]Final Answer[/bold green]",
                border_style="green",
            )
        )

    # ── Persistence ─────────────────────────────

    def persist(self, response: SupportResponse) -> None:
        """Append run to JSONL log file for offline analysis."""
        record = {
            "timestamp":         datetime.utcnow().isoformat(),
            "session_id":        self.session_id,
            "resolution_status": response.resolution_status,
            "escalation_reason": response.escalation_reason,
            "confidence":        response.confidence,
            "tools_used":        list(response.tools_used),
            "step_count":        len(self.steps),
            "latency_ms":        response.latency_ms,
            "answer_preview":    response.answer[:300],
            "steps": [
                {
                    "step":       s.step,
                    "thought":    s.thought,
                    "tool":       s.tool_call.tool if s.tool_call else None,
                    "guardrail":  s.guardrail_triggered,
                }
                for s in self.steps
            ],
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")

# ─────────────────────────────────────────────
# Session header printer
# ─────────────────────────────────────────────

def print_session_header(session_id: str, user_message: str, order_id: Optional[str]) -> None:
    console.print()
    console.rule(f"[bold blue]🤖 Support Agent — Session {session_id}[/bold blue]")
    console.print(f"[bold]User:[/bold] {user_message}")
    if order_id:
        console.print(f"[dim]Order ID provided: {order_id}[/dim]")
    console.print()
