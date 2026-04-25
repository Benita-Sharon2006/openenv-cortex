"""
CORTEX — Shared Pydantic models for client and server.
Defines Action, Observation, and State contracts.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────

class CortexAction(BaseModel):
    """
    The single action type the agent sends each step.

    tool_name   : which CORTEX tool to call
    arguments   : free-form kwargs forwarded to that tool
    """
    tool_name: str = Field(
        description=(
            "One of: inspect_systems | repair_pipeline | resolve_hr | "
            "handle_ticket | run_audit | brief_ceo | open_incident | "
            "close_incident | escalate | no_op"
        )
    )
    arguments: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# SUB-OBSERVATIONS (one per system)
# ─────────────────────────────────────────────

class PipelineStatus(BaseModel):
    pipeline_id: str
    name: str
    broken: bool
    rows_affected: int
    error_msg: str


class HRConflict(BaseModel):
    conflict_id: str
    employee_a: str
    employee_b: str
    severity: Literal["low", "medium", "high"]
    resolved: bool
    morale_impact: float


class CustomerTicket(BaseModel):
    ticket_id: str
    customer: str
    issue: str
    sla_seconds_remaining: float
    linked_pipeline_id: Optional[str]
    resolved: bool
    satisfaction: Optional[float]


class AuditFlag(BaseModel):
    flag_id: str
    category: Literal["budget_overrun", "fraud_suspected", "compliance_gap"]
    description: str
    resolved: bool
    severity: Literal["low", "medium", "high", "critical"]


class CEOQuestion(BaseModel):
    question_id: str
    question: str
    answered: bool
    correct: Optional[bool]


class Incident(BaseModel):
    incident_id: str
    source_system: str
    title: str
    priority: Literal["P1", "P2", "P3", "P4"]
    open: bool
    age_seconds: float


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────

class CortexObservation(BaseModel):
    """Everything the agent sees at each step."""
    step: int
    max_steps: int
    difficulty: Literal["easy", "medium", "hard"]

    # System 1 — Data Operations
    pipelines: List[PipelineStatus]
    broken_pipeline_count: int

    # System 2 — HR
    hr_conflicts: List[HRConflict]
    team_morale: float

    # System 3 — Customer Support
    tickets: List[CustomerTicket]
    sla_breaches: int

    # System 4 — Financial Audit
    audit_flags: List[AuditFlag]
    budget_health: float

    # System 5 — CEO Dashboard
    ceo_questions: List[CEOQuestion]
    ceo_trust: float

    # System 6 — Incident Command
    incidents: List[Incident]
    open_incident_count: int

    # Meta
    reward: float = 0.0
    done: bool = False
    last_action_result: str
    cumulative_reward: float
    tool_hint: str


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class CortexState(BaseModel):
    """Full internal state — superset of observation."""
    observation: CortexObservation
    scores: Dict[str, float] = Field(default_factory=dict)
    done: bool = False
    episode_id: str = ""