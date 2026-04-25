"""
CORTEX Environment — server-side implementation.
All six corporate systems live here.
"""
from __future__ import annotations

import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core import Environment
from openenv.core.client_types import StepResult

from models import (
    AuditFlag,
    CEOQuestion,
    CortexAction,
    CortexObservation,
    CortexState,
    CustomerTicket,
    HRConflict,
    Incident,
    PipelineStatus,
)


# ──────────────────────────────────────────────────────────────
# SCENARIO TEMPLATES
# ──────────────────────────────────────────────────────────────

PIPELINE_TEMPLATES = [
    ("etl_sales", "Sales ETL Pipeline", "NullPointerException in transform step"),
    ("report_finance", "Finance Report Generator", "Timeout: DB query exceeded 30s"),
    ("crm_sync", "CRM Sync Job", "Schema mismatch: expected 12 cols got 9"),
    ("ml_feature", "ML Feature Store", "Memory overflow during aggregation"),
    ("inventory_feed", "Inventory Feed", "FTP authentication failure"),
]

HR_TEMPLATES = [
    ("hr_001", "Alice Kumar", "Bob Patel", "high",
     "Alice filed a formal complaint about Bob's conduct in sprint review"),
    ("hr_002", "Carol Singh", "Dave Rao", "medium",
     "Disagreement over project ownership causing team friction"),
    ("hr_003", "Eve Thomas", "Frank Nair", "low",
     "Minor communication breakdown, both parties willing to resolve"),
    ("hr_004", "Grace Iyer", "Hank Menon", "high",
     "Workplace harassment allegation — requires immediate action"),
]

TICKET_TEMPLATES = [
    ("TKT-001", "Infosys Ltd", "Invoice totals wrong — overcharged by 12%",
     "etl_sales"),
    ("TKT-002", "Wipro Corp", "Dashboard showing stale data from 3 days ago",
     "report_finance"),
    ("TKT-003", "TCS Global", "API returning 500 for all product queries",
     None),
    ("TKT-004", "HCL Tech", "Cannot export monthly report — button broken",
     "report_finance"),
    ("TKT-005", "Cognizant", "Wrong inventory count causing shipment delays",
     "inventory_feed"),
    ("TKT-006", "Tech Mahindra", "Customer data appears corrupted in portal",
     "crm_sync"),
]

AUDIT_TEMPLATES = [
    ("AUD-001", "budget_overrun", "Q2 cloud spend is 34% over approved budget",
     "high"),
    ("AUD-002", "fraud_suspected",
     "Unusual payment of 48L to unregistered vendor detected", "critical"),
    ("AUD-003", "compliance_gap",
     "GDPR data retention policy violated — logs kept beyond 90 days", "medium"),
    ("AUD-004", "budget_overrun",
     "Marketing discretionary fund exceeded by 12L", "low"),
]

CEO_QUESTIONS = [
    ("CEO-Q1",
     "What is the current status of our top revenue pipeline, and why is it down?"),
    ("CEO-Q2",
     "How many open customer tickets breach SLA today and what is the root cause?"),
    ("CEO-Q3",
     "Is the fraud flag in Finance real or a false positive? Give me confidence."),
    ("CEO-Q4",
     "What is team morale right now and is the HR situation under control?"),
    ("CEO-Q5",
     "Give me a one-sentence health summary across all six systems."),
]

INCIDENT_TITLES = [
    ("INC-001", "data_ops", "Critical ETL Failure — Revenue Impact", "P1"),
    ("INC-002", "customer_support", "SLA Breach Wave — 3 Enterprise Clients", "P1"),
    ("INC-003", "hr_system", "HR Escalation — Legal Team Notified", "P2"),
    ("INC-004", "financial_audit", "Fraud Flag Requires CFO Approval", "P2"),
    ("INC-005", "ceo_dashboard", "CEO Waiting for Status Briefing", "P3"),
    ("INC-006", "data_ops", "Secondary Pipeline Degradation", "P3"),
]


# ──────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ──────────────────────────────────────────────────────────────

VALID_TOOLS = {
    "inspect_systems",
    "repair_pipeline",
    "resolve_hr",
    "handle_ticket",
    "run_audit",
    "brief_ceo",
    "open_incident",
    "close_incident",
    "escalate",
    "no_op",
}

DIFFICULTY_CONFIG = {
    "easy": {
        "num_pipelines": 1, "num_hr": 1, "num_tickets": 2,
        "num_audits": 0, "num_ceo": 1, "num_incidents": 1,
        "sla_seconds": 300, "max_steps": 30,
    },
    "medium": {
        "num_pipelines": 3, "num_hr": 2, "num_tickets": 4,
        "num_audits": 1, "num_ceo": 2, "num_incidents": 3,
        "sla_seconds": 180, "max_steps": 50,
    },
    "hard": {
        "num_pipelines": 5, "num_hr": 4, "num_tickets": 6,
        "num_audits": 4, "num_ceo": 5, "num_incidents": 6,
        "sla_seconds": 90, "max_steps": 80,
    },
}


# ──────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ──────────────────────────────────────────────────────────────

class CortexEnvironment(Environment):
    """
    CORTEX: Six interconnected corporate systems.
    The agent must triage, prioritize, and resolve cascading crises.
    """

    def __init__(self) -> None:
        self._episode_id = ""
        self._step = 0
        self._difficulty = "medium"
        self._start_time = time.time()

        self._pipelines: List[PipelineStatus] = []
        self._hr_conflicts: List[HRConflict] = []
        self._tickets: List[CustomerTicket] = []
        self._audit_flags: List[AuditFlag] = []
        self._ceo_questions: List[CEOQuestion] = []
        self._incidents: List[Incident] = []

        self._team_morale: float = 0.8
        self._budget_health: float = 0.9
        self._ceo_trust: float = 0.9
        self._sla_breaches: int = 0

        self._data_repair_score: float = 0.0
        self._customer_score: float = 0.0
        self._hr_score: float = 0.0
        self._compliance_score: float = 0.0
        self._financial_score: float = 0.0
        self._ceo_score: float = 0.0
        self._incident_score: float = 0.0

        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_result: str = "Episode not started. Call reset first."

    async def reset(self, difficulty: str = "medium") -> CortexObservation:
        self._difficulty = difficulty if difficulty in DIFFICULTY_CONFIG else "medium"
        cfg = DIFFICULTY_CONFIG[self._difficulty]
        self._episode_id = str(uuid.uuid4())[:8]
        self._step = 0
        self._start_time = time.time()
        self._done = False
        self._cumulative_reward = 0.0
        self._sla_breaches = 0
        self._team_morale = 0.8
        self._budget_health = 0.9
        self._ceo_trust = 0.9
        self._data_repair_score = 0.0
        self._customer_score = 0.0
        self._hr_score = 0.0
        self._compliance_score = 0.0
        self._financial_score = 0.0
        self._ceo_score = 0.0
        self._incident_score = 0.0

        templates = random.sample(PIPELINE_TEMPLATES, cfg["num_pipelines"])
        self._pipelines = [
            PipelineStatus(
                pipeline_id=pid, name=name, broken=True,
                rows_affected=random.randint(500, 50_000),
                error_msg=err,
            )
            for pid, name, err in templates
        ]

        hr_templates = random.sample(HR_TEMPLATES, cfg["num_hr"])
        self._hr_conflicts = [
            HRConflict(
                conflict_id=cid, employee_a=a, employee_b=b,
                severity=sev, resolved=False, morale_impact=-0.15,
            )
            for cid, a, b, sev, _ in hr_templates
        ]

        tkt_templates = random.sample(TICKET_TEMPLATES, cfg["num_tickets"])
        self._tickets = [
            CustomerTicket(
                ticket_id=tid, customer=cust, issue=issue,
                sla_seconds_remaining=cfg["sla_seconds"],
                linked_pipeline_id=linked,
                resolved=False, satisfaction=None,
            )
            for tid, cust, issue, linked in tkt_templates
        ]

        aud_templates = random.sample(
            AUDIT_TEMPLATES, min(cfg["num_audits"], len(AUDIT_TEMPLATES))
        )
        self._audit_flags = [
            AuditFlag(
                flag_id=fid, category=cat, description=desc,
                resolved=False, severity=sev,
            )
            for fid, cat, desc, sev in aud_templates
        ]

        ceo_q = random.sample(CEO_QUESTIONS, cfg["num_ceo"])
        self._ceo_questions = [
            CEOQuestion(question_id=qid, question=q, answered=False, correct=None)
            for qid, q in ceo_q
        ]

        inc_templates = random.sample(
            INCIDENT_TITLES, min(cfg["num_incidents"], len(INCIDENT_TITLES))
        )
        self._incidents = [
            Incident(
                incident_id=iid, source_system=src, title=title,
                priority=pri, open=True, age_seconds=0.0,
            )
            for iid, src, title, pri in inc_templates
        ]

        self._last_result = (
            f"CORTEX episode {self._episode_id} started [{self._difficulty.upper()}]. "
            f"{len(self._pipelines)} pipelines broken, "
            f"{len(self._hr_conflicts)} HR conflicts, "
            f"{len(self._tickets)} customer tickets. "
            f"Use inspect_systems to get a full status report."
        )

        return self._build_observation()

    async def step(self, action: CortexAction) -> StepResult:
        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs, reward=0.0, done=True,
                info={"error": "Episode already done. Call reset()."}
            )

        self._step += 1
        self._tick_sla()

        reward, msg = self._dispatch(action)
        self._last_result = msg
        self._cumulative_reward += reward

        done = self._check_done()
        self._done = done

        obs = self._build_observation()
        return StepResult(observation=obs, reward=reward, done=done, info={})

    @property
    def state(self) -> CortexState:
        return CortexState(
            observation=self._build_observation(),
            scores=self._score_dict(),
            done=self._done,
            episode_id=self._episode_id,
        )

    def _tick_sla(self) -> None:
        for ticket in self._tickets:
            if not ticket.resolved:
                ticket.sla_seconds_remaining = max(
                    0.0, ticket.sla_seconds_remaining - 10.0
                )
                if ticket.sla_seconds_remaining == 0.0:
                    self._sla_breaches += 1
                    self._customer_score = max(-1.0, self._customer_score - 0.15)

        for inc in self._incidents:
            if inc.open:
                inc.age_seconds += 10.0

    def _dispatch(self, action: CortexAction) -> Tuple[float, str]:
        t = action.tool_name
        args = action.arguments

        if t not in VALID_TOOLS:
            return -0.05, f"Unknown tool '{t}'. Valid tools: {sorted(VALID_TOOLS)}"

        if t == "inspect_systems":
            return self._inspect_systems()
        elif t == "repair_pipeline":
            return self._repair_pipeline(args.get("pipeline_id", ""))
        elif t == "resolve_hr":
            return self._resolve_hr(
                args.get("conflict_id", ""),
                args.get("decision", "mediate"),
            )
        elif t == "handle_ticket":
            return self._handle_ticket(
                args.get("ticket_id", ""),
                args.get("resolution", ""),
            )
        elif t == "run_audit":
            return self._run_audit(args.get("flag_id", ""))
        elif t == "brief_ceo":
            return self._brief_ceo(
                args.get("question_id", ""),
                args.get("answer", ""),
            )
        elif t == "open_incident":
            return self._open_incident(args.get("source_system", ""))
        elif t == "close_incident":
            return self._close_incident(args.get("incident_id", ""))
        elif t == "escalate":
            return self._escalate(args.get("incident_id", ""))
        else:
            return -0.02, "no_op taken. Time is passing — SLA timers tick."

    def _inspect_systems(self) -> Tuple[float, str]:
        broken = [p.name for p in self._pipelines if p.broken]
        open_hr = [c.conflict_id for c in self._hr_conflicts if not c.resolved]
        urgent_tkts = [
            t.ticket_id for t in self._tickets
            if not t.resolved and t.sla_seconds_remaining < 60
        ]
        unresolved_audit = [a.flag_id for a in self._audit_flags if not a.resolved]
        pending_ceo = [q.question_id for q in self._ceo_questions if not q.answered]
        open_inc = [i.incident_id for i in self._incidents if i.open]

        msg = (
            f"SYSTEM STATUS:\n"
            f"  Data Ops   -> {len(broken)} broken pipelines: {broken}\n"
            f"  HR         -> {len(open_hr)} open conflicts: {open_hr} | morale={self._team_morale:.2f}\n"
            f"  CustSupport-> {len(urgent_tkts)} URGENT tickets (SLA<60s): {urgent_tkts}\n"
            f"  Audit      -> {len(unresolved_audit)} unresolved flags: {unresolved_audit}\n"
            f"  CEO        -> {len(pending_ceo)} unanswered questions: {pending_ceo} | trust={self._ceo_trust:.2f}\n"
            f"  Incidents  -> {len(open_inc)} open: {open_inc}"
        )
        return 0.02, msg

    def _repair_pipeline(self, pipeline_id: str) -> Tuple[float, str]:
        for p in self._pipelines:
            if p.pipeline_id == pipeline_id:
                if not p.broken:
                    return -0.02, f"Pipeline {pipeline_id} is already healthy."
                p.broken = False
                self._data_repair_score += 0.25
                resolved_tkts = []
                for t in self._tickets:
                    if t.linked_pipeline_id == pipeline_id and not t.resolved:
                        t.resolved = True
                        t.satisfaction = round(0.6 + t.sla_seconds_remaining / 600, 2)
                        self._customer_score += 0.15
                        resolved_tkts.append(t.ticket_id)
                cascade = (
                    f" Cascade: auto-resolved tickets {resolved_tkts}."
                    if resolved_tkts else ""
                )
                return (
                    0.3,
                    f"Pipeline '{p.name}' repaired. Rows back online: {p.rows_affected}.{cascade}",
                )
        return -0.05, f"No pipeline with id '{pipeline_id}'. Check inspect_systems."

    def _resolve_hr(self, conflict_id: str, decision: str) -> Tuple[float, str]:
        for c in self._hr_conflicts:
            if c.conflict_id == conflict_id:
                if c.resolved:
                    return -0.02, f"Conflict {conflict_id} already resolved."
                c.resolved = True
                if c.severity == "high" and decision not in ("mediate", "terminate_b"):
                    self._team_morale = max(0.0, self._team_morale - 0.1)
                    self._hr_score += 0.1
                    return (
                        0.1,
                        f"Conflict {conflict_id} closed but handling was suboptimal "
                        f"for severity '{c.severity}'. Morale dipped.",
                    )
                else:
                    self._team_morale = min(1.0, self._team_morale + 0.05)
                    self._hr_score += 0.25
                    return (
                        0.25,
                        f"HR conflict {conflict_id} resolved via '{decision}'. "
                        f"Morale: {self._team_morale:.2f}",
                    )
        return -0.05, f"No HR conflict '{conflict_id}'."

    def _handle_ticket(self, ticket_id: str, resolution: str) -> Tuple[float, str]:
        for t in self._tickets:
            if t.ticket_id == ticket_id:
                if t.resolved:
                    return -0.02, f"Ticket {ticket_id} already resolved."
                t.resolved = True
                sat = round(0.4 + t.sla_seconds_remaining / 400, 2)
                t.satisfaction = min(1.0, sat)
                self._customer_score += 0.2 * t.satisfaction
                return (
                    0.2 * t.satisfaction,
                    f"Ticket {ticket_id} resolved. Customer satisfaction: {t.satisfaction:.2f}. "
                    f"SLA time remaining was {t.sla_seconds_remaining:.0f}s.",
                )
        return -0.05, f"No ticket '{ticket_id}'."

    def _run_audit(self, flag_id: str) -> Tuple[float, str]:
        for a in self._audit_flags:
            if a.flag_id == flag_id:
                if a.resolved:
                    return -0.02, f"Audit flag {flag_id} already resolved."
                a.resolved = True
                sev_bonus = {"low": 0.1, "medium": 0.2, "high": 0.3, "critical": 0.4}
                bonus = sev_bonus[a.severity]
                self._compliance_score += bonus
                if a.category == "budget_overrun":
                    self._budget_health = min(1.0, self._budget_health + 0.1)
                    self._financial_score += 0.15
                elif a.category == "fraud_suspected":
                    self._financial_score += 0.3
                return (
                    bonus,
                    f"Audit flag {flag_id} ({a.severity} / {a.category}) resolved. "
                    f"Budget health: {self._budget_health:.2f}",
                )
        return -0.05, f"No audit flag '{flag_id}'."

    def _brief_ceo(self, question_id: str, answer: str) -> Tuple[float, str]:
        for q in self._ceo_questions:
            if q.question_id == question_id:
                if q.answered:
                    return -0.02, f"CEO question {question_id} already answered."
                q.answered = True
                keywords = ["pipeline", "morale", "audit", "ticket", "incident",
                            "sla", "fraud", "compliance", "budget", "resolved"]
                hits = sum(1 for kw in keywords if kw in answer.lower())
                correct = hits >= 2 and len(answer) >= 40
                q.correct = correct
                if correct:
                    self._ceo_trust = min(1.0, self._ceo_trust + 0.1)
                    self._ceo_score += 0.3
                    return (
                        0.3,
                        f"CEO accepted briefing on '{question_id}'. "
                        f"Trust: {self._ceo_trust:.2f}",
                    )
                else:
                    self._ceo_trust = max(0.0, self._ceo_trust - 0.1)
                    self._ceo_score += 0.05
                    return (
                        0.05,
                        f"CEO unsatisfied with '{question_id}'. "
                        f"Answer lacked specifics. Trust: {self._ceo_trust:.2f}",
                    )
        return -0.05, f"No CEO question '{question_id}'."

    def _open_incident(self, source_system: str) -> Tuple[float, str]:
        existing = {i.source_system for i in self._incidents if i.open}
        if source_system in existing:
            return -0.02, f"Incident for '{source_system}' already open."
        new_inc = Incident(
            incident_id=f"INC-{random.randint(100, 999)}",
            source_system=source_system,
            title=f"Agent-raised incident for {source_system}",
            priority="P3",
            open=True,
            age_seconds=0.0,
        )
        self._incidents.append(new_inc)
        self._incident_score += 0.1
        return 0.1, f"Incident opened for '{source_system}' -> {new_inc.incident_id}."

    def _close_incident(self, incident_id: str) -> Tuple[float, str]:
        for inc in self._incidents:
            if inc.incident_id == incident_id:
                if not inc.open:
                    return -0.02, f"Incident {incident_id} already closed."
                inc.open = False
                self._incident_score += 0.15
                return (
                    0.15,
                    f"Incident {incident_id} closed. Age: {inc.age_seconds:.0f}s.",
                )
        return -0.05, f"No incident '{incident_id}'."

    def _escalate(self, incident_id: str) -> Tuple[float, str]:
        for inc in self._incidents:
            if inc.incident_id == incident_id:
                if inc.priority == "P1":
                    return -0.02, f"Already P1 — cannot escalate further."
                prev = inc.priority
                levels = ["P4", "P3", "P2", "P1"]
                idx = levels.index(inc.priority)
                inc.priority = levels[idx + 1] if idx + 1 < 4 else "P1"
                self._incident_score += 0.05
                return (
                    0.05,
                    f"Incident {incident_id} escalated {prev} -> {inc.priority}.",
                )
        return -0.05, f"No incident '{incident_id}'."

    def _compute_final_reward(self) -> float:
        cfg = DIFFICULTY_CONFIG[self._difficulty]
        max_steps = cfg["max_steps"]

        dr = min(1.0, self._data_repair_score)
        cs = min(1.0, max(0.0, self._customer_score + 0.5))
        hr = min(1.0, self._hr_score)
        co = min(1.0, self._compliance_score)
        fi = min(1.0, self._financial_score + self._budget_health * 0.3)
        ce = min(1.0, self._ceo_score + self._ceo_trust * 0.3)
        im = min(1.0, self._incident_score)
        eff = max(0.0, 1.0 - self._step / max_steps)

        weighted = (
            dr * 0.20
            + cs * 0.20
            + hr * 0.15
            + co * 0.15
            + fi * 0.10
            + ce * 0.10
            + im * 0.05
            + eff * 0.05
        )
        penalty = self._sla_breaches * 0.03 + max(0, 0.5 - self._team_morale) * 0.1
        final = weighted - penalty
        return round(max(0.01, min(0.99, final)), 4)

    def _check_done(self) -> bool:
        cfg = DIFFICULTY_CONFIG[self._difficulty]
        if self._step >= cfg["max_steps"]:
            return True
        all_fixed = (
            all(not p.broken for p in self._pipelines)
            and all(c.resolved for c in self._hr_conflicts)
            and all(t.resolved for t in self._tickets)
            and all(a.resolved for a in self._audit_flags)
            and all(q.answered for q in self._ceo_questions)
            and all(not i.open for i in self._incidents)
        )
        return all_fixed

    def _score_dict(self) -> Dict[str, float]:
        return {
            "data_repair": round(self._data_repair_score, 3),
            "customer": round(self._customer_score, 3),
            "hr": round(self._hr_score, 3),
            "compliance": round(self._compliance_score, 3),
            "financial": round(self._financial_score, 3),
            "ceo": round(self._ceo_score, 3),
            "incident": round(self._incident_score, 3),
            "sla_breaches": self._sla_breaches,
            "team_morale": round(self._team_morale, 3),
            "ceo_trust": round(self._ceo_trust, 3),
        }

    def _build_observation(self) -> CortexObservation:
        cfg = DIFFICULTY_CONFIG[self._difficulty]
        open_incidents = [i for i in self._incidents if i.open]
        broken = [p for p in self._pipelines if p.broken]
        urgent = [t for t in self._tickets
                  if not t.resolved and t.sla_seconds_remaining < 60]

        if broken:
            hint = "Broken pipelines may cascade to customer tickets — try repair_pipeline first."
        elif urgent:
            hint = f"SLA critical on {urgent[0].ticket_id} — use handle_ticket immediately."
        elif any(not a.resolved for a in self._audit_flags if a.severity == "critical"):
            hint = "Critical audit flag open — run_audit for compliance points."
        elif any(not q.answered for q in self._ceo_questions):
            hint = "CEO is waiting — use brief_ceo with a detailed answer."
        elif open_incidents:
            hint = "Close resolved incidents with close_incident to earn efficiency points."
        else:
            hint = "Systems looking healthy — verify with inspect_systems before declaring done."

        return CortexObservation(
            step=self._step,
            max_steps=cfg["max_steps"],
            difficulty=self._difficulty,
            pipelines=self._pipelines,
            broken_pipeline_count=len(broken),
            hr_conflicts=self._hr_conflicts,
            team_morale=round(self._team_morale, 3),
            tickets=self._tickets,
            sla_breaches=self._sla_breaches,
            audit_flags=self._audit_flags,
            budget_health=round(self._budget_health, 3),
            ceo_questions=self._ceo_questions,
            ceo_trust=round(self._ceo_trust, 3),
            incidents=self._incidents,
            open_incident_count=len(open_incidents),
            last_action_result=self._last_result,
            cumulative_reward=round(self._cumulative_reward, 4),
            tool_hint=hint,
        )