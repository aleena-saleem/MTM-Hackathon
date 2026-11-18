"""
core/schema.py — Shared JSON data structures for the MTM Hackathon project.

All three team members should import from here rather than defining
ad-hoc dicts. These are plain Python TypedDicts so they work without
any extra dependencies (no Pydantic required).

Top-level output document
--------------------------
{
  "claims":    { <key>: Claim },          # active/accepted claims
  "rejected":  [ RejectedItem, ... ],     # claims that were blocked
  "messages":  [ str, ... ],              # raw message texts in order
  "plan":      Plan | null,               # latest computed plan
  "trace":     [ TraceEntry, ... ]        # one entry per LLM call
}
"""

from __future__ import annotations
from typing import TypedDict, Literal, Any


# ---------------------------------------------------------------------------
# Individual claim stored in state["claims"][key]
# ---------------------------------------------------------------------------
class Claim(TypedDict):
    value: Any                                      # number | str | list | dict
    area: Literal["venue", "schedule", "food", "judging", "budget"]
    date: str                                       # "DD Mon YYYY"
    sender: str
    confidence: float                               # 0.0 – 1.0
    rationale: str


# ---------------------------------------------------------------------------
# A claim that was blocked by the merge/validation logic
# ---------------------------------------------------------------------------
class RejectedItem(TypedDict):
    message: str        # full raw message text
    key: str            # claim key that was rejected
    value: Any          # the value that was rejected
    reason: str         # human-readable explanation


# ---------------------------------------------------------------------------
# One entry in the model call trace
# ---------------------------------------------------------------------------
class TraceEntry(TypedDict):
    call: int                           # sequential call number (1-based)
    model: str                          # model identifier returned by the API
    area: str                           # primary area of the message
    input: str                          # raw message passed to the model
    output: dict                        # parsed JSON returned by the model
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float
    status: Literal["success", "error"]
    merge_decision: dict                # keys: accepted_keys, rejected, summary


# ---------------------------------------------------------------------------
# Budget breakdown inside Plan
# ---------------------------------------------------------------------------
class BudgetBreakdown(TypedDict):
    limit: float
    venue: float
    food: float
    judges: float
    committed_other: float
    total: float
    remaining: float


# ---------------------------------------------------------------------------
# Single validation check
# ---------------------------------------------------------------------------
class CheckResult(TypedDict):
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Full computed plan
# ---------------------------------------------------------------------------
class Plan(TypedDict):
    event: dict         # attendees, teams, duration_hours, staff
    venue: dict         # name, capacity, cost, standby_generator
    schedule: list      # list of {time, activity}
    food: dict          # type, boxes, unit_price, delivery, total
    judging: dict       # judges, required_judges, honorarium_each, ...
    budget: BudgetBreakdown
    checks: list[CheckResult]
    valid: bool


# ---------------------------------------------------------------------------
# Full shared state document
# ---------------------------------------------------------------------------
class SharedState(TypedDict):
    claims: dict[str, Claim]
    rejected: list[RejectedItem]
    messages: list[str]


# ---------------------------------------------------------------------------
# Full output document (state + plan + trace)
# ---------------------------------------------------------------------------
class OutputDocument(TypedDict):
    claims: dict[str, Claim]
    rejected: list[RejectedItem]
    messages: list[str]
    plan: Plan | None
    trace: list[TraceEntry]


# ---------------------------------------------------------------------------
# Factory helpers — create empty instances to initialise state
# ---------------------------------------------------------------------------
def empty_state() -> SharedState:
    """Return a blank SharedState ready for use."""
    return {"claims": {}, "rejected": [], "messages": []}


def empty_output() -> OutputDocument:
    """Return a blank OutputDocument."""
    return {"claims": {}, "rejected": [], "messages": [], "plan": None, "trace": []}
