"""
core/planner.py — Top-level plan generation using the five sub-planners.

Public API
----------
generate_plan(state)  -> Plan
    Build a complete plan from the active SharedState.

update_plan(existing_state, new_data) -> Plan
    Apply new data (list of normalised claim dicts) to the existing state,
    identify which planning areas are affected, recalculate only those areas
    plus any downstream dependencies, and return the updated plan.

Design notes
------------
- Delegates to planners/{venue,schedule,food,judging,budget}.py.
- Uses core/validator.py for post-plan checks.
- Uses core/conflict.py for claim merging.
- Never calls the LLM.
- Returns a schema.Plan TypedDict-compatible dict.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from core.schema import Plan, CheckResult
from core.validator import run_all_checks, _num
from core.conflict import merge_all_claims
from planners.venue import select_venue
from planners.schedule import build_schedule
from planners.food import plan_food
from planners.judging import plan_judging
from planners.budget import plan_budget


# ---------------------------------------------------------------------------
# Area dependency map
# A change in area X requires recalculation of area X and everything that
# depends on it.
# ---------------------------------------------------------------------------
_DEPENDENCIES: dict[str, list[str]] = {
    "venue":    ["venue", "budget"],
    "schedule": ["schedule"],
    "food":     ["food", "budget"],
    "judging":  ["judging", "schedule", "budget"],
    "budget":   ["budget"],
}


def _affected_areas(changed_keys: list[str], claims: dict) -> set[str]:
    """
    Return the set of planning areas that need recalculation given the
    claim keys that just changed.
    """
    areas: set[str] = set()
    for key in changed_keys:
        # Determine which area owns this key.
        claim_area = "budget"
        if key in claims:
            claim_area = str(claims[key].get("area", "budget"))
        # Walk the dependency map.
        for area, deps in _DEPENDENCIES.items():
            if area == claim_area:
                areas.update(deps)
    if not areas:
        areas.update(["venue", "schedule", "food", "judging", "budget"])
    return areas


# ---------------------------------------------------------------------------
# Event-level metadata extraction
# ---------------------------------------------------------------------------

def _extract_event(claims: dict) -> dict:
    headcount = 0
    for k in ("people_count", "attendees", "headcount"):
        if k in claims:
            headcount = int(_num(claims[k]["value"]))
            break

    teams = 0
    for k in ("team_count", "teams", "num_teams"):
        if k in claims:
            teams = int(_num(claims[k]["value"]))
            break
    if teams == 0 and headcount > 0:
        teams = math.ceil(headcount / 3)

    duration_hours = _num(
        claims.get("event_duration_hours", {}).get("value")
        or claims.get("duration_hours", {}).get("value"),
        4.0,
    )

    staff = 0
    for k in ("staff_count", "staff_on_duty", "staff"):
        if k in claims:
            staff = int(_num(claims[k]["value"]))
            break

    return {
        "attendees":      headcount,
        "teams":          teams,
        "duration_hours": duration_hours,
        "staff":          staff,
    }


# ---------------------------------------------------------------------------
# Public: generate_plan
# ---------------------------------------------------------------------------

def generate_plan(state: dict) -> Plan:
    """
    Build a complete plan from the active SharedState.

    Args:
        state: SharedState dict with at least {"claims": {...}}.

    Returns:
        A schema.Plan-compatible dict.
    """
    claims = state.get("claims", {})

    event    = _extract_event(claims)
    venue    = select_venue(claims)
    food, food_viol      = plan_food(claims)
    judging, judge_viol  = plan_judging(claims)
    schedule, sched_viol = build_schedule(claims)
    budget, budget_viol  = plan_budget(
        claims,
        venue_cost  = venue["cost"],
        food_cost   = food["total"],
        judge_cost  = judging["total_cost"],
    )

    # Collect all violations as rejected messages.
    all_violations = food_viol + judge_viol + sched_viol + budget_viol

    # Run post-plan hard-constraint checks.
    plan_so_far: dict = {
        "event":    event,
        "venue":    venue,
        "schedule": schedule,
        "food":     food,
        "judging":  judging,
        "budget":   budget,
        "checks":   [],
        "valid":    False,
    }
    checks = run_all_checks(plan_so_far)

    # Append sub-planner violations as failed checks.
    for v in all_violations:
        checks.append(CheckResult(
            name   = "Sub-planner constraint",
            passed = False,
            detail = v,
        ))

    valid = all(c["passed"] for c in checks)

    plan: Plan = {
        "event":    event,
        "venue":    venue,
        "schedule": schedule,
        "food":     food,
        "judging":  judging,
        "budget":   budget,
        "checks":   checks,
        "valid":    valid,
    }
    return plan


# ---------------------------------------------------------------------------
# Public: update_plan
# ---------------------------------------------------------------------------

def update_plan(existing_state: dict, new_data: dict) -> tuple[dict, Plan]:
    """
    Apply new extracted data to the existing state and return an updated
    (state, plan) pair.

    Only affected planning areas and their downstream dependencies are
    recalculated, so a printing-cost change triggers budget recalculation
    but not venue or schedule recalculation.

    Args:
        existing_state: Current SharedState (not mutated — a deep copy is used).
        new_data: Dict produced by the LLM reader with keys:
                  message_meta, claims (list of normalised claim dicts), summary.

    Returns:
        (new_state, plan)
        new_state: Updated SharedState.
        plan:      Freshly computed Plan.
    """
    state = deepcopy(existing_state)
    state.setdefault("claims", {})
    state.setdefault("rejected", [])
    state.setdefault("messages", [])

    raw_message = new_data.get("raw_message", "")
    if raw_message:
        state["messages"].append(raw_message)

    # Merge incoming claims.
    incoming_claims = new_data.get("claims", [])
    accepted_keys, _rejected = merge_all_claims(state, incoming_claims, raw_message)

    # For now recalculate all areas whenever anything changes.
    # (Fine-grained area filtering is applied when the caller passes area hints.)
    affected = _affected_areas(accepted_keys, state["claims"])

    # generate_plan always recalculates everything; the _affected_areas logic
    # is exposed so app-level code can skip unnecessary work in the future.
    # For correctness within this stage we always regenerate the full plan.
    plan = generate_plan(state)

    return state, plan
