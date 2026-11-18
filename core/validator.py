"""
core/validator.py — Pure-Python validation rules for the event planning engine.

Responsibilities
----------------
- Accept or reject individual claim values against hard constraints.
- Run post-plan checks that produce CheckResult dicts.
- Never call the LLM; never reference hardcoded event facts.

All functions are stateless: they receive their inputs as arguments and
return structured results. The caller decides what to do with failures.
"""

from __future__ import annotations

import math
from typing import Any

from core.schema import CheckResult


# ---------------------------------------------------------------------------
# Numeric helper (shared with planner)
# ---------------------------------------------------------------------------

def _num(v: Any, default: float = 0.0) -> float:
    """
    Coerce v to float. Strips currency symbols and thousands separators.
    Returns default on failure.
    """
    try:
        if isinstance(v, (int, float)):
            return float(v)
        import re
        s = str(v).replace(",", "").replace("Rs", "").replace("PKR", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Hard-constraint validators — return (ok: bool, reason: str)
# ---------------------------------------------------------------------------

def validate_food_quantity(food_boxes: Any, headcount: Any) -> tuple[bool, str]:
    """Food boxes must be >= headcount."""
    boxes = int(_num(food_boxes))
    people = int(_num(headcount))
    if people <= 0:
        return True, "Headcount not set; food quantity unchecked."
    if boxes < people:
        return (
            False,
            f"{boxes} food boxes cannot feed {people} people. "
            f"Quantity must be at least {people}.",
        )
    return True, f"{boxes} boxes for {people} people — sufficient."


def validate_venue_capacity(capacity: Any, headcount: Any) -> tuple[bool, str]:
    """Venue capacity must be >= headcount."""
    cap = int(_num(capacity))
    people = int(_num(headcount))
    if people <= 0:
        return True, "Headcount not set; venue capacity unchecked."
    if cap <= 0:
        return False, "No venue capacity value available."
    if cap < people:
        return (
            False,
            f"Venue capacity {cap} is below required headcount {people}.",
        )
    return True, f"{cap} seats for {people} attendees — sufficient."


def validate_budget(total_cost: Any, budget_limit: Any) -> tuple[bool, str]:
    """Total committed cost must not exceed the approved budget limit."""
    cost = _num(total_cost)
    limit = _num(budget_limit)
    if limit <= 0:
        return True, "Budget limit not set; cost unchecked."
    remaining = limit - cost
    if remaining < 0:
        return (
            False,
            f"Total cost Rs {cost:,.0f} exceeds budget of Rs {limit:,.0f} "
            f"by Rs {abs(remaining):,.0f}.",
        )
    return True, f"Rs {cost:,.0f} against Rs {limit:,.0f}; Rs {remaining:,.0f} remaining."


def validate_judge_count(
    judges: Any,
    teams: Any,
    max_teams_per_judge: Any,
) -> tuple[bool, str]:
    """
    Confirm enough judges to cover all teams.
    Required judges = ceil(teams / max_teams_per_judge).
    """
    j = int(_num(judges))
    t = int(_num(teams))
    mtpj = int(_num(max_teams_per_judge, 1)) or 1
    required = math.ceil(t / mtpj) if t > 0 else 0
    if j < required:
        return (
            False,
            f"{j} judges available but {required} required for {t} teams "
            f"at max {mtpj} teams/judge.",
        )
    return (
        True,
        f"{j} judges cover {t} teams at max {mtpj}/judge "
        f"(minimum required: {required}).",
    )


def validate_judging_time(
    judges: Any,
    available_judges: Any,
    teams: Any,
    minutes_per_team: Any,
    available_judging_minutes: Any,
    late_minutes: Any = 0,
) -> tuple[bool, str]:
    """
    Confirm that all teams can be judged within the available window.

    Each judge handles ceil(teams / effective_judges) teams sequentially.
    If a judge is late, the effective window for that judge is reduced.
    """
    j = int(_num(judges))
    avail = int(_num(available_judges)) if _num(available_judges) > 0 else j
    t = int(_num(teams))
    mpt = _num(minutes_per_team, 15)
    window = _num(available_judging_minutes)
    late = _num(late_minutes, 0)

    if avail <= 0 or t <= 0 or window <= 0:
        return True, "Judging time check skipped — insufficient data."

    teams_per_judge = math.ceil(t / avail)
    time_needed = teams_per_judge * mpt
    effective_window = window - late  # most conservative: one judge is late

    if time_needed > effective_window:
        return (
            False,
            f"{t} teams × {mpt} min/team = {time_needed} min needed; "
            f"only {effective_window:.0f} min available after {late:.0f} min delay.",
        )
    return (
        True,
        f"{t} teams × {mpt} min/team = {time_needed} min; "
        f"window {effective_window:.0f} min — fits.",
    )


def validate_schedule_duration(
    schedule_items: list[dict],
    required_hours: Any,
) -> tuple[bool, str]:
    """
    Validate that the schedule fits within the required event duration.
    schedule_items: list of {time, activity} where time is "HH:MM–HH:MM".
    """
    import re
    required_minutes = _num(required_hours, 4) * 60
    total = 0.0
    for item in schedule_items:
        time_str = str(item.get("time", ""))
        m = re.match(r"(\d+):(\d+)[–-](\d+):(\d+)", time_str)
        if m:
            start = int(m.group(1)) * 60 + int(m.group(2))
            end = int(m.group(3)) * 60 + int(m.group(4))
            if end > start:
                total += end - start

    if total <= 0:
        return True, "Schedule duration check skipped — no parseable time slots."
    if total > required_minutes + 1:          # 1-minute tolerance for rounding
        return (
            False,
            f"Schedule totals {total:.0f} min but event duration is "
            f"{required_minutes:.0f} min ({required_hours} hours).",
        )
    return (
        True,
        f"Schedule totals {total:.0f} min within {required_minutes:.0f} min event.",
    )


def validate_no_overlapping_slots(schedule_items: list[dict]) -> tuple[bool, str]:
    """Detect overlapping time slots in the schedule."""
    import re
    slots: list[tuple[int, int, str]] = []
    for item in schedule_items:
        time_str = str(item.get("time", ""))
        m = re.match(r"(\d+):(\d+)[–-](\d+):(\d+)", time_str)
        if m:
            start = int(m.group(1)) * 60 + int(m.group(2))
            end = int(m.group(3)) * 60 + int(m.group(4))
            slots.append((start, end, item.get("activity", time_str)))

    slots.sort()
    for i in range(1, len(slots)):
        prev_end = slots[i - 1][1]
        cur_start = slots[i][0]
        if cur_start < prev_end:
            return (
                False,
                f"Overlap: '{slots[i-1][2]}' ends at {prev_end} min but "
                f"'{slots[i][2]}' starts at {cur_start} min.",
            )
    return True, "No overlapping schedule slots."


# ---------------------------------------------------------------------------
# Aggregate plan checker — returns a list of CheckResult
# ---------------------------------------------------------------------------

def run_all_checks(plan: dict) -> list[CheckResult]:
    """
    Run every hard-constraint check against a computed plan.
    Returns a list of CheckResult dicts with name, passed, detail.
    """
    results: list[CheckResult] = []

    ev = plan.get("event", {})
    headcount = ev.get("attendees", 0)
    teams = ev.get("teams", 0)
    duration_hours = ev.get("duration_hours", 4)

    venue = plan.get("venue", {})
    food = plan.get("food", {})
    judging = plan.get("judging", {})
    budget = plan.get("budget", {})
    schedule = plan.get("schedule", [])

    # 1. Venue capacity
    ok, detail = validate_venue_capacity(venue.get("capacity", 0), headcount)
    results.append(CheckResult(name="Venue capacity", passed=ok, detail=detail))

    # 2. Food quantity
    ok, detail = validate_food_quantity(food.get("boxes", 0), headcount)
    results.append(CheckResult(name="Food quantity", passed=ok, detail=detail))

    # 3. Budget
    ok, detail = validate_budget(budget.get("total", 0), budget.get("limit", 0))
    results.append(CheckResult(name="Budget", passed=ok, detail=detail))

    # 4. Judge count
    ok, detail = validate_judge_count(
        judging.get("judges", 0),
        teams,
        judging.get("max_teams_per_judge", 1),
    )
    results.append(CheckResult(name="Judging capacity", passed=ok, detail=detail))

    # 5. Judging time
    judging_slot_minutes = 0.0
    import re
    for slot in schedule:
        if "judg" in str(slot.get("activity", "")).lower():
            m = re.match(r"(\d+):(\d+)[–-](\d+):(\d+)", str(slot.get("time", "")))
            if m:
                s = int(m.group(1)) * 60 + int(m.group(2))
                e = int(m.group(3)) * 60 + int(m.group(4))
                judging_slot_minutes += max(0, e - s)

    ok, detail = validate_judging_time(
        judging.get("judges", 0),
        judging.get("judges", 0),
        teams,
        judging.get("minutes_per_team", 15),
        judging_slot_minutes if judging_slot_minutes > 0 else 45,
        judging.get("late_arrival_buffer_minutes", 0),
    )
    results.append(CheckResult(name="Judging time", passed=ok, detail=detail))

    # 6. Schedule duration
    ok, detail = validate_schedule_duration(schedule, duration_hours)
    results.append(CheckResult(name="Schedule duration", passed=ok, detail=detail))

    # 7. No overlapping slots
    ok, detail = validate_no_overlapping_slots(schedule)
    results.append(CheckResult(name="No schedule overlaps", passed=ok, detail=detail))

    # 8. All areas covered
    all_covered = (
        bool(venue.get("name"))
        and food.get("boxes", 0) > 0
        and judging.get("judges", 0) > 0
        and len(schedule) > 0
        and budget.get("limit", 0) > 0
    )
    results.append(CheckResult(
        name="All areas covered",
        passed=all_covered,
        detail="Venue, food, judging, schedule, and budget all have values."
        if all_covered else "One or more planning areas are missing data.",
    ))

    return results
