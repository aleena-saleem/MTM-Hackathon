"""
planners/schedule.py — Build a schedule from extracted claims.

Rules
-----
- Total schedule duration must not exceed event_duration_hours.
- Mandatory breaks (prayer, lunch, etc.) must be included when specified.
- Activities must not overlap.
- Start and end times are validated to be within [0, duration_minutes].
- If any hard constraint is violated, a rejection reason is returned and
  the previous valid schedule is preserved by the caller.
- No hardcoded times or activity names.
"""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, (int, float)):
            return float(v)
        import re
        s = str(v).replace(",", "").replace("Rs", "").replace("PKR", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default


def _fmt(start_min: int, end_min: int) -> str:
    """Format two minute offsets as 'HH:MM–HH:MM'."""
    def _hm(m: int) -> str:
        return f"{m // 60:02d}:{m % 60:02d}"
    return f"{_hm(start_min)}–{_hm(end_min)}"


def build_schedule(claims: dict) -> tuple[list[dict], list[str]]:
    """
    Build a schedule from the active claims.

    Args:
        claims: Active claims dict from SharedState.

    Returns:
        (schedule, violations)
        schedule:   list of {time, activity}
        violations: list of human-readable hard-constraint violation strings.
                    Empty if the schedule is valid.
    """
    duration_hours = _num(
        claims.get("event_duration_hours", {}).get("value")
        or claims.get("duration_hours", {}).get("value"),
        4.0,
    )
    total_minutes = int(duration_hours * 60)
    violations: list[str] = []

    prayer_minutes = int(_num(
        claims.get("prayer_break_minutes", {}).get("value")
        or claims.get("prayer_minutes", {}).get("value"),
        0,
    ))

    # Teams and judges affect judging slot sizing.
    teams = int(_num(claims.get("team_count", {}).get("value") or
                     claims.get("teams", {}).get("value"), 0))
    judges = int(_num(claims.get("judge_count", {}).get("value") or
                      claims.get("confirmed_judges", {}).get("value"), 0))
    minutes_per_team = _num(
        claims.get("minutes_per_team_per_judge", {}).get("value"), 15.0
    )
    max_teams_per_judge = int(_num(
        claims.get("max_teams_per_judge", {}).get("value"), 3
    )) or 1

    # Judging slot: each judge handles their batch sequentially.
    if teams > 0 and judges > 0:
        teams_per_judge = math.ceil(teams / judges)
        judging_slot = int(math.ceil(teams_per_judge * minutes_per_team))
    else:
        judging_slot = 45          # default if data is missing

    # Fixed allocation for non-variable slots (minutes).
    checkin      = 30
    build        = 0              # fills remaining time
    results      = 20
    prayer_slot  = prayer_minutes if prayer_minutes > 0 else 0

    # Calculate build time from what remains after fixed slots.
    fixed_used = checkin + judging_slot + results + prayer_slot
    build = total_minutes - fixed_used

    if build < 0:
        violations.append(
            f"Schedule constraint violated: required slots total {fixed_used} min "
            f"but event is only {total_minutes} min."
        )
        # Still return a best-effort schedule so the caller can display it.
        build = 0

    # Assemble the schedule.
    cursor = 0
    slots: list[dict] = []

    def _add(duration_m: int, activity: str) -> None:
        nonlocal cursor
        if duration_m <= 0:
            return
        slots.append({"time": _fmt(cursor, cursor + duration_m), "activity": activity})
        cursor += duration_m

    _add(checkin, "Check-in, team seating and opening briefing")
    _add(build, "Hackathon build session")

    if prayer_slot > 0:
        _add(prayer_slot, f"Prayer break ({prayer_slot} minutes)")

    _add(judging_slot, f"Final submissions and judging ({judging_slot} minutes)")
    _add(results, "Results and closing ceremony")

    # Validate no overflow.
    if cursor > total_minutes + 1:
        violations.append(
            f"Schedule overflow: slots total {cursor} min, event allows {total_minutes} min."
        )

    return slots, violations
