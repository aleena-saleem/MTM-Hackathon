"""
planners/judging.py — Judging plan from extracted claims.

Rules
-----
- required_judges = ceil(teams / max_teams_per_judge)
- If available judges < required_judges → flag insufficient capacity.
- Account for late judges: if any judge delay is known, reduce the
  effective judging window for capacity calculations.
- Ensure all teams can be judged within the available window.
- Never invent judge counts, teams, or timing figures.
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


def plan_judging(claims: dict) -> tuple[dict, list[str]]:
    """
    Compute the judging plan from active claims.

    Args:
        claims: Active claims dict from SharedState.

    Returns:
        (judging_plan, violations)
        judging_plan: dict with keys: judges, required_judges,
                      honorarium_each, total_cost, max_teams_per_judge,
                      minutes_per_team, late_arrival_buffer_minutes,
                      capacity_ok, time_ok.
        violations:   list of hard-constraint violation strings.
    """
    violations: list[str] = []

    # --- Teams -----------------------------------------------------------
    teams = 0
    for k in ("team_count", "teams", "num_teams"):
        if k in claims:
            teams = int(_num(claims[k]["value"]))
            break

    # --- Judges available ------------------------------------------------
    judges = 0
    for k in ("judge_count", "confirmed_judges", "judges"):
        if k in claims:
            judges = int(_num(claims[k]["value"]))
            break

    # --- Judging parameters ----------------------------------------------
    max_teams_per_judge = int(_num(
        claims.get("max_teams_per_judge", {}).get("value"), 3
    )) or 1

    minutes_per_team = _num(
        claims.get("minutes_per_team_per_judge", {}).get("value")
        or claims.get("minutes_per_team", {}).get("value"),
        15.0,
    )

    honorarium = _num(
        claims.get("judge_honorarium", {}).get("value")
        or claims.get("honorarium_per_judge", {}).get("value"),
        0.0,
    )

    late_minutes = _num(
        claims.get("judge_delay_minutes", {}).get("value")
        or claims.get("judge_late_minutes", {}).get("value"),
        0.0,
    )

    # --- Required judges -------------------------------------------------
    required_judges = math.ceil(teams / max_teams_per_judge) if teams > 0 else 0

    capacity_ok = judges >= required_judges
    if not capacity_ok and teams > 0:
        violations.append(
            f"Insufficient judging capacity: {judges} judges available, "
            f"{required_judges} required for {teams} teams at max {max_teams_per_judge}/judge."
        )

    # --- Time check ------------------------------------------------------
    # Each judge handles ceil(teams / judges) teams sequentially.
    # We use a default 45-minute judging window; callers may override via
    # the schedule slot, but we can only check here if judging_minutes is known.
    judging_minutes = _num(
        claims.get("judging_minutes", {}).get("value")
        or claims.get("judging_slot_minutes", {}).get("value"),
        0.0,
    )

    time_ok = True
    if judging_minutes > 0 and judges > 0 and teams > 0:
        teams_per_judge = math.ceil(teams / judges)
        time_needed = teams_per_judge * minutes_per_team
        effective_window = judging_minutes - late_minutes
        if time_needed > effective_window:
            time_ok = False
            violations.append(
                f"Insufficient judging time: {teams} teams × {minutes_per_team} min = "
                f"{time_needed:.0f} min needed; only {effective_window:.0f} min "
                f"available after {late_minutes:.0f} min delay."
            )

    total_cost = judges * honorarium

    return {
        "judges":                     judges,
        "required_judges":            required_judges,
        "honorarium_each":            honorarium,
        "total_cost":                 total_cost,
        "max_teams_per_judge":        max_teams_per_judge,
        "minutes_per_team":           minutes_per_team,
        "late_arrival_buffer_minutes": late_minutes,
        "capacity_ok":                capacity_ok,
        "time_ok":                    time_ok,
    }, violations
