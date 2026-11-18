"""
planners/venue.py — Venue selection from extracted claims.

Rules
-----
- Required capacity  = headcount extracted from claims.
- Feasible venues    = all options whose capacity >= required capacity.
- Selected venue     = lowest-cost feasible option.
- Mandatory facility = standby_generator (if required by claims).
- If no feasible venue exists, the plan records the best available and
  the validator will flag the failure — this module never silently
  invents a venue.
- No hardcoded venue names, prices, or capacities.
"""

from __future__ import annotations

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


def _requires_generator(claims: dict) -> bool:
    """Return True if any claim says standby_generator is required/true."""
    raw = None
    for k in ("standby_generator", "generator_required", "generator"):
        if k in claims:
            raw = claims[k].get("value")
            break
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).lower() in ("true", "yes", "1", "required")


def select_venue(claims: dict) -> dict:
    """
    Select the best venue from claims and return a venue dict.

    Args:
        claims: The active claims dict from SharedState.

    Returns:
        dict with keys: name, capacity, cost, standby_generator, feasible.
        feasible=False means the selected venue does not meet capacity.
    """
    # Determine required headcount.
    headcount = 0
    for k in ("people_count", "attendees", "headcount"):
        if k in claims:
            headcount = int(_num(claims[k]["value"]))
            break

    need_generator = _requires_generator(claims)

    # Build the candidate list from venue_options (array of venue dicts).
    candidates: list[dict] = []
    venues_raw = None
    for k in ("venue_options", "venues", "hall_options"):
        if k in claims:
            venues_raw = claims[k]["value"]
            break

    if isinstance(venues_raw, dict):
        venues_raw = [venues_raw]

    if isinstance(venues_raw, list):
        for item in venues_raw:
            if isinstance(item, dict):
                candidates.append(item)

    # If individual venue claims are present (venue_name / venue_capacity /
    # venue_cost), synthesise a single candidate from them.
    if not candidates:
        name = None
        cap = 0
        cost = 0.0
        gen = False
        for k in ("venue_name", "hall_name"):
            if k in claims:
                name = str(claims[k]["value"])
                break
        for k in ("venue_capacity", "hall_capacity"):
            if k in claims:
                cap = int(_num(claims[k]["value"]))
                break
        for k in ("venue_cost", "hall_cost", "venue_rate", "hall_rate"):
            if k in claims:
                cost = _num(claims[k]["value"])
                break
        for k in ("standby_generator", "generator"):
            if k in claims:
                v = claims[k]["value"]
                gen = str(v).lower() in ("true", "yes", "1", "required") if not isinstance(v, bool) else v
                break
        if name or cap or cost:
            candidates.append({
                "name": name or "Extracted venue",
                "capacity": cap,
                "cost": cost,
                "generator": gen,
            })

    if not candidates:
        return {
            "name": "No venue extracted",
            "capacity": 0,
            "cost": 0.0,
            "standby_generator": False,
            "feasible": False,
        }

    # Filter: capacity >= headcount, honour generator requirement.
    feasible = []
    for v in candidates:
        cap = int(_num(v.get("capacity", 0)))
        gen = v.get("generator", False)
        if isinstance(gen, str):
            gen = gen.lower() in ("true", "yes", "1")
        if cap >= headcount:
            if need_generator and not gen:
                continue          # mandatory facility not met
            feasible.append(v)

    pool = feasible if feasible else candidates   # fall back to all if none feasible
    is_feasible = bool(feasible)

    # Select lowest-cost option.
    selected = sorted(pool, key=lambda x: _num(x.get("cost", 0)))[0]

    gen_val = selected.get("generator", False)
    if isinstance(gen_val, str):
        gen_val = gen_val.lower() in ("true", "yes", "1")

    return {
        "name": str(selected.get("name", "Unknown")),
        "capacity": int(_num(selected.get("capacity", 0))),
        "cost": _num(selected.get("cost", 0)),
        "standby_generator": bool(gen_val),
        "feasible": is_feasible,
    }
