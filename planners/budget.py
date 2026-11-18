"""
planners/budget.py — Budget calculation from extracted claims.

Rules
-----
- Include ALL mandatory and committed expenses.
- Keep optional expenses separate from committed total.
- Reject optional additions that would cause the total to exceed the limit.
- Never silently remove mandatory expenses to make the budget fit.
- Calculate remaining = limit - committed_total.
- Report each line item so callers can audit the breakdown.
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


def plan_budget(
    claims: dict,
    venue_cost: float,
    food_cost: float,
    judge_cost: float,
) -> tuple[dict, list[str]]:
    """
    Compute the budget breakdown.

    Args:
        claims:     Active claims dict from SharedState.
        venue_cost: Pre-calculated venue cost (from planners/venue.py).
        food_cost:  Pre-calculated food cost (from planners/food.py).
        judge_cost: Pre-calculated judge cost (from planners/judging.py).

    Returns:
        (budget, violations)
        budget: dict with keys: limit, venue, food, judges,
                committed_other, total, remaining,
                optional_items, optional_total.
        violations: list of hard-constraint violation strings.
    """
    violations: list[str] = []

    # --- Budget limit ----------------------------------------------------
    budget_limit = 0.0
    for k in ("total_budget", "budget", "approved_budget", "budget_limit"):
        if k in claims:
            budget_limit = _num(claims[k]["value"])
            break

    # --- Pre-committed costs (object claim) ------------------------------
    committed_other = 0.0
    prizes_in_committed = False
    committed_raw = None
    for k in ("committed_costs", "pre_committed_costs", "fixed_costs"):
        if k in claims:
            committed_raw = claims[k]["value"]
            break

    if isinstance(committed_raw, dict):
        for item_key, item_val in committed_raw.items():
            committed_other += _num(item_val)
            if "prize" in str(item_key).lower():
                prizes_in_committed = True

    # --- Prize cost (separate claim, not double-counted) -----------------
    prize_cost = 0.0
    if not prizes_in_committed:
        for k in ("prize_cost", "prizes", "prize_money"):
            if k in claims:
                prize_cost = _num(claims[k]["value"])
                break

    # --- Any other individually committed line items ---------------------
    # Look for printing_cost, t_shirt_cost, etc., as separate claims.
    _COMMITTED_KEYS = {
        "printing_cost", "banner_cost", "t_shirt_cost", "certificate_cost",
        "photography_cost", "sound_cost", "equipment_cost", "misc_cost",
    }
    for k in _COMMITTED_KEYS:
        if k in claims:
            committed_other += _num(claims[k]["value"])

    committed_total = venue_cost + food_cost + judge_cost + committed_other + prize_cost

    # --- Optional items (not counted in total unless accepted) -----------
    optional_items: dict[str, float] = {}
    _OPTIONAL_KEYS = {
        "dj_cost", "photographer_cost", "extra_refreshments",
        "optional_equipment", "extra_staff_cost",
    }
    for k in _OPTIONAL_KEYS:
        if k in claims:
            optional_items[k] = _num(claims[k]["value"])

    optional_total = sum(optional_items.values())

    # --- Violations ------------------------------------------------------
    remaining = budget_limit - committed_total

    if budget_limit > 0 and remaining < 0:
        violations.append(
            f"Budget exceeded: committed Rs {committed_total:,.0f} against "
            f"limit Rs {budget_limit:,.0f} (over by Rs {abs(remaining):,.0f})."
        )

    if budget_limit > 0 and optional_total > 0:
        projected = committed_total + optional_total
        if projected > budget_limit:
            over = projected - budget_limit
            violations.append(
                f"Optional expenses (Rs {optional_total:,.0f}) would exceed "
                f"the remaining budget by Rs {over:,.0f}; not included in total."
            )

    return {
        "limit":           budget_limit,
        "venue":           venue_cost,
        "food":            food_cost,
        "judges":          judge_cost,
        "committed_other": committed_other + prize_cost,
        "total":           committed_total,
        "remaining":       remaining,
        "optional_items":  optional_items,
        "optional_total":  optional_total,
    }, violations
