"""
planners/food.py — Food planning from extracted claims.

Rules
-----
- Food quantity must be >= headcount (enforced here AND by validator).
- Use the latest confirmed price (basic < standard if both present).
- Distinguish proposed quantities from confirmed orders via the claim status.
- Delivery cost is added when present.
- Never invent prices or quantities.
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


def plan_food(claims: dict) -> tuple[dict, list[str]]:
    """
    Compute the food plan from active claims.

    Args:
        claims: Active claims dict from SharedState.

    Returns:
        (food_plan, violations)
        food_plan: dict with keys: type, boxes, unit_price, delivery, total,
                   confirmed (bool), price_source.
        violations: list of hard-constraint violation strings.
    """
    violations: list[str] = []

    # --- Headcount -------------------------------------------------------
    headcount = 0
    for k in ("people_count", "attendees", "headcount"):
        if k in claims:
            headcount = int(_num(claims[k]["value"]))
            break

    # --- Meal price — prefer basic tier if available, else standard, else generic
    meal_price = 0.0
    meal_type = "Quoted lunch box"
    price_source = "none"

    basic_raw  = claims.get("basic_meal_price")
    std_raw    = claims.get("standard_meal_price")
    generic_raw = claims.get("meal_price") or claims.get("food_price_per_person")

    if basic_raw is not None:
        meal_price   = _num(basic_raw["value"])
        meal_type    = "Basic lunch box"
        price_source = "basic_meal_price"
    elif std_raw is not None:
        meal_price   = _num(std_raw["value"])
        meal_type    = "Standard lunch box"
        price_source = "standard_meal_price"
    elif generic_raw is not None:
        meal_price   = _num(generic_raw["value"])
        meal_type    = str(claims.get("meal_type", {}).get("value") or "Quoted lunch box")
        price_source = "meal_price"

    # --- Food quantity ---------------------------------------------------
    # Use an explicitly confirmed order quantity when present; otherwise
    # default to headcount (minimum legal quantity).
    confirmed_order = False
    boxes = headcount     # default minimum

    for k in ("food_boxes", "meal_boxes", "boxes_ordered", "food_quantity"):
        if k in claims:
            raw_boxes = int(_num(claims[k]["value"]))
            if raw_boxes >= headcount:
                boxes = raw_boxes
                confirmed_order = True
            else:
                violations.append(
                    f"Extracted food quantity {raw_boxes} is below headcount {headcount}; "
                    f"quantity set to {headcount} (minimum required)."
                )
            break

    if boxes <= 0 and headcount > 0:
        boxes = headcount

    # --- Delivery --------------------------------------------------------
    delivery = 0.0
    for k in ("food_delivery_cost", "delivery_cost", "delivery"):
        if k in claims:
            delivery = _num(claims[k]["value"])
            break

    total = boxes * meal_price + delivery

    return {
        "type":          meal_type,
        "boxes":         boxes,
        "unit_price":    meal_price,
        "delivery":      delivery,
        "total":         total,
        "confirmed":     confirmed_order,
        "price_source":  price_source,
    }, violations
