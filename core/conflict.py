"""
core/conflict.py — Conflict detection and claim-merge logic.

Responsibilities
----------------
- Compare an incoming normalised claim against the existing state.
- Classify the relationship: new, correction, confirmation, conflict, or stale.
- Decide whether to accept or reject the incoming claim.
- Record rejections with a human-readable reason.
- Enforce food-quantity hard constraint before any date/confidence logic.
- Never call the LLM; never reference hardcoded event facts.

This module replaces and extends the inline merge logic in planner.merge_claims
so the planning engine can use it without depending on the legacy planner module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.schema import Claim, RejectedItem

# Statuses that cannot overwrite a confirmed claim unless newer.
_UNCONFIRMED_STATUSES = {"suggestion", "tentative", "question", "unconfirmed"}


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(s: Any) -> datetime:
    """Parse a date string to datetime; fall back to 2026-01-01."""
    for fmt in ("%d %b %Y", "%d %B %Y", "%d %b", "%d %B", "%Y-%m-%d"):
        try:
            d = datetime.strptime(str(s).strip(), fmt)
            return d.replace(year=2026) if d.year == 1900 else d
        except ValueError:
            pass
    return datetime(2026, 1, 1)


# ---------------------------------------------------------------------------
# Numeric helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

DECISION_ACCEPT     = "accept"
DECISION_REJECT     = "reject"


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def evaluate_claim(
    key: str,
    incoming: dict,
    existing_claims: dict[str, Claim],
    message_text: str,
) -> tuple[str, str]:
    """
    Decide whether to accept or reject an incoming claim.

    Args:
        key:             Normalised snake_case claim key.
        incoming:        Dict with keys: value, area, status, confidence, rationale, date, sender.
        existing_claims: Current active claims dict.
        message_text:    Raw message text (for rejected item records).

    Returns:
        (decision, reason)
        decision: DECISION_ACCEPT or DECISION_REJECT
        reason:   Human-readable explanation (empty string when accepting a new claim).
    """
    value      = incoming.get("value")
    status     = str(incoming.get("status", "confirmed")).lower()
    confidence = float(incoming.get("confidence") or 0.7)
    sender     = str(incoming.get("sender", "Unknown"))
    date_str   = incoming.get("date", "1 Jan 2026")
    msg_date   = _parse_date(date_str)

    # --- Hard constraint: food quantity cannot be below headcount ----------------
    _FOOD_QTY_KEYS = {"food_boxes", "meal_boxes", "food_quantity", "boxes_ordered"}
    if key in _FOOD_QTY_KEYS:
        # Try to find the headcount in existing claims.
        headcount_raw = None
        for hk in ("people_count", "attendees", "headcount"):
            if hk in existing_claims:
                headcount_raw = existing_claims[hk]["value"]
                break
        if headcount_raw is not None:
            headcount = int(_num(headcount_raw))
            if headcount > 0 and int(_num(value)) < headcount:
                return (
                    DECISION_REJECT,
                    f"Rejected: {value} food boxes cannot feed {headcount} people. "
                    f"Quantity must be at least {headcount}.",
                )

    # --- Unconfirmed / tentative claims are never applied to the plan ------------
    if status in _UNCONFIRMED_STATUSES:
        return (
            DECISION_REJECT,
            f"Not applied: claim has status '{status}', not an approved/confirmed fact.",
        )

    # --- New claim (no existing entry) — accept unconditionally -----------------
    if key not in existing_claims:
        return DECISION_ACCEPT, ""

    # --- Existing claim present — compare ----------------------------------------
    old = existing_claims[key]
    old_date = _parse_date(old.get("date", "1 Jan 2026"))
    old_confidence = float(old.get("confidence") or 0.7)
    old_value = old.get("value")

    # Same value — always accept (confirmation).
    if old_value == value:
        return DECISION_ACCEPT, ""

    # Incoming is older than the active claim → stale, reject.
    if msg_date < old_date:
        return (
            DECISION_REJECT,
            f"Rejected as outdated: {sender}'s {msg_date:%d %b} claim predates "
            f"the active {old_date:%d %b} claim.",
        )

    # Same date but incoming has no stronger confidence → reject.
    if msg_date == old_date and confidence <= old_confidence:
        return (
            DECISION_REJECT,
            "Rejected: same-date conflicting claim has no stronger confidence or authority.",
        )

    # Incoming is newer (or same date with higher confidence) → accept.
    return DECISION_ACCEPT, ""


# ---------------------------------------------------------------------------
# Batch merge (used by core/updater.py and core/planner.py)
# ---------------------------------------------------------------------------

def merge_all_claims(
    state: dict,
    claims: list[dict],
    message_text: str,
) -> tuple[list[str], list[RejectedItem]]:
    """
    Merge a list of normalised claim dicts into state["claims"].
    Returns (accepted_keys, rejected_items).

    state must have keys: "claims" and "rejected".
    Each claim dict must have: key, value, area, status, confidence,
    rationale, date, sender.
    """
    state.setdefault("claims", {})
    state.setdefault("rejected", [])

    accepted: list[str] = []
    rejected_items: list[RejectedItem] = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        key = str(claim.get("key", "")).strip().lower()
        if not key:
            continue

        decision, reason = evaluate_claim(
            key, claim, state["claims"], message_text
        )

        if decision == DECISION_ACCEPT:
            state["claims"][key] = Claim(
                value      = claim.get("value"),
                area       = claim.get("area", "budget"),   # type: ignore[arg-type]
                date       = str(claim.get("date", "1 Jan 2026")),
                sender     = str(claim.get("sender", "Unknown")),
                confidence = float(claim.get("confidence") or 0.7),
                rationale  = str(claim.get("rationale", "")),
            )
            accepted.append(key)
        else:
            item = RejectedItem(
                message = message_text,
                key     = key,
                value   = claim.get("value"),
                reason  = reason,
            )
            state["rejected"].append(item)
            rejected_items.append(item)

    return accepted, rejected_items
