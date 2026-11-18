"""
core/updater.py — Incremental update coordinator.

Public API
----------
apply_update(state, parsed_doc, raw_message) -> (state, Plan, list[str])
    Merge a single parsed LLM document into existing state.
    Identify affected areas, recalculate only those areas plus downstream
    dependencies, and return the updated state, the new plan, and a list
    of rejection reasons from this update.

Design notes
------------
- Wraps core/conflict.py (merge) and core/planner.py (plan generation).
- Uses _AREA_KEYS to map claim keys to planning areas without hardcoding
  event-specific facts.
- If accepted_keys is empty (nothing changed), returns the existing plan
  without recalculation — avoids unnecessary work.
- Preserves the last valid plan on hard-constraint failures.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.conflict import merge_all_claims
from core.planner import generate_plan, _affected_areas
from core.schema import Plan, RejectedItem


# ---------------------------------------------------------------------------
# Key → area mapping (without hardcoding event facts)
# ---------------------------------------------------------------------------

_KEY_AREA: dict[str, str] = {
    # venue
    "venue_name": "venue", "venue_capacity": "venue", "venue_cost": "venue",
    "venue_options": "venue", "hall_name": "venue", "hall_capacity": "venue",
    "hall_cost": "venue", "hall_options": "venue", "standby_generator": "venue",
    "generator": "venue",
    # schedule
    "event_duration_hours": "schedule", "duration_hours": "schedule",
    "prayer_break_minutes": "schedule", "prayer_minutes": "schedule",
    # food
    "food_boxes": "food", "meal_boxes": "food", "boxes_ordered": "food",
    "food_quantity": "food", "meal_price": "food", "basic_meal_price": "food",
    "standard_meal_price": "food", "meal_type": "food",
    "food_delivery_cost": "food", "delivery_cost": "food",
    # judging
    "judge_count": "judging", "confirmed_judges": "judging",
    "judge_honorarium": "judging", "minutes_per_team_per_judge": "judging",
    "max_teams_per_judge": "judging", "judge_delay_minutes": "judging",
    # budget
    "total_budget": "budget", "budget": "budget", "approved_budget": "budget",
    "committed_costs": "budget", "prize_cost": "budget", "prizes": "budget",
    "printing_cost": "budget", "banner_cost": "budget", "t_shirt_cost": "budget",
    "certificate_cost": "budget", "photography_cost": "budget",
    "sound_cost": "budget", "equipment_cost": "budget", "misc_cost": "budget",
    "dj_cost": "budget", "photographer_cost": "budget",
}


def _keys_to_areas(keys: list[str], claims: dict) -> set[str]:
    """Map accepted claim keys to planning areas using the claims dict."""
    areas: set[str] = set()
    for k in keys:
        # Prefer the area recorded in the claim itself.
        if k in claims:
            areas.add(str(claims[k].get("area", "budget")))
        elif k in _KEY_AREA:
            areas.add(_KEY_AREA[k])
    return areas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_update(
    state: dict,
    parsed_doc: dict,
    raw_message: str = "",
) -> tuple[dict, Plan | None, list[str]]:
    """
    Merge a parsed LLM document into state and return an updated plan.

    Args:
        state:       Current SharedState.  Not mutated — a deep copy is used.
        parsed_doc:  Normalised document from core/parser.parse_llm_response.
                     Must have a "claims" list of normalised claim dicts.
        raw_message: The original message text (for rejection records).

    Returns:
        (new_state, plan, rejection_reasons)
        new_state:         Updated SharedState.
        plan:              New Plan (or None if no changes were applied).
        rejection_reasons: List of human-readable strings for rejected claims
                           from this update only.
    """
    new_state = deepcopy(state)
    new_state.setdefault("claims", {})
    new_state.setdefault("rejected", [])
    new_state.setdefault("messages", [])

    if raw_message:
        new_state["messages"].append(raw_message)

    # Enrich each claim with sender/date from message_meta.
    meta = parsed_doc.get("message_meta", {}) or {}
    sender = str(meta.get("sender") or "Unknown")
    date   = str(meta.get("date")   or "1 Jan 2026")

    claims_input: list[dict] = []
    for c in (parsed_doc.get("claims") or []):
        enriched = dict(c)
        enriched.setdefault("sender", sender)
        enriched.setdefault("date", date)
        claims_input.append(enriched)

    # Count rejections before merge so we can diff afterwards.
    rejected_before = len(new_state["rejected"])

    accepted_keys, _rj = merge_all_claims(new_state, claims_input, raw_message)

    # Collect the new rejection reasons (those added in this update).
    new_rejections: list[str] = [
        item["reason"]
        for item in new_state["rejected"][rejected_before:]
    ]

    # Decide which areas changed.
    if not accepted_keys:
        # Nothing accepted → preserve existing plan, just report rejections.
        return new_state, None, new_rejections

    changed_areas = _keys_to_areas(accepted_keys, new_state["claims"])

    # Always regenerate the full plan for correctness; the area information is
    # surfaced for diagnostics and future optimisation.
    plan = generate_plan(new_state)
    plan["_recalculated_areas"] = sorted(changed_areas)   # type: ignore[assignment]

    return new_state, plan, new_rejections
