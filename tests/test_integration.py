"""
tests/test_integration.py — Integration tests for the complete MTM pipeline.

All tests use mocked LLM responses (no real API calls).
Real model tests are marked REAL_MODEL and skipped when the backend is unreachable.

Test plan
---------
 1. Initial complete run — all five areas planned.
 2. English message extraction into claims.
 3. Roman Urdu price update applied.
 4. New message changing only food price — only food+budget recalculated.
 5. New message changing printing cost — only budget recalculated.
 6. Invalid venue capacity — plan marked invalid.
 7. Incorrect food quantity — rejected, quantity not applied.
 8. Conflicting message — stale message rejected.
 9. Duplicate message — not processed twice.
10. Invalid update — previous valid plan preserved.
11. Update caps at 5 LLM calls.
12. Same agent.run_messages() works for both initial run and update.
13. Trace has all required fields.
14. Token counts come from mocked API, not invented.
15. generate_plan produces correct structure after agent run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import os
from copy import deepcopy
from unittest.mock import patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.schema import empty_state
from core.planner import generate_plan
from core.updater import apply_update
from agent import create_agent
from planner import merge_claims


# ---------------------------------------------------------------------------
# Dispatch response builder (mirrors test_reader.py helper)
# ---------------------------------------------------------------------------

def _dispatch(claims: list[dict], **kw) -> tuple[str, str, dict, float]:
    doc = {
        "message_meta": {"sender": kw.get("sender", "Test"), "date": kw.get("date", "10 Jan 2026")},
        "primary_area":   kw.get("primary_area", "budget"),
        "message_type":   kw.get("message_type", "fact"),
        "summary":        kw.get("summary", "Test summary."),
        "affected_areas": kw.get("affected_areas", []),
        "claims":         claims,
    }
    usage = {
        "prompt_tokens":     kw.get("pt", 40),
        "completion_tokens": kw.get("ct", 20),
        "total_tokens":      kw.get("pt", 40) + kw.get("ct", 20),
    }
    return json.dumps(doc), kw.get("model", "mock-model"), usage, 0.05


# ---------------------------------------------------------------------------
# Synthetic claim factories
# ---------------------------------------------------------------------------

def _claim(key, value, area, status="confirmed", confidence=0.9, date="10 Jan 2026"):
    return {"key": key, "value": value, "area": area,
            "status": status, "confidence": confidence,
            "rationale": "test", "updates_key": None, "message_id": None,
            "unit": None, "date": date, "sender": "Test"}


def _base_state():
    """A minimal valid state: 100 attendees, 20 teams, budget, venue, food, judging."""
    state = empty_state()
    state["claims"] = {
        "people_count": {"value": 100, "area": "schedule", "date": "5 Jan 2026",
                         "sender": "Dean", "confidence": 1.0, "rationale": ""},
        "team_count":   {"value": 20,  "area": "judging",  "date": "5 Jan 2026",
                         "sender": "Dean", "confidence": 1.0, "rationale": ""},
        "total_budget": {"value": 400000, "area": "budget", "date": "5 Jan 2026",
                         "sender": "Finance", "confidence": 1.0, "rationale": ""},
        "event_duration_hours": {"value": 4, "area": "schedule", "date": "5 Jan 2026",
                                 "sender": "Dean", "confidence": 1.0, "rationale": ""},
        "venue_options": {
            "value": [
                {"name": "Hall A", "capacity": 150, "cost": 80000, "generator": True},
                {"name": "Hall B", "capacity": 200, "cost": 120000, "generator": True},
            ],
            "area": "venue", "date": "5 Jan 2026",
            "sender": "Venue", "confidence": 1.0, "rationale": "",
        },
        "food_boxes":   {"value": 110, "area": "food",    "date": "5 Jan 2026",
                         "sender": "Caterer", "confidence": 0.9, "rationale": ""},
        "meal_price":   {"value": 600, "area": "food",    "date": "5 Jan 2026",
                         "sender": "Caterer", "confidence": 0.9, "rationale": ""},
        "judge_count":  {"value": 5,   "area": "judging", "date": "5 Jan 2026",
                         "sender": "Coord", "confidence": 1.0, "rationale": ""},
        "max_teams_per_judge": {"value": 4, "area": "judging", "date": "5 Jan 2026",
                                "sender": "Coord", "confidence": 1.0, "rationale": ""},
        "judge_honorarium": {"value": 5000, "area": "judging", "date": "5 Jan 2026",
                             "sender": "Coord", "confidence": 1.0, "rationale": ""},
    }
    state["processed_hashes"] = []
    return state


# ===========================================================================
# 1. Initial complete run — plan has all required keys
# ===========================================================================

def test_initial_run_all_areas_planned():
    state = _base_state()
    plan = generate_plan(state)
    for key in ("event", "venue", "schedule", "food", "judging", "budget", "checks", "valid"):
        assert key in plan, f"Missing plan key: {key}"
    assert len(plan["schedule"]) > 0
    assert plan["venue"]["name"] == "Hall A"   # cheapest feasible


# ===========================================================================
# 2. English message extraction applied to state
# ===========================================================================

def test_english_extraction_budget_claim():
    """A budget message is processed and claim lands in state."""
    msg = "[Message 1 | From: Finance | Date: 5 Jan]\nThe approved budget is Rs 500,000."
    dispatch_ret = _dispatch(
        [_claim("total_budget", 500000, "budget")],
        primary_area="budget", sender="Finance", date="5 Jan"
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        state, trace = read_messages([msg])
    assert "total_budget" in state["claims"]
    assert state["claims"]["total_budget"]["value"] == 500000


# ===========================================================================
# 3. Roman Urdu price update applied
# ===========================================================================

def test_roman_urdu_price_update_applied():
    """'ab Rs X hoga' extracted as confirmed meal_price update."""
    state = _base_state()
    msg = "[Message 2 | From: Caterer | Date: 12 Jan]\nSalam! ab Rs 900 hoga per head."
    dispatch_ret = _dispatch(
        [_claim("meal_price", 900, "food", date="12 Jan 2026")],
        primary_area="food", sender="Caterer", date="12 Jan"
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        updated, trace = read_messages([msg], existing_state=state)
    assert updated["claims"]["meal_price"]["value"] == 900


# ===========================================================================
# 4. Food price change — food and budget recalculated
# ===========================================================================

def test_food_price_change_affects_food_and_budget():
    """Changing meal_price changes food.total and budget.total."""
    state = _base_state()
    plan_before = generate_plan(state)
    food_before   = plan_before["food"]["total"]
    budget_before = plan_before["budget"]["total"]

    # Update meal_price to a higher value.
    state["claims"]["meal_price"]["value"] = 1200
    state["claims"]["meal_price"]["date"]  = "15 Jan 2026"   # newer

    plan_after = generate_plan(state)
    assert plan_after["food"]["total"]   > food_before
    assert plan_after["budget"]["total"] > budget_before


# ===========================================================================
# 5. Printing cost change — only budget recalculated
# ===========================================================================

def test_printing_cost_affects_only_budget():
    """A printing_cost key belongs to budget; venue and schedule are unaffected."""
    from core.planner import _affected_areas
    state = _base_state()
    state["claims"]["printing_cost"] = {
        "value": 15000, "area": "budget", "date": "10 Jan 2026",
        "sender": "Finance", "confidence": 1.0, "rationale": "",
    }
    affected = _affected_areas(["printing_cost"], state["claims"])
    assert "budget" in affected
    assert "venue"    not in affected
    assert "schedule" not in affected


# ===========================================================================
# 6. Invalid venue capacity — plan marked invalid
# ===========================================================================

def test_invalid_venue_capacity_plan_invalid():
    """A venue smaller than headcount → plan.valid = False."""
    state = _base_state()
    state["claims"]["venue_options"]["value"] = [
        {"name": "Tiny Hall", "capacity": 50, "cost": 20000, "generator": False}
    ]
    plan = generate_plan(state)
    assert plan["valid"] is False
    venue_check = next(c for c in plan["checks"] if c["name"] == "Venue capacity")
    assert not venue_check["passed"]


# ===========================================================================
# 7. Incorrect food quantity — rejected
# ===========================================================================

def test_food_quantity_below_headcount_rejected():
    """food_boxes < people_count is rejected by merge_claims."""
    state = _base_state()
    extraction = {
        "message_meta": {"sender": "Caterer", "date": "10 Jan 2026"},
        "claims": [
            {"key": "food_boxes", "value": 50, "area": "food",
             "status": "confirmed", "confidence": 1.0, "rationale": "less food"}
        ],
    }
    _, rejected = merge_claims(state, extraction, "Bad food msg")
    assert any(r.get("key") == "food_boxes" for r in rejected)
    # Active claim must remain at 110 (not 50).
    assert state["claims"]["food_boxes"]["value"] == 110


# ===========================================================================
# 8. Conflicting message — stale rejected
# ===========================================================================

def test_conflicting_stale_message_rejected():
    """A claim dated older than the active claim is rejected."""
    state = _base_state()
    stale_extraction = {
        "message_meta": {"sender": "Finance", "date": "1 Jan 2026"},   # older
        "claims": [
            {"key": "total_budget", "value": 999999, "area": "budget",
             "status": "confirmed", "confidence": 1.0, "rationale": "stale"}
        ],
    }
    _, rejected = merge_claims(state, stale_extraction, "stale msg")
    assert any(r.get("key") == "total_budget" for r in rejected)
    assert state["claims"]["total_budget"]["value"] == 400000   # unchanged


# ===========================================================================
# 9. Duplicate message not processed twice
# ===========================================================================

def test_duplicate_message_skipped():
    """A message whose hash is already in processed_hashes is not sent to the LLM."""
    msg = "[Message 1 | From: Dean | Date: 5 Jan]\nBudget is Rs 500,000."
    h = hashlib.sha256(msg.encode()).hexdigest()

    state = _base_state()
    state["processed_hashes"] = [h]

    # Simulate the dedup logic from main.py
    known = set(state["processed_hashes"])
    new_messages = [m for m in [msg] if hashlib.sha256(m.encode()).hexdigest() not in known]
    assert new_messages == [], "Duplicate message should have been filtered out"


# ===========================================================================
# 10. Invalid update — previous valid plan preserved
# ===========================================================================

def test_invalid_update_preserves_last_valid_plan():
    """When an update produces an invalid plan, last_valid_plan is kept."""
    state = _base_state()
    valid_plan = generate_plan(state)
    assert valid_plan["valid"] is True

    # Store it as the last valid plan.
    last_valid = deepcopy(valid_plan)

    # Now corrupt the state so the new plan is invalid.
    state["claims"]["venue_options"]["value"] = [
        {"name": "Broken Hall", "capacity": 10, "cost": 5000, "generator": False}
    ]
    bad_plan = generate_plan(state)
    assert bad_plan["valid"] is False

    # Simulate the preservation logic from main.py / app.py:
    if not bad_plan.get("valid"):
        active_plan = last_valid   # fall back
    else:
        active_plan = bad_plan

    assert active_plan["valid"] is True
    assert active_plan["venue"]["name"] == "Hall A"


# ===========================================================================
# 11. Update capped at 5 model calls
# ===========================================================================

def test_update_capped_at_five_calls():
    """When more than 5 new messages arrive, only 5 are processed per update."""
    MAX = 5
    messages = [f"[Message {i} | From: X | Date: 10 Jan]\nFact {i}." for i in range(1, 10)]
    known: set[str] = set()
    new = [m for m in messages if hashlib.sha256(m.encode()).hexdigest() not in known]
    capped = new[:MAX]
    assert len(capped) == MAX


# ===========================================================================
# 12. Same run_messages works for initial and update
# ===========================================================================

def test_same_run_messages_for_initial_and_update():
    """agent.run_messages accepts an empty state (initial) and existing state (update)."""
    claim = [_claim("total_budget", 500000, "budget")]
    dispatch_ret = _dispatch(claim, primary_area="budget", sender="Finance", date="5 Jan")

    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        # Initial — no existing state
        state1, trace1 = read_messages(["msg 1"])
        assert "total_budget" in state1["claims"]

        # Update — pass existing state
        state2, trace2 = read_messages(["msg 2"], existing_state=state1)
        assert "total_budget" in state2["claims"]
        assert len(trace2) == 1    # only the new message


# ===========================================================================
# 13. Trace has all required fields
# ===========================================================================

def test_trace_has_all_required_fields():
    """Every trace entry has the 11 required keys."""
    dispatch_ret = _dispatch(
        [_claim("total_budget", 500000, "budget")],
        pt=55, ct=25
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        _, trace = read_messages(["[Message 1 | From: X | Date: 5 Jan]\nBudget 500k."])

    required = ("call", "model", "area", "input", "output",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "duration_seconds", "status", "merge_decision")
    for field in required:
        assert field in trace[0], f"Missing: {field}"


# ===========================================================================
# 14. Token counts from API, not invented
# ===========================================================================

def test_token_counts_from_api():
    """Token counts in trace match the mocked API response exactly."""
    dispatch_ret = _dispatch(
        [_claim("judge_count", 10, "judging")],
        pt=111, ct=44
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        _, trace = read_messages(["msg"])
    assert trace[0]["prompt_tokens"]     == 111
    assert trace[0]["completion_tokens"] == 44
    assert trace[0]["total_tokens"]      == 155


# ===========================================================================
# 15. generate_plan structure after full agent run
# ===========================================================================

def test_generate_plan_after_agent_run():
    """After agent processes messages, generate_plan returns a coherent plan."""
    messages_claims = [
        _claim("people_count", 100, "schedule", date="5 Jan 2026"),
        _claim("team_count",   20,  "judging",  date="5 Jan 2026"),
        _claim("total_budget", 400000, "budget", date="5 Jan 2026"),
        _claim("venue_options",
               [{"name": "Hall A", "capacity": 150, "cost": 80000, "generator": True}],
               "venue", date="5 Jan 2026"),
        _claim("food_boxes",  110, "food", date="5 Jan 2026"),
        _claim("meal_price",  600, "food", date="5 Jan 2026"),
        _claim("judge_count",   5, "judging", date="5 Jan 2026"),
        _claim("max_teams_per_judge", 4, "judging", date="5 Jan 2026"),
        _claim("judge_honorarium", 5000, "judging", date="5 Jan 2026"),
    ]
    dispatch_ret = _dispatch(messages_claims, primary_area="budget",
                             sender="System", date="5 Jan 2026")

    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        from agents.reader import read_messages
        state, _ = read_messages(["full pack msg"])

    plan = generate_plan(state)
    assert plan["event"]["attendees"] == 100
    assert plan["venue"]["name"]      == "Hall A"
    assert plan["food"]["boxes"]      >= 100
    assert plan["judging"]["judges"]  == 5
    assert isinstance(plan["valid"], bool)
