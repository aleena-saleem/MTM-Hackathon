"""
tests/test_planning_engine.py — Test suite for the Stage 3 planning engine.

Test categories
---------------
1.  Venue: insufficient capacity flagged.
2.  Venue: feasible venue selected (lowest cost).
3.  Venue: mandatory generator respected.
4.  Food: quantity below headcount rejected.
5.  Food: confirmed order >= headcount accepted.
6.  Food: price source preference (basic < standard < generic).
7.  Judging: insufficient judges flagged.
8.  Judging: judge arriving late reduces effective window.
9.  Judging: required_judges calculation.
10. Schedule: build time fills remaining slot.
11. Schedule: prayer break included.
12. Schedule: schedule exceeding event duration flagged.
13. Budget: total cost calculated correctly.
14. Budget: budget exceeding approved limit flagged.
15. Budget: optional items not silently added to total.
16. Budget: mandatory expenses never removed to fit budget.
17. Conflict: valid correction accepted.
18. Conflict: stale (older-date) message rejected.
19. Conflict: unconfirmed claim rejected.
20. Conflict: food quantity below headcount rejected by conflict module.
21. Updater: only affected areas marked for recalculation.
22. Updater: previous valid plan preserved when nothing accepted.
23. Planner: generate_plan returns correct structure.
24. Planner: valid flag False when any check fails.
25. Validator: all checks pass on a correct plan.
26. Validator: venue capacity check catches shortfall.
27. Validator: budget check catches overspend.
28. Validator: judging time check catches late-judge scenario.
29. Validator: schedule overlap detected.

All tests use synthetic data only. No hardcoded sample-pack facts.
"""

from __future__ import annotations

import sys
import os
import math

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.schema import empty_state
from core.conflict import evaluate_claim, merge_all_claims, DECISION_ACCEPT, DECISION_REJECT
from core.validator import (
    validate_food_quantity,
    validate_venue_capacity,
    validate_budget,
    validate_judge_count,
    validate_judging_time,
    validate_schedule_duration,
    validate_no_overlapping_slots,
    run_all_checks,
)
from core.planner import generate_plan, update_plan, _affected_areas
from core.updater import apply_update
from planners.venue import select_venue
from planners.schedule import build_schedule
from planners.food import plan_food
from planners.judging import plan_judging
from planners.budget import plan_budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(value, area="budget", status="confirmed", confidence=0.9,
                date="10 Jan 2026", sender="Test"):
    return {
        "value": value, "area": area, "status": status,
        "confidence": confidence, "rationale": "test", "date": date, "sender": sender,
    }


def _claims(**kwargs) -> dict:
    """
    Build a claims dict where each kwarg is key=value.
    Area is inferred from key name when possible.
    """
    _area_map = {
        "people_count": "schedule", "attendees": "schedule",
        "team_count": "judging", "event_duration_hours": "schedule",
        "venue_options": "venue", "venue_name": "venue",
        "venue_capacity": "venue", "venue_cost": "venue",
        "standby_generator": "venue",
        "food_boxes": "food", "meal_price": "food", "basic_meal_price": "food",
        "standard_meal_price": "food", "food_delivery_cost": "food",
        "judge_count": "judging", "judge_honorarium": "judging",
        "minutes_per_team_per_judge": "judging", "max_teams_per_judge": "judging",
        "judge_delay_minutes": "judging", "judging_minutes": "judging",
        "total_budget": "budget", "committed_costs": "budget",
        "prayer_break_minutes": "schedule",
    }
    result = {}
    for k, v in kwargs.items():
        area = _area_map.get(k, "budget")
        result[k] = {
            "value": v, "area": area, "date": "10 Jan 2026",
            "sender": "Test", "confidence": 0.9, "rationale": "test",
        }
    return result


def _state_with(**kwargs) -> dict:
    state = empty_state()
    state["claims"] = _claims(**kwargs)
    return state


# ===========================================================================
# 1-3. Venue tests
# ===========================================================================

def test_venue_insufficient_capacity_flagged():
    """When no venue option meets headcount, feasible=False."""
    claims = _claims(
        people_count=300,
        venue_options=[
            {"name": "Small Hall", "capacity": 100, "cost": 50000, "generator": False},
            {"name": "Medium Hall", "capacity": 200, "cost": 80000, "generator": False},
        ],
    )
    result = select_venue(claims)
    assert result["feasible"] is False, "Should be infeasible when no venue fits 300 people"


def test_venue_feasible_lowest_cost_selected():
    """The cheapest feasible venue is selected."""
    claims = _claims(
        people_count=150,
        venue_options=[
            {"name": "Expensive Hall", "capacity": 200, "cost": 150000, "generator": True},
            {"name": "Cheap Hall",     "capacity": 200, "cost": 80000,  "generator": True},
            {"name": "Tiny Hall",      "capacity": 100, "cost": 40000,  "generator": False},
        ],
    )
    result = select_venue(claims)
    assert result["name"] == "Cheap Hall"
    assert result["feasible"] is True


def test_venue_generator_requirement_honoured():
    """A venue without a generator is excluded when standby_generator is required."""
    claims = _claims(
        people_count=100,
        standby_generator=True,
        venue_options=[
            {"name": "No Gen Hall",   "capacity": 150, "cost": 60000, "generator": False},
            {"name": "With Gen Hall", "capacity": 150, "cost": 90000, "generator": True},
        ],
    )
    result = select_venue(claims)
    assert result["name"] == "With Gen Hall"
    assert result["standby_generator"] is True


# ===========================================================================
# 4-6. Food tests
# ===========================================================================

def test_food_quantity_below_headcount_flagged():
    """plan_food reports a violation when extracted boxes < headcount."""
    claims = _claims(people_count=200, food_boxes=100, meal_price=500)
    food, violations = plan_food(claims)
    # Quantity is raised to headcount and violation recorded.
    assert food["boxes"] == 200
    assert any("100" in v and "200" in v for v in violations)


def test_food_confirmed_quantity_accepted():
    """An extracted food quantity >= headcount is accepted as-is."""
    claims = _claims(people_count=150, food_boxes=160, meal_price=600)
    food, violations = plan_food(claims)
    assert food["boxes"] == 160
    assert food["confirmed"] is True
    assert not any("below" in v.lower() for v in violations)


def test_food_price_prefers_basic_over_standard():
    """When both basic and standard prices are present, basic is used."""
    claims = _claims(
        people_count=100,
        basic_meal_price=700,
        standard_meal_price=1000,
    )
    food, _ = plan_food(claims)
    assert food["unit_price"] == 700
    assert food["type"] == "Basic lunch box"


def test_food_uses_standard_when_no_basic():
    """When only standard price is present, it is used."""
    claims = _claims(people_count=100, standard_meal_price=1000)
    food, _ = plan_food(claims)
    assert food["unit_price"] == 1000
    assert food["type"] == "Standard lunch box"


# ===========================================================================
# 7-9. Judging tests
# ===========================================================================

def test_judging_insufficient_judges_flagged():
    """Fewer judges than required should produce a violation."""
    claims = _claims(
        team_count=30,
        judge_count=3,
        max_teams_per_judge=5,   # requires ceil(30/5) = 6 judges
    )
    judging, violations = plan_judging(claims)
    assert judging["required_judges"] == 6
    assert judging["capacity_ok"] is False
    assert len(violations) > 0


def test_judging_late_arrival_reduces_effective_window():
    """A late judge should reduce the effective judging window and flag a violation."""
    claims = _claims(
        team_count=10,
        judge_count=2,
        max_teams_per_judge=5,
        minutes_per_team_per_judge=10.0,
        judge_delay_minutes=30.0,
        judging_minutes=50.0,   # 50 min window - 30 late = 20 min effective
    )
    # Each judge handles 5 teams × 10 min = 50 min; window is 20 → violation.
    judging, violations = plan_judging(claims)
    assert judging["time_ok"] is False
    assert any("delay" in v.lower() or "time" in v.lower() for v in violations)


def test_judging_required_judges_formula():
    """required_judges = ceil(teams / max_teams_per_judge)."""
    for teams, mtpj, expected in [(10, 3, 4), (9, 3, 3), (1, 3, 1), (0, 3, 0)]:
        claims = _claims(team_count=teams, judge_count=99, max_teams_per_judge=mtpj)
        judging, _ = plan_judging(claims)
        assert judging["required_judges"] == expected, \
            f"teams={teams}, mtpj={mtpj}: expected {expected}, got {judging['required_judges']}"


# ===========================================================================
# 10-12. Schedule tests
# ===========================================================================

def test_schedule_has_no_overlapping_slots():
    """build_schedule should never produce overlapping time slots."""
    claims = _claims(
        event_duration_hours=4,
        team_count=20,
        judge_count=4,
        max_teams_per_judge=5,
        minutes_per_team_per_judge=10.0,
    )
    schedule, _ = build_schedule(claims)
    ok, detail = validate_no_overlapping_slots(schedule)
    assert ok, detail


def test_schedule_prayer_break_included():
    """Prayer break appears in the schedule when prayer_break_minutes is set."""
    claims = _claims(
        event_duration_hours=4,
        prayer_break_minutes=15,
        team_count=20,
        judge_count=4,
        max_teams_per_judge=5,
        minutes_per_team_per_judge=10.0,
    )
    schedule, _ = build_schedule(claims)
    activities = [s["activity"].lower() for s in schedule]
    assert any("prayer" in a for a in activities), \
        f"No prayer break in schedule: {activities}"


def test_schedule_exceeds_duration_flagged():
    """A schedule that cannot fit in the event window should report a violation."""
    # Very short event (1 hour = 60 min) with many teams → build time goes negative.
    claims = _claims(
        event_duration_hours=1,
        team_count=50,
        judge_count=1,
        max_teams_per_judge=50,
        minutes_per_team_per_judge=5.0,  # 50 teams × 5 min = 250 min judging alone
    )
    _, violations = build_schedule(claims)
    assert len(violations) > 0, "Expected a violation for schedule overflow"


# ===========================================================================
# 13-16. Budget tests
# ===========================================================================

def test_budget_total_calculated_correctly():
    """budget total = venue + food + judges + committed_other."""
    state = _state_with(
        people_count=100,
        food_boxes=100, meal_price=600,
        judge_count=4, judge_honorarium=5000,
        total_budget=400000,
    )
    plan = generate_plan(state)
    b = plan["budget"]
    expected_food = 100 * 600
    expected_judges = 4 * 5000
    # venue cost is 0 (no venue claim); committed_other = 0.
    assert b["food"]   == pytest.approx(expected_food)
    assert b["judges"] == pytest.approx(expected_judges)


def test_budget_exceeds_limit_flagged():
    """Budget violation is recorded when total > limit."""
    state = _state_with(
        people_count=100,
        food_boxes=100, meal_price=1200,
        judge_count=10, judge_honorarium=10000,
        total_budget=50000,   # too low
    )
    plan = generate_plan(state)
    budget_check = next(c for c in plan["checks"] if c["name"] == "Budget")
    assert not budget_check["passed"]
    assert plan["valid"] is False


def test_budget_optional_items_not_added_to_total():
    """Optional items like dj_cost must not appear in budget total."""
    # Inject an optional claim directly into claims.
    state = _state_with(
        people_count=50,
        food_boxes=50, meal_price=500,
        judge_count=2, judge_honorarium=3000,
        total_budget=200000,
    )
    state["claims"]["dj_cost"] = {
        "value": 50000, "area": "budget", "date": "10 Jan 2026",
        "sender": "Student", "confidence": 0.5, "rationale": "optional",
    }
    _, violations = plan_budget(
        state["claims"],
        venue_cost=0, food_cost=25000, judge_cost=6000,
    )
    # dj_cost is optional — it should not appear in total.
    budget, _ = plan_budget(state["claims"], venue_cost=0, food_cost=25000, judge_cost=6000)
    assert "dj_cost" in budget["optional_items"]
    assert budget["total"] == pytest.approx(31000)   # food + judges only


def test_budget_mandatory_expenses_never_removed():
    """
    If mandatory costs exceed the limit, the plan is flagged as invalid —
    mandatory expenses are never silently dropped to make the budget fit.
    """
    state = _state_with(
        people_count=100,
        food_boxes=100, meal_price=800,     # Rs 80,000
        judge_count=5, judge_honorarium=8000,  # Rs 40,000
        total_budget=50000,                  # impossible limit
    )
    plan = generate_plan(state)
    # Food and judge costs must still be in the plan.
    assert plan["food"]["total"]      == pytest.approx(80000)
    assert plan["judging"]["total_cost"] == pytest.approx(40000)
    assert plan["valid"] is False


# ===========================================================================
# 17-20. Conflict handling tests
# ===========================================================================

def test_conflict_valid_correction_accepted():
    """A newer claim for the same key replaces the older one."""
    existing = {
        "judge_count": {
            "value": 5, "area": "judging", "date": "5 Jan 2026",
            "sender": "A", "confidence": 0.9, "rationale": "",
        }
    }
    incoming = {
        "value": 10, "area": "judging", "status": "confirmed",
        "confidence": 0.95, "date": "12 Jan 2026", "sender": "A",
    }
    decision, reason = evaluate_claim("judge_count", incoming, existing, "msg")
    assert decision == DECISION_ACCEPT, reason


def test_conflict_stale_message_rejected():
    """A claim dated older than the active claim is rejected."""
    existing = {
        "venue_cost": {
            "value": 120000, "area": "venue", "date": "10 Jan 2026",
            "sender": "Venue A", "confidence": 0.9, "rationale": "",
        }
    }
    stale = {
        "value": 80000, "area": "venue", "status": "confirmed",
        "confidence": 0.9, "date": "3 Jan 2026", "sender": "Venue A",
    }
    decision, reason = evaluate_claim("venue_cost", stale, existing, "msg")
    assert decision == DECISION_REJECT
    assert "outdated" in reason.lower()


def test_conflict_unconfirmed_claim_rejected():
    """Claims with status=unconfirmed/suggestion/tentative are rejected."""
    for status in ("unconfirmed", "suggestion", "tentative", "question"):
        incoming = {
            "value": 99, "area": "budget", "status": status,
            "confidence": 0.5, "date": "10 Jan 2026", "sender": "X",
        }
        decision, reason = evaluate_claim("misc_key", incoming, {}, "msg")
        assert decision == DECISION_REJECT, f"Status '{status}' should be rejected"
        assert status in reason.lower()


def test_conflict_food_quantity_below_headcount_rejected():
    """Food quantity below headcount is rejected regardless of date/confidence."""
    existing = {
        "people_count": {
            "value": 200, "area": "schedule", "date": "5 Jan 2026",
            "sender": "Dean", "confidence": 1.0, "rationale": "",
        }
    }
    bad_food = {
        "value": 50, "area": "food", "status": "confirmed",
        "confidence": 1.0, "date": "12 Jan 2026", "sender": "Caterer",
    }
    decision, reason = evaluate_claim("food_boxes", bad_food, existing, "msg")
    assert decision == DECISION_REJECT
    assert "50" in reason and "200" in reason


# ===========================================================================
# 21-22. Updater / incremental update tests
# ===========================================================================

def test_updater_no_accepted_claims_returns_no_plan():
    """When no claims are accepted, apply_update returns plan=None."""
    state = _state_with(total_budget=500000)
    parsed_doc = {
        "message_meta": {"sender": "X", "date": "10 Jan 2026"},
        "claims": [
            {
                "key": "dj_cost", "value": 50000, "area": "budget",
                "status": "suggestion",   # will be rejected
                "confidence": 0.5, "rationale": "optional", "updates_key": None,
            }
        ],
        "summary": "DJ suggestion",
    }
    new_state, plan, rejections = apply_update(state, parsed_doc, "DJ suggestion msg")
    assert plan is None, "No plan should be returned when no claims were accepted"
    assert len(rejections) == 1


def test_updater_previous_state_preserved_on_no_accept():
    """When nothing is accepted, the existing claims survive unchanged."""
    state = _state_with(total_budget=600000, people_count=200)
    parsed_doc = {
        "message_meta": {"sender": "X", "date": "5 Jan 2026"},  # older date
        "claims": [
            {
                "key": "total_budget", "value": 999999, "area": "budget",
                "status": "confirmed", "confidence": 0.5, "rationale": "",
                "updates_key": None,
            }
        ],
        "summary": "Stale budget",
    }
    # Seed the existing budget with a newer date so this one is stale.
    state["claims"]["total_budget"]["date"] = "12 Jan 2026"

    new_state, plan, _ = apply_update(state, parsed_doc, "stale msg")
    # Original value must be preserved.
    assert new_state["claims"]["total_budget"]["value"] == 600000


def test_updater_printing_cost_affects_only_budget():
    """A printing_cost change should mark budget as affected, not venue/schedule."""
    claims = _claims(people_count=100, total_budget=300000)
    claims["printing_cost"] = {
        "value": 5000, "area": "budget", "date": "10 Jan 2026",
        "sender": "Test", "confidence": 0.9, "rationale": "test",
    }
    affected = _affected_areas(["printing_cost"], claims)
    assert "budget" in affected
    assert "venue" not in affected
    assert "schedule" not in affected


# ===========================================================================
# 23-24. Planner integration tests
# ===========================================================================

def test_generate_plan_returns_required_structure():
    """generate_plan must return all required top-level keys."""
    state = _state_with(
        people_count=100,
        team_count=20,
        food_boxes=100, meal_price=600,
        judge_count=4, judge_honorarium=5000,
        max_teams_per_judge=5,
        total_budget=400000,
        venue_options=[{"name": "Hall A", "capacity": 150, "cost": 80000, "generator": True}],
    )
    plan = generate_plan(state)
    for key in ("event", "venue", "schedule", "food", "judging", "budget", "checks", "valid"):
        assert key in plan, f"Missing key: {key}"
    assert isinstance(plan["checks"], list)
    assert isinstance(plan["valid"], bool)
    assert isinstance(plan["schedule"], list)


def test_generate_plan_valid_false_on_check_failure():
    """valid=False when any check fails."""
    # Venue capacity below headcount → will fail.
    state = _state_with(
        people_count=300,
        venue_options=[{"name": "Tiny", "capacity": 50, "cost": 10000, "generator": False}],
        food_boxes=300, meal_price=500,
        judge_count=5, judge_honorarium=3000,
        max_teams_per_judge=5,
        total_budget=500000,
    )
    plan = generate_plan(state)
    assert plan["valid"] is False


# ===========================================================================
# 25-29. Validator unit tests
# ===========================================================================

def test_validator_all_pass_on_valid_plan():
    """run_all_checks returns all passed=True for a fully valid plan.

    Data is designed so every hard constraint is satisfied:
    - 20 teams, 4 judges, max 5 teams/judge → 4 judges handle 5 teams each
    - 5 teams × 10 min/team = 50 min needed; judging slot = 60 min → fits
    - Venue capacity 150 >= 100 attendees
    - Food 110 boxes >= 100 attendees
    - Budget total Rs 166,000 <= limit Rs 400,000
    - Schedule slots total 240 min (4 hours) with no overlaps
    """
    plan = {
        "event":    {"attendees": 100, "teams": 20, "duration_hours": 4, "staff": 5},
        "venue":    {"name": "Hall A", "capacity": 150, "cost": 80000, "standby_generator": True, "feasible": True},
        "food":     {"type": "Lunch", "boxes": 110, "unit_price": 600, "delivery": 0, "total": 66000},
        "judging":  {"judges": 4, "required_judges": 4, "honorarium_each": 5000,
                     "total_cost": 20000, "max_teams_per_judge": 5,
                     "minutes_per_team": 10, "late_arrival_buffer_minutes": 0},
        "budget":   {"limit": 400000, "venue": 80000, "food": 66000,
                     "judges": 20000, "committed_other": 0,
                     "total": 166000, "remaining": 234000},
        # Slots: 30 + 110 + 60 + 40 = 240 min = 4 hours; judging slot = 60 min
        "schedule": [
            {"time": "00:00–00:30", "activity": "Check-in"},
            {"time": "00:30–02:20", "activity": "Build session"},
            {"time": "02:20–03:20", "activity": "Judging (60 min)"},
            {"time": "03:20–04:00", "activity": "Results"},
        ],
        "checks": [],
        "valid": True,
    }
    checks = run_all_checks(plan)
    failed = [c for c in checks if not c["passed"]]
    assert not failed, f"Unexpected failures: {failed}"


def test_validator_venue_capacity_shortfall():
    """validate_venue_capacity returns False when capacity < headcount."""
    ok, detail = validate_venue_capacity(50, 200)
    assert not ok
    assert "50" in detail and "200" in detail


def test_validator_budget_overspend():
    """validate_budget returns False when total > limit."""
    ok, detail = validate_budget(600000, 500000)
    assert not ok
    assert "600" in detail.replace(",", "") or "over" in detail.lower()


def test_validator_judging_time_late_judge():
    """validate_judging_time catches a scenario where delay makes it impossible."""
    # 10 teams, 1 judge, 10 min/team = 100 min needed; window 60 - 20 late = 40 min.
    ok, detail = validate_judging_time(
        judges=1, available_judges=1,
        teams=10, minutes_per_team=10,
        available_judging_minutes=60,
        late_minutes=20,
    )
    assert not ok
    assert "100" in detail or "needed" in detail.lower()


def test_validator_schedule_overlap_detected():
    """validate_no_overlapping_slots catches overlapping entries."""
    schedule = [
        {"time": "00:00–01:00", "activity": "A"},
        {"time": "00:45–01:30", "activity": "B"},   # overlaps with A
    ]
    ok, detail = validate_no_overlapping_slots(schedule)
    assert not ok
    assert "overlap" in detail.lower() or "ends at" in detail.lower()
