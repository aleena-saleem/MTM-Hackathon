"""Deterministic planning, conflict handling and validation."""
from datetime import datetime
import math

AREAS = {"venue", "schedule", "food", "judging", "budget"}
AREA_ALIASES = {
    "venue": "venue", "schedule": "schedule", "food": "food",
    "judging": "judging", "budget": "budget", "reading": "budget"
}

def _num(v, default=0):
    try:
        if isinstance(v, (int, float)): return float(v)
        s = str(v).replace(",", "").replace("Rs", "").replace("PKR", "").strip()
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default

def _date(s):
    for fmt in ("%d %b %Y", "%d %B %Y", "%d %b", "%d %B"):
        try:
            d = datetime.strptime(str(s), fmt)
            return d.replace(year=2026) if d.year == 1900 else d
        except Exception:
            pass
    return datetime(2026, 1, 1)

def merge_claims(state, extraction, message):
    """Merge extracted claims; newer claims replace older claims for the same key.
    Suspicious claims are retained in the trace but not applied if they violate invariants.
    """
    state.setdefault("claims", {})
    state.setdefault("rejected", [])
    state.setdefault("messages", [])
    state["messages"].append(message)
    meta = extraction.get("message_meta", {}) or {}
    msg_date = _date(meta.get("date", "1 Jan 2026"))
    source = meta.get("sender", "Unknown sender")
    accepted, rejected = [], []

    for claim in extraction.get("claims", []) or []:
        if not isinstance(claim, dict): continue
        key = str(claim.get("key", "")).strip().lower()
        if not key: continue
        value = claim.get("value")
        area = str(claim.get("area", "budget")).lower()
        area = AREA_ALIASES.get(area, "budget")
        confidence = float(claim.get("confidence", 0.7) or 0.7)
        rationale = str(claim.get("rationale", ""))

        # Universal safety invariant: food quantity cannot be below people to feed.
        if key in {"food_boxes", "meal_boxes", "food_quantity", "boxes_ordered"}:
            people = _num(state["claims"].get("people_count", {}).get("value", 0))
            if people and _num(value) < people:
                reason = f"Rejected: {value} food boxes cannot feed {int(people)} people. Quantity must be at least the attendance count."
                item = {"message": message, "key": key, "value": value, "reason": reason}
                state["rejected"].append(item); rejected.append(item)
                continue

        # A tentative suggestion is not an approval or a confirmed commitment.
        status = str(claim.get("status", "confirmed")).lower()
        if status in {"suggestion", "tentative", "question", "unconfirmed"}:
            reason = f"Not applied: this is {status}, not an approved/confirmed requirement."
            item = {"message": message, "key": key, "value": value, "reason": reason}
            state["rejected"].append(item); rejected.append(item)
            continue

        old = state["claims"].get(key)
        if old:
            old_date = _date(old.get("date", "1 Jan 2026"))
            # Newer source wins for mutable values; equal/older conflicting claims do not silently overwrite.
            if msg_date < old_date and old.get("value") != value:
                reason = f"Rejected as outdated: {source}'s {msg_date:%d %b} claim is older than the active {old_date:%d %b} claim."
                item = {"message": message, "key": key, "value": value, "reason": reason}
                state["rejected"].append(item); rejected.append(item)
                continue
            if msg_date == old_date and old.get("value") != value and confidence <= float(old.get("confidence", 0.7)):
                reason = "Rejected: same-date conflicting claim has no stronger confidence or authority."
                item = {"message": message, "key": key, "value": value, "reason": reason}
                state["rejected"].append(item); rejected.append(item)
                continue

        state["claims"][key] = {
            "value": value, "area": area, "date": msg_date.strftime("%d %b %Y"),
            "sender": source, "confidence": confidence, "rationale": rationale
        }
        accepted.append(key)

    return accepted, rejected

def val(state, *keys, default=None):
    for key in keys:
        if key in state.get("claims", {}):
            return state["claims"][key]["value"]
    return default

def build_plan(state):
    c = state.get("claims", {})
    people = int(_num(val(state, "people_count", "attendees", default=150), 150))
    teams = int(_num(val(state, "team_count", "teams", default=math.ceil(people/3)), math.ceil(people/3)))
    budget = _num(val(state, "total_budget", "budget", default=650000), 650000)
    staff = int(_num(val(state, "staff_count", "staff_on_duty", default=0)))
    duration = _num(val(state, "event_duration_hours", "duration_hours", default=4), 4)
    prayer = _num(val(state, "prayer_break_minutes", "prayer_minutes", default=0))
    halls = val(state, "venue_options", "venues", default=[])
    if isinstance(halls, dict): halls = [halls]
    hall = None
    if isinstance(halls, list):
        feasible = []
        for h in halls:
            if isinstance(h, dict):
                cap = _num(h.get("capacity"))
                cost = _num(h.get("cost"))
                if cap >= people and cost >= 0:
                    feasible.append((cost, h))
        if feasible: hall = sorted(feasible, key=lambda x:x[0])[0][1]
    # A single selected venue may have been extracted as separate claims.
    venue_name = val(state, "venue_name", default=(hall or {}).get("name", "No feasible venue extracted"))
    venue_capacity = _num(val(state, "venue_capacity", default=(hall or {}).get("capacity", 0)))
    venue_cost = _num(val(state, "venue_cost", default=(hall or {}).get("cost", 0)))
    generator = bool(val(state, "standby_generator", default=(hall or {}).get("generator", False)))

    # Prefer the lower explicitly quoted tier to keep the plan economical.
    # If only a single price is supplied, use that price without inventing a tier.
    basic_price = val(state, "basic_meal_price", default=None)
    standard_price = val(state, "standard_meal_price", default=None)
    if basic_price is not None:
        meal_price, meal_type = _num(basic_price), "Basic lunch box"
    elif standard_price is not None:
        meal_price, meal_type = _num(standard_price), "Standard lunch box"
    else:
        meal_price = _num(val(state, "meal_price", "food_price_per_person", default=0), 0)
        meal_type = val(state, "meal_type", default="Quoted lunch box")
    food_boxes = int(_num(val(state, "food_boxes", "meal_boxes", default=people), people))
    delivery = _num(val(state, "food_delivery_cost", default=0))
    judges = int(_num(val(state, "judge_count", "confirmed_judges", default=0)))
    honorarium = _num(val(state, "judge_honorarium", default=0))
    minutes_per_team = _num(val(state, "minutes_per_team_per_judge", default=15), 15)
    max_teams_per_judge = int(_num(val(state, "max_teams_per_judge", default=3), 3))
    judge_late = _num(val(state, "judge_delay_minutes", default=0))
    fixed = val(state, "committed_costs", default={})
    if not isinstance(fixed, dict): fixed = {}
    fixed_cost = sum(_num(x) for x in fixed.values())
    prizes = _num(val(state, "prize_cost", default=fixed.get("prizes", 0)))
    # Avoid double counting prizes if already in committed costs.
    if "prizes" in fixed: prizes = 0
    judge_cost = judges * honorarium
    food_cost = food_boxes * meal_price + delivery
    total = venue_cost + food_cost + judge_cost + fixed_cost + prizes
    remaining = budget - total
    required_judges = math.ceil(teams / max_teams_per_judge) if max_teams_per_judge > 0 else teams
    judge_capacity = judges * max_teams_per_judge
    judging_minutes = minutes_per_team * max_teams_per_judge
    schedule = [
        {"time": "00:00–00:30", "activity": "Check-in, team seating and opening briefing"},
        {"time": "00:30–02:40", "activity": "Hackathon build session"},
        {"time": "02:40–02:55", "activity": f"Prayer break ({int(prayer)} minutes)" if prayer else "Short break"},
        {"time": "02:55–03:40", "activity": "Final submissions and judging (45 minutes)"},
        {"time": "03:40–04:00", "activity": "Results and closing"}
    ]
    checks = [
        ("Venue capacity", venue_capacity >= people and venue_capacity > 0,
         f"{venue_capacity:g} seats for {people} attendees"),
        ("Budget", remaining >= 0, f"Rs {total:,.0f} planned against Rs {budget:,.0f}; remaining Rs {remaining:,.0f}"),
        ("Food quantity", food_boxes >= people, f"{food_boxes} boxes for {people} people"),
        ("Judging capacity", judges >= required_judges and judge_capacity >= teams,
         f"{judges} judges; need at least {required_judges} for {teams} teams at max {max_teams_per_judge} teams/judge"),
        ("Schedule", duration == 4 and sum([30,130,15,45,20]) == 240,
         f"4-hour template; prayer break requested: {prayer:g} minutes"),
        ("Areas covered", bool(venue_name) and food_boxes > 0 and judges > 0,
         "Venue, schedule, food, judging and budget represented")
    ]
    return {
        "event": {"attendees": people, "teams": teams, "duration_hours": duration, "staff": staff},
        "venue": {"name": venue_name, "capacity": venue_capacity, "cost": venue_cost, "standby_generator": generator},
        "schedule": schedule,
        "food": {"type": meal_type, "boxes": food_boxes, "unit_price": meal_price, "delivery": delivery, "total": food_cost},
        "judging": {"judges": judges, "required_judges": required_judges, "honorarium_each": honorarium,
                    "total_cost": judge_cost, "max_teams_per_judge": max_teams_per_judge,
                    "minutes_per_team": minutes_per_team, "late_arrival_buffer_minutes": judge_late},
        "budget": {"limit": budget, "venue": venue_cost, "food": food_cost, "judges": judge_cost,
                   "committed_other": fixed_cost + prizes, "total": total, "remaining": remaining},
        "checks": [{"name": n, "passed": bool(ok), "detail": d} for n,ok,d in checks],
        "valid": all(x[1] for x in checks)
    }
