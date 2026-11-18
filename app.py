"""
app.py — Mind to Machine: Professional Event Planning Dashboard.

Sections
--------
  Sidebar      — project info, LLM status, run controls, downloads
  Tab 1        — Dashboard overview (live metrics)
  Tab 2        — Event Plan  (Venue · Schedule · Food · Judging · Budget)
  Tab 3        — Message Updates (process new messages, show recalc'd areas)
  Tab 4        — Rejected Messages (reasons, constraints, plan preservation)
  Tab 5        — Execution Trace (every real API call, real tokens, timing)

All displayed figures come directly from the backend pipeline.
Nothing is hardcoded or fabricated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy

# Load .env FIRST — before any project module reads os.getenv().
# override=True ensures .env values beat any stale system env vars
# (e.g. a placeholder OPENAI_API_KEY set in Windows environment).
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st

import config
from agent import create_agent
from core.planner import generate_plan
from core.schema import empty_state

# ─────────────────────────────────────────────────────────────
# Constants & paths
# ─────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
STATE_FILE   = os.path.join(ROOT, "output", "state.json")
MESSAGES_FILE = os.path.join(ROOT, "data", "messages.txt")
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

_AREAS_ALL = ["venue", "schedule", "food", "judging", "budget"]

# ─────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mind to Machine | Event Planner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Minimal CSS — professional palette, no animations
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card-style containers */
div[data-testid="stMetric"] {
    background: #f8f9fb;
    border: 1px solid #e2e6ea;
    border-radius: 8px;
    padding: 12px 16px;
}
/* PASS / FAIL row colours handled inline via st.success / st.error */
/* Tighten expander headers */
details > summary { font-size: 0.92rem; }
/* Subtle divider */
hr { border-top: 1px solid #e2e6ea; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────

def _hash(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()


def _fresh_state() -> dict:
    s = empty_state()
    s.update({
        "processed_hashes": [],
        "trace": [],
        "plan": None,
        "last_valid_plan": None,
        "run_count": 0,
        "total_tokens": 0,
        "total_duration": 0.0,
        "last_run_duration": 0.0,
        "last_recalculated_areas": [],
    })
    return s


def _load_disk_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
            # Backfill any keys added in later stages
            for k, v in _fresh_state().items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return _fresh_state()


def _save_disk_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False, default=str)


# Bootstrap session state once per browser session
if "app_state" not in st.session_state:
    st.session_state.app_state = _load_disk_state()

S: dict = st.session_state.app_state   # short alias


# ─────────────────────────────────────────────────────────────
# Message splitting
# ─────────────────────────────────────────────────────────────

def _split_messages(text: str) -> list[str]:
    parts = [x.strip() for x in re.split(r"(?=\[Message\s+\d+)", text) if x.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


# ─────────────────────────────────────────────────────────────
# Pipeline runner  (shared by full-pack and update paths)
# ─────────────────────────────────────────────────────────────

def _run_pipeline(messages: list[str]) -> tuple[bool, str]:
    """
    Run the agent on `messages`, update S in-place, save to disk.
    Returns (success: bool, message: str).
    """
    known = set(S.get("processed_hashes", []))
    new_msgs = [m for m in messages if _hash(m) not in known]
    if not new_msgs:
        return False, "All submitted messages have already been processed."

    MAX = 5
    capped = len(new_msgs) > MAX
    if capped:
        new_msgs = new_msgs[:MAX]

    t0 = time.time()
    agent_input = {
        "claims":   S["claims"],
        "rejected": S["rejected"],
        "messages": S["messages"],
    }
    agent = create_agent()
    call_offset = len(S["trace"])

    result_state, _ = agent.run_messages(new_msgs, agent_input)

    # Renumber trace entries so they continue from previous runs
    for i, e in enumerate(agent.trace):
        e["call"] = call_offset + i + 1

    # Merge back into persistent state
    S["claims"]   = result_state["claims"]
    S["rejected"] = result_state["rejected"]
    S["messages"] = result_state["messages"]
    for m in new_msgs:
        h = _hash(m)
        if h not in known:
            S["processed_hashes"].append(h)
            known.add(h)
    S["trace"].extend(agent.trace)

    # Generate plan
    new_plan = generate_plan(S)
    if new_plan.get("valid"):
        S["last_valid_plan"] = deepcopy(new_plan)
    elif S.get("last_valid_plan") is None:
        S["last_valid_plan"] = deepcopy(new_plan)
    S["plan"] = new_plan

    # Identify recalculated areas (areas whose claims changed)
    accepted_keys: list[str] = []
    for e in agent.trace:
        md = e.get("merge_decision", {})
        accepted_keys.extend(md.get("accepted_keys", []))
    S["last_recalculated_areas"] = _keys_to_areas(accepted_keys, S["claims"])

    dur = round(time.time() - t0, 2)
    tokens = sum(e.get("total_tokens", 0) for e in agent.trace if e.get("status") == "success")
    S["total_tokens"]      = S.get("total_tokens", 0) + tokens
    S["total_duration"]    = round(S.get("total_duration", 0.0) + dur, 2)
    S["last_run_duration"] = dur
    S["run_count"]         = S.get("run_count", 0) + 1
    _save_disk_state(S)

    cap_note = f" (capped at {MAX}; submit again for the rest)" if capped else ""
    return True, (
        f"Processed {len(new_msgs)} message(s){cap_note} · "
        f"{len(agent.trace)} LLM call(s) · {tokens:,} tokens · {dur}s"
    )


_KEY_AREA_MAP: dict[str, str] = {
    "venue_name": "venue", "venue_capacity": "venue", "venue_cost": "venue",
    "venue_options": "venue", "standby_generator": "venue",
    "event_duration_hours": "schedule", "prayer_break_minutes": "schedule",
    "food_boxes": "food", "meal_price": "food", "basic_meal_price": "food",
    "standard_meal_price": "food", "food_delivery_cost": "food",
    "judge_count": "judging", "judge_honorarium": "judging",
    "max_teams_per_judge": "judging", "minutes_per_team_per_judge": "judging",
    "judge_delay_minutes": "judging",
    "total_budget": "budget", "committed_costs": "budget", "printing_cost": "budget",
    "prize_cost": "budget", "dj_cost": "budget",
}

def _keys_to_areas(keys: list[str], claims: dict) -> list[str]:
    areas: set[str] = set()
    for k in keys:
        if k in claims:
            areas.add(str(claims[k].get("area", "budget")))
        elif k in _KEY_AREA_MAP:
            areas.add(_KEY_AREA_MAP[k])
    return sorted(areas)


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def _rs(v) -> str:
    try:
        return f"Rs {float(v):,.0f}"
    except Exception:
        return str(v)


def _pill(text: str, colour: str) -> str:
    """Inline HTML badge."""
    bg = {"green": "#d4edda", "red": "#f8d7da", "yellow": "#fff3cd",
          "blue": "#cce5ff", "grey": "#e2e6ea"}.get(colour, "#e2e6ea")
    fg = {"green": "#155724", "red": "#721c24", "yellow": "#856404",
          "blue": "#004085", "grey": "#495057"}.get(colour, "#495057")
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:12px;font-size:0.82rem;font-weight:600;">{text}</span>'
    )


def _validity_banner(plan: dict) -> None:
    if plan.get("valid"):
        st.success("✅  Plan is **VALID** — all hard constraints satisfied.")
    else:
        failed = [c["name"] for c in plan.get("checks", []) if not c.get("passed")]
        st.error(
            "⚠️  Plan has **failed checks** — "
            + ", ".join(failed) if failed else "see Validation tab."
        )


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Mind to Machine")
    st.markdown("*AI-powered hackathon event planner*")
    st.divider()

    # LLM status
    model_info = config.active_model_info()
    st.markdown("**LLM Provider**")
    st.markdown(
        f"Provider: `{model_info['backend'].upper()}`  \n"
        f"Model: `{model_info['model']}`  \n"
        f"Endpoint: `{model_info['base_url']}`"
    )

    # Connection probe (lightweight — no actual call)
    warnings = config.validate()
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.markdown(_pill("● Connected", "green"), unsafe_allow_html=True)

    st.divider()

    # Messages file
    st.markdown("**Messages file**")
    st.code(MESSAGES_FILE, language=None)
    st.caption(
        "To add new messages: append `[Message N | From: ... | Date: ...]` "
        "blocks to that file, then click **Run messages file** below, "
        "or paste them directly in the Updates tab."
    )

    st.divider()

    # Quick-run button  (reads data/messages.txt)
    if st.button("▶ Run messages file", use_container_width=True, type="primary",
                 key="sidebar_run_btn"):
        try:
            raw = open(MESSAGES_FILE, encoding="utf-8").read()
        except FileNotFoundError:
            st.error(f"File not found: {MESSAGES_FILE}")
            raw = ""
        if raw.strip():
            msgs = _split_messages(raw)
            with st.spinner("Processing…"):
                ok, msg = _run_pipeline(msgs)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)
        else:
            st.warning("data/messages.txt is empty.")

    st.divider()

    # Aggregate stats
    st.markdown("**Session stats**")
    s_col1, s_col2 = st.columns(2)
    s_col1.metric("Runs",        S.get("run_count", 0))
    s_col2.metric("Total calls", len(S.get("trace", [])))
    s_col1.metric("Tokens",      f"{S.get('total_tokens', 0):,}")
    s_col2.metric("Time (s)",    f"{S.get('total_duration', 0):.1f}")

    st.divider()

    # Reset
    if st.button("🗑️ Reset all state", use_container_width=True, key="sidebar_reset"):
        st.session_state.app_state = _fresh_state()
        _save_disk_state(st.session_state.app_state)
        st.rerun()

    # Downloads
    st.divider()
    st.markdown("**Export**")
    if S.get("plan"):
        st.download_button(
            "⬇️ Final plan (JSON)",
            json.dumps(S["plan"], indent=2, default=str),
            file_name="final_plan.json", mime="application/json",
            use_container_width=True,
        )
    st.download_button(
        "⬇️ Full state (JSON)",
        json.dumps(S, indent=2, default=str),
        file_name="full_state.json", mime="application/json",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════
tab_overview, tab_plan, tab_updates, tab_rejected, tab_trace = st.tabs([
    "📊 Overview",
    "📋 Event Plan",
    "✏️ Updates",
    "🚫 Rejected",
    "🔍 Execution Trace",
])

plan: dict | None = S.get("plan")

# ───────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW (live dashboard)
# ───────────────────────────────────────────────────────────────
with tab_overview:
    if not plan:
        st.markdown("## Welcome to Mind to Machine")
        st.info(
            "No plan generated yet.  \n\n"
            "**How to get started:**  \n"
            "1. Your message pack is already loaded from `data/messages.txt`.  \n"
            "2. Click **▶ Run messages file** in the sidebar.  \n"
            "3. Or paste messages in the **Updates** tab and click **Process New Messages**.  \n\n"
            "The dashboard will populate with real results from the AI backend."
        )
        st.stop()

    # Validity banner
    _validity_banner(plan)

    st.markdown("### Live Event Metrics")

    ev = plan.get("event", {})
    ve = plan.get("venue", {})
    bu = plan.get("budget", {})
    tr = S.get("trace", [])
    success_tr = [e for e in tr if e.get("status") == "success"]

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("👥 Students",      ev.get("attendees", "—"))
    r1c2.metric("🏆 Teams",         ev.get("teams", "—"))
    r1c3.metric("⏱️ Duration",       f"{ev.get('duration_hours', '—')} hrs")
    r1c4.metric("👷 Staff on duty",  ev.get("staff", "—"))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("🏢 Venue",         ve.get("name", "—"))
    r2c2.metric("💰 Budget used",    _rs(bu.get("total", 0)))
    r2c3.metric("💵 Remaining",      _rs(bu.get("remaining", 0)))
    r2c4.metric(
        "✅ Plan status",
        "VALID" if plan.get("valid") else "INVALID",
        delta=None,
    )

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.metric("🤖 Model calls",    len(tr))
    r3c2.metric("🔢 Tokens (total)", f"{S.get('total_tokens', 0):,}")
    r3c3.metric("⏳ Last run (s)",   S.get("last_run_duration", 0))
    r3c4.metric("🔁 Runs",           S.get("run_count", 0))

    st.divider()

    # Mini check summary
    st.markdown("### Validation Summary")
    checks = plan.get("checks", [])
    passed = [c for c in checks if c["passed"]]
    failed = [c for c in checks if not c["passed"]]

    ch1, ch2 = st.columns(2)
    with ch1:
        for c in passed:
            st.success(f"✔ {c['name']}")
    with ch2:
        for c in failed:
            st.error(f"✘ {c['name']}  —  {c['detail']}")
    if not checks:
        st.info("No validation checks available.")

    # Recalculated areas from last update
    recalc = S.get("last_recalculated_areas", [])
    if recalc:
        st.divider()
        st.markdown("### Last Update")
        unchanged = [a for a in _AREAS_ALL if a not in recalc]
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown("**Recalculated areas**")
            for a in recalc:
                st.markdown(_pill(a, "blue"), unsafe_allow_html=True)
        with a_col2:
            st.markdown("**Unchanged areas**")
            for a in unchanged:
                st.markdown(_pill(a, "grey"), unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — FULL EVENT PLAN
# ───────────────────────────────────────────────────────────────
with tab_plan:
    if not plan:
        st.info("Generate a plan first — use the sidebar or the Updates tab.")
        st.stop()

    _validity_banner(plan)

    plan_tabs = st.tabs(["🏢 Venue", "🗓️ Schedule", "🍱 Food", "⚖️ Judging", "💰 Budget", "🔎 Validation"])

    # ── Venue ──────────────────────────────────────────────────
    with plan_tabs[0]:
        v  = plan.get("venue", {})
        ev = plan.get("event", {})
        st.markdown("#### Venue Selection")

        feasible = v.get("feasible", True)
        if not feasible:
            st.error("⚠️ No venue option meets the required capacity.")
        else:
            st.success("✅ Feasible venue selected — lowest-cost option that meets requirements.")

        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Selected venue**")
            st.markdown(f"### {v.get('name', '—')}")
            gen = v.get("standby_generator", False)
            st.markdown(
                _pill("Standby generator: YES", "green") if gen
                else _pill("Standby generator: NO", "grey"),
                unsafe_allow_html=True,
            )
        with vc2:
            st.markdown("**Capacity check**")
            cap     = v.get("capacity", 0)
            needed  = ev.get("attendees", 0)
            cap_ok  = cap >= needed
            st.metric("Venue capacity",    cap)
            st.metric("Required attendance", needed)
            if cap_ok:
                st.markdown(_pill(f"+{cap - needed} seats spare", "green"), unsafe_allow_html=True)
            else:
                st.markdown(_pill(f"SHORT by {needed - cap}", "red"), unsafe_allow_html=True)

        st.divider()
        st.markdown("**Cost and details**")
        v_data = {
            "Rental cost": _rs(v.get("cost", 0)),
            "Capacity": f"{v.get('capacity', 0):,} seats",
            "Standby generator": "Yes" if v.get("standby_generator") else "No",
            "Selection criterion": "Lowest cost among feasible options",
        }
        for label, value in v_data.items():
            lc, rc = st.columns([1, 2])
            lc.markdown(f"**{label}**")
            rc.markdown(value)

    # ── Schedule ───────────────────────────────────────────────
    with plan_tabs[1]:
        st.markdown("#### Event Schedule")
        schedule = plan.get("schedule", [])
        ev = plan.get("event", {})

        if not schedule:
            st.warning("No schedule generated.")
        else:
            # Timeline table
            import pandas as pd
            rows = []
            for slot in schedule:
                rows.append({
                    "Time slot":  slot.get("time", ""),
                    "Activity":   slot.get("activity", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Event duration", f"{ev.get('duration_hours', '?')} hours")
            sc2.metric("Schedule slots", len(schedule))
            sc3.metric("Teams",          ev.get("teams", "—"))

        # Schedule checks
        sched_checks = [c for c in plan.get("checks", [])
                        if any(w in c["name"].lower()
                               for w in ("schedule", "overlap", "duration"))]
        if sched_checks:
            st.divider()
            st.markdown("**Schedule validation**")
            for c in sched_checks:
                fn = st.success if c["passed"] else st.error
                fn(f"{'PASS' if c['passed'] else 'FAIL'} — {c['name']}: {c['detail']}")

    # ── Food ───────────────────────────────────────────────────
    with plan_tabs[2]:
        st.markdown("#### Food Planning")
        fd = plan.get("food", {})
        ev = plan.get("event", {})

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Meal boxes required",  fd.get("boxes", "—"))
        fc2.metric("Price per person",     _rs(fd.get("unit_price", 0)))
        fc3.metric("Total food cost",      _rs(fd.get("total", 0)))

        st.divider()

        boxes   = fd.get("boxes", 0)
        headct  = ev.get("attendees", 0)
        qty_ok  = boxes >= headct if headct else True
        if qty_ok:
            st.success(f"✅ Quantity OK — {boxes} boxes for {headct} people.")
        else:
            st.error(f"⚠️ Quantity insufficient — {boxes} boxes for {headct} people.")

        st.markdown("**Breakdown**")
        f_data = {
            "Meal type":          fd.get("type", "—"),
            "Boxes":              fd.get("boxes", "—"),
            "Unit price":         _rs(fd.get("unit_price", 0)),
            "Delivery":           _rs(fd.get("delivery", 0)),
            "Total":              _rs(fd.get("total", 0)),
            "Order confirmed":    "Yes" if fd.get("confirmed") else "No (using extracted quantity)",
            "Price source":       fd.get("price_source", "—"),
        }
        for label, value in f_data.items():
            lc, rc = st.columns([1, 2])
            lc.markdown(f"**{label}**")
            rc.markdown(str(value))

    # ── Judging ────────────────────────────────────────────────
    with plan_tabs[3]:
        st.markdown("#### Judging Panel")
        jd = plan.get("judging", {})
        ev = plan.get("event", {})

        jc1, jc2, jc3, jc4 = st.columns(4)
        jc1.metric("Teams",             ev.get("teams", "—"))
        jc2.metric("Judges available",  jd.get("judges", "—"))
        jc3.metric("Judges required",   jd.get("required_judges", "—"))
        jc4.metric("Max teams / judge", jd.get("max_teams_per_judge", "—"))

        st.divider()

        judges   = jd.get("judges", 0)
        required = jd.get("required_judges", 0)
        cap_ok   = judges >= required
        if cap_ok:
            st.success(
                f"✅ Judging capacity OK — {judges} judges cover "
                f"{ev.get('teams', '?')} teams."
            )
        else:
            st.error(
                f"⚠️ Insufficient judges — {judges} available, "
                f"{required} required."
            )

        late = jd.get("late_arrival_buffer_minutes", 0)
        if late:
            st.warning(
                f"⏱️ Late arrival buffer: {late} min — one judge may arrive late. "
                "Effective judging window is reduced accordingly."
            )

        st.markdown("**Team allocation**")
        import math
        teams  = ev.get("teams", 0)
        if judges > 0 and teams > 0:
            per_judge = math.ceil(teams / judges)
            st.info(
                f"With {judges} judges and {teams} teams: "
                f"≈ {per_judge} team(s) per judge "
                f"× {jd.get('minutes_per_team', '?')} min/team."
            )

        st.divider()
        st.markdown("**Cost summary**")
        j_data = {
            "Honorarium each":    _rs(jd.get("honorarium_each", 0)),
            "Total judges cost":  _rs(jd.get("total_cost", 0)),
            "Minutes per team":   f"{jd.get('minutes_per_team', '?')} min",
            "Late buffer":        f"{late} min" if late else "None",
        }
        for label, value in j_data.items():
            lc, rc = st.columns([1, 2])
            lc.markdown(f"**{label}**")
            rc.markdown(str(value))

    # ── Budget ─────────────────────────────────────────────────
    with plan_tabs[4]:
        st.markdown("#### Budget Breakdown")
        b = plan.get("budget", {})

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Approved limit",  _rs(b.get("limit", 0)))
        bc2.metric("Total committed", _rs(b.get("total", 0)))
        remaining = b.get("remaining", 0)
        bc3.metric("Remaining", _rs(remaining),
                   delta=f"{'over' if remaining < 0 else 'under'} budget")

        st.divider()

        # Itemised table
        import pandas as pd
        budget_rows = [
            {"Line item": "Venue",           "Amount (Rs)": f"{b.get('venue', 0):,.0f}",        "Type": "Mandatory"},
            {"Line item": "Food",            "Amount (Rs)": f"{b.get('food', 0):,.0f}",         "Type": "Mandatory"},
            {"Line item": "Judges",          "Amount (Rs)": f"{b.get('judges', 0):,.0f}",       "Type": "Mandatory"},
            {"Line item": "Other committed", "Amount (Rs)": f"{b.get('committed_other', 0):,.0f}", "Type": "Mandatory"},
            {"Line item": "TOTAL",           "Amount (Rs)": f"{b.get('total', 0):,.0f}",        "Type": "—"},
            {"Line item": "REMAINING",       "Amount (Rs)": f"{remaining:,.0f}",                "Type": "—"},
        ]
        df_b = pd.DataFrame(budget_rows)
        st.dataframe(df_b, use_container_width=True, hide_index=True)

        # Optional items
        opt = b.get("optional_items", {})
        if opt:
            st.divider()
            st.markdown("**Optional items (not included in total — would exceed budget if added)**")
            opt_rows = [
                {"Item": k, "Amount (Rs)": f"{v:,.0f}", "Status": "Not committed"}
                for k, v in opt.items()
            ]
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)

        # Budget check
        bud_check = next(
            (c for c in plan.get("checks", []) if c["name"] == "Budget"), None
        )
        if bud_check:
            st.divider()
            fn = st.success if bud_check["passed"] else st.error
            fn(f"{'PASS' if bud_check['passed'] else 'FAIL'} — {bud_check['detail']}")

    # ── Validation ─────────────────────────────────────────────
    with plan_tabs[5]:
        st.markdown("#### All Validation Checks")
        checks = plan.get("checks", [])
        if not checks:
            st.info("No checks recorded.")
        else:
            passed_n = sum(1 for c in checks if c["passed"])
            st.markdown(
                f"**{passed_n} of {len(checks)} checks passed.** "
                + (_pill("Plan VALID", "green") if plan.get("valid") else _pill("Plan INVALID", "red")),
                unsafe_allow_html=True,
            )
            st.divider()
            for c in checks:
                fn = st.success if c["passed"] else st.error
                fn(f"{'PASS' if c['passed'] else 'FAIL'} — **{c['name']}**: {c['detail']}")


# ───────────────────────────────────────────────────────────────
# TAB 3 — MESSAGE UPDATES
# ───────────────────────────────────────────────────────────────
with tab_updates:
    st.markdown("## ✏️ Process New Messages")
    st.markdown(
        "Paste one or more messages exactly as received — English or Roman Urdu. "
        "Duplicate messages are detected automatically and skipped. "
        "Only the planning areas affected by new information are recalculated."
    )

    st.divider()

    new_msg_text = st.text_area(
        "New message(s)",
        height=220,
        placeholder=(
            "[Message 9 | From: Caterer | Date: 15 Nov]\n"
            "Printing cost revised to Rs 15,000.\n\n"
            "[Message 10 | From: Venue | Date: 16 Nov]\n"
            "Hall A is confirmed. Deposit paid."
        ),
        key="update_textarea",
    )

    if st.button("⚡ Process New Messages", type="primary",
                 use_container_width=True, key="update_btn"):
        raw = new_msg_text.strip()
        if not raw:
            st.warning("Please enter at least one message.")
        else:
            msgs = _split_messages(raw)
            known = set(S.get("processed_hashes", []))
            dupes = [m for m in msgs if _hash(m) in known]
            new   = [m for m in msgs if _hash(m) not in known]

            if dupes:
                st.info(f"{len(dupes)} duplicate message(s) skipped — already processed.")
            if not new:
                st.warning("No new messages to process.")
            else:
                prev_plan = deepcopy(S.get("plan"))
                with st.spinner(f"Sending {min(len(new), 5)} message(s) to {config.LLM_BACKEND.upper()}…"):
                    ok, result_msg = _run_pipeline(msgs)

                if ok:
                    st.success(result_msg)
                    curr_plan = S.get("plan")
                    if curr_plan and not curr_plan.get("valid") and prev_plan and prev_plan.get("valid"):
                        st.warning(
                            "⚠️ The updated plan has constraint failures. "
                            "The last valid plan is preserved in **output/state.json** "
                            "(`last_valid_plan` key)."
                        )
                    elif curr_plan and curr_plan.get("valid"):
                        st.success("✅ Updated plan is valid.")

                    recalc   = S.get("last_recalculated_areas", [])
                    unchanged = [a for a in _AREAS_ALL if a not in recalc]

                    u1, u2 = st.columns(2)
                    with u1:
                        st.markdown("**Recalculated areas**")
                        if recalc:
                            for a in recalc:
                                st.markdown(f"🔄 {_pill(a, 'blue')}", unsafe_allow_html=True)
                        else:
                            st.markdown("_(none — no confirmed claims accepted)_")
                    with u2:
                        st.markdown("**Unchanged areas**")
                        for a in unchanged:
                            st.markdown(f"✓ {_pill(a, 'grey')}", unsafe_allow_html=True)

                    st.rerun()
                else:
                    st.info(result_msg)

    # Show last update summary if available
    last_recalc = S.get("last_recalculated_areas", [])
    if last_recalc and S.get("run_count", 0) > 0:
        st.divider()
        st.markdown("**Last update summary**")
        unchanged = [a for a in _AREAS_ALL if a not in last_recalc]
        u1b, u2b = st.columns(2)
        with u1b:
            st.markdown("Recalculated:")
            for a in last_recalc:
                st.markdown(_pill(a, "blue"), unsafe_allow_html=True)
        with u2b:
            st.markdown("Unchanged:")
            for a in unchanged:
                st.markdown(_pill(a, "grey"), unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 4 — REJECTED MESSAGES
# ───────────────────────────────────────────────────────────────
with tab_rejected:
    st.markdown("## 🚫 Rejected Claims")
    st.markdown(
        "These are claims the system **refused to apply** because they violated "
        "a hard constraint, were outdated, unconfirmed, or conflicted with a "
        "higher-confidence claim."
    )

    rejected = S.get("rejected", [])

    if not rejected:
        st.info(
            "No claims have been rejected yet. "
            "Rejected items appear here as the system processes messages."
        )
    else:
        last_valid = S.get("last_valid_plan")
        if last_valid:
            st.success(
                "✅ The last valid plan is preserved. "
                "Rejected claims did not corrupt the plan."
            )

        # Summary row
        rj_col1, rj_col2, rj_col3 = st.columns(3)
        rj_col1.metric("Total rejected",  len(rejected))
        rj_col2.metric(
            "Outdated",
            sum(1 for r in rejected if "outdated" in r.get("reason", "").lower()),
        )
        rj_col3.metric(
            "Constraint violations",
            sum(1 for r in rejected
                if any(w in r.get("reason", "").lower()
                       for w in ("cannot feed", "exceeds", "insufficient", "below"))),
        )

        st.divider()

        for i, item in enumerate(reversed(rejected), 1):
            reason = item.get("reason", "No reason recorded.")
            key    = item.get("key", "?")
            value  = item.get("value", "?")

            # Classify for colour
            r_lower = reason.lower()
            if "outdated" in r_lower:
                badge = _pill("OUTDATED", "yellow")
            elif any(w in r_lower for w in ("cannot feed", "exceeds", "insufficient", "below")):
                badge = _pill("CONSTRAINT VIOLATION", "red")
            elif any(w in r_lower for w in ("suggestion", "tentative", "question", "unconfirmed")):
                badge = _pill("NOT CONFIRMED", "grey")
            else:
                badge = _pill("CONFLICT", "yellow")

            with st.expander(
                f"Rejection #{i} — key: `{key}` · value: `{value}`",
                expanded=(i <= 3),
            ):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"**Reason:** {reason}")

                msg_text = item.get("message", "")
                if msg_text:
                    st.markdown("**Original message:**")
                    st.code(msg_text[:600] + ("…" if len(msg_text) > 600 else ""),
                            language=None)

                # Was the plan preserved?
                if last_valid:
                    st.markdown(
                        _pill("✓ Previous valid plan preserved", "green"),
                        unsafe_allow_html=True,
                    )


# ───────────────────────────────────────────────────────────────
# TAB 5 — EXECUTION TRACE
# ───────────────────────────────────────────────────────────────
with tab_trace:
    st.markdown("## 🔍 Execution Trace")
    st.markdown(
        "Every API call is recorded exactly as it occurred. "
        "Token counts are taken directly from the provider response — "
        "nothing is estimated or fabricated."
    )

    trace = S.get("trace", [])

    if not trace:
        st.info(
            "No model calls recorded yet. "
            "Run the message pack or submit an update to see the trace."
        )
    else:
        success_tr = [e for e in trace if e.get("status") == "success"]
        failed_tr  = [e for e in trace if e.get("status") != "success"]

        total_pt  = sum(e.get("prompt_tokens", 0)     for e in success_tr)
        total_ct  = sum(e.get("completion_tokens", 0) for e in success_tr)
        total_tok = sum(e.get("total_tokens", 0)      for e in success_tr)

        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Total calls",    len(trace))
        t2.metric("Successful",     len(success_tr))
        t3.metric("Failed",         len(failed_tr))
        t4.metric("Prompt tokens",  f"{total_pt:,}")
        t5.metric("Completion tok", f"{total_ct:,}")
        t6.metric("Total tokens",   f"{total_tok:,}")

        st.divider()

        dur_col1, dur_col2 = st.columns(2)
        dur_col1.metric("Total runtime (s)",       S.get("total_duration", 0))
        dur_col2.metric("Last run duration (s)",   S.get("last_run_duration", 0))

        st.divider()
        st.markdown("#### Call-by-call detail")
        st.caption("Newest calls shown first.")

        for entry in reversed(trace):
            status  = entry.get("status", "unknown")
            is_ok   = status == "success"
            call_no = entry.get("call", "?")
            model   = entry.get("model", "?")
            area    = entry.get("area", "?")
            tok     = entry.get("total_tokens", 0)
            dur_e   = entry.get("duration_seconds", 0)

            status_badge = _pill("✓ success", "green") if is_ok else _pill("✗ error", "red")
            area_badge   = _pill(area, "blue")

            header = (
                f"Call #{call_no} &nbsp;{status_badge}&nbsp;{area_badge}&nbsp;"
                f"· {model} · **{tok} tokens** · {dur_e}s"
            )
            with st.expander(f"Call #{call_no} · {area} · {tok} tokens · {dur_e}s"):
                st.markdown(
                    f"{status_badge} &nbsp; {area_badge} &nbsp; model: `{model}`",
                    unsafe_allow_html=True,
                )

                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Input message**")
                    inp = entry.get("input", "")
                    st.code(inp[:800] + ("…" if len(inp) > 800 else ""), language=None)
                with ec2:
                    st.markdown("**LLM output (parsed)**")
                    out = entry.get("output", {})
                    # Show summary and claims list; hide raw for readability
                    if isinstance(out, dict) and "error" not in out:
                        st.markdown(f"*Summary:* {out.get('summary', '—')}")
                        claims_list = out.get("claims", [])
                        if claims_list:
                            import pandas as pd
                            cdf_rows = [
                                {
                                    "key":    c.get("key", ""),
                                    "value":  str(c.get("value", "")),
                                    "area":   c.get("area", ""),
                                    "status": c.get("status", ""),
                                    "conf":   c.get("confidence", ""),
                                }
                                for c in claims_list
                            ]
                            st.dataframe(pd.DataFrame(cdf_rows),
                                         use_container_width=True, hide_index=True)
                        else:
                            st.caption("No claims extracted.")
                    else:
                        st.json(out)

                md = entry.get("merge_decision", {})
                if md:
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown("**Accepted keys**")
                        ak = md.get("accepted_keys", [])
                        st.markdown(", ".join(f"`{k}`" for k in ak) if ak else "_none_")
                    with mc2:
                        st.markdown("**Rejected keys**")
                        rk = md.get("rejected", [])
                        st.markdown(", ".join(f"`{k}`" for k in rk) if rk else "_none_")

                st.markdown(
                    f"Prompt: **{entry.get('prompt_tokens', 0)}** · "
                    f"Completion: **{entry.get('completion_tokens', 0)}** · "
                    f"Total: **{entry.get('total_tokens', 0)}**"
                )
