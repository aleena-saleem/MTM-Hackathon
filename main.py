"""
main.py — One-command entry point for the MTM event planning system.

Usage (both initial run and updates):
    python main.py

Behaviour
---------
On first run:
  1. Reads data/messages.txt.
  2. Sends every message to the configured LLM (Groq by default).
  3. Extracts claims, applies conflict detection and validation.
  4. Generates a complete event plan across five planning areas.
  5. Saves state + plan + trace to output/state.json.
  6. Prints the plan summary and trace to stdout.

On subsequent runs (same command):
  1. Loads the saved state from output/state.json.
  2. Reads data/messages.txt again.
  3. Deduplicates messages already processed (by content hash).
  4. Sends ONLY new messages to the LLM (max 5 calls per update).
  5. Applies incremental updates — only affected areas are recalculated.
  6. Preserves the last valid plan if the new plan is invalid.
  7. Saves updated state + plan + trace back to output/state.json.

No flags or manual code changes are needed to switch between modes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from copy import deepcopy

import config
from agent import create_agent
from planner import build_plan
from core.planner import generate_plan
from core.schema import empty_state, empty_output

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
MESSAGES_FILE = os.path.join(ROOT, "data", "messages.txt")
OUTPUT_DIR  = os.path.join(ROOT, "output")
STATE_FILE  = os.path.join(OUTPUT_DIR, "state.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

def load_messages(path: str) -> list[str]:
    """Split a plain-text message pack on [Message N ...] headers."""
    try:
        text = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"ERROR: Messages file not found: {path}", file=sys.stderr)
        sys.exit(1)

    parts = [x.strip() for x in re.split(r"(?=\[Message\s+\d+)", text) if x.strip()]
    # Drop comment lines (lines starting with #) from each part.
    cleaned = []
    for p in parts:
        body = "\n".join(
            line for line in p.splitlines() if not line.strip().startswith("#")
        ).strip()
        if body:
            cleaned.append(body)
    return cleaned or ([text.strip()] if text.strip() else [])


def _hash(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Persistent state helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load state from disk, or return a fresh empty state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all required keys exist (state may be from an earlier version).
            data.setdefault("claims", {})
            data.setdefault("rejected", [])
            data.setdefault("messages", [])
            data.setdefault("processed_hashes", [])
            data.setdefault("trace", [])
            data.setdefault("plan", None)
            data.setdefault("last_valid_plan", None)
            data.setdefault("run_count", 0)
            data.setdefault("total_tokens", 0)
            data.setdefault("total_duration", 0.0)
            return data
        except (json.JSONDecodeError, KeyError):
            print("[WARN] Corrupted state file — starting fresh.", file=sys.stderr)

    fresh = empty_state()
    fresh["processed_hashes"] = []
    fresh["trace"]             = []
    fresh["plan"]              = None
    fresh["last_valid_plan"]   = None
    fresh["run_count"]         = 0
    fresh["total_tokens"]      = 0
    fresh["total_duration"]    = 0.0
    return fresh


def save_state(state: dict) -> None:
    """Persist state to output/state.json."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 70) -> str:
    return char * width


def _print_plan(plan: dict) -> None:
    print(_sep("═"))
    print("  EVENT PLAN")
    print(_sep("═"))

    ev = plan.get("event", {})
    print(f"  Attendees : {ev.get('attendees', '?')}   "
          f"Teams : {ev.get('teams', '?')}   "
          f"Duration : {ev.get('duration_hours', '?')} hours   "
          f"Staff : {ev.get('staff', '?')}")

    print(_sep())
    v = plan.get("venue", {})
    print(f"  VENUE     : {v.get('name')}  "
          f"capacity={v.get('capacity')}  "
          f"cost=Rs {v.get('cost', 0):,.0f}  "
          f"generator={'yes' if v.get('standby_generator') else 'no'}")

    print(_sep())
    print("  SCHEDULE")
    for slot in plan.get("schedule", []):
        print(f"    {slot.get('time',''):>15}  {slot.get('activity','')}")

    print(_sep())
    fd = plan.get("food", {})
    print(f"  FOOD      : {fd.get('type')}  "
          f"boxes={fd.get('boxes')}  "
          f"@ Rs {fd.get('unit_price', 0):,.0f}  "
          f"delivery=Rs {fd.get('delivery', 0):,.0f}  "
          f"total=Rs {fd.get('total', 0):,.0f}")

    print(_sep())
    jd = plan.get("judging", {})
    print(f"  JUDGING   : judges={jd.get('judges')}  "
          f"required={jd.get('required_judges')}  "
          f"honorarium=Rs {jd.get('honorarium_each', 0):,.0f} each  "
          f"total=Rs {jd.get('total_cost', 0):,.0f}")
    if jd.get("late_arrival_buffer_minutes"):
        print(f"              late buffer={jd['late_arrival_buffer_minutes']} min")

    print(_sep())
    b = plan.get("budget", {})
    print(f"  BUDGET    : limit=Rs {b.get('limit', 0):,.0f}  "
          f"total=Rs {b.get('total', 0):,.0f}  "
          f"remaining=Rs {b.get('remaining', 0):,.0f}")

    print(_sep())
    checks = plan.get("checks", [])
    print("  VALIDATION CHECKS")
    for c in checks:
        icon = "PASS" if c["passed"] else "FAIL"
        print(f"    [{icon}] {c['name']:30s} {c['detail']}")

    status = "VALID ✓" if plan.get("valid") else "INVALID ✗ — see FAIL checks above"
    print(_sep("═"))
    print(f"  PLAN STATUS: {status}")
    print(_sep("═"))


def _print_rejected(rejected: list[dict]) -> None:
    if not rejected:
        print("  No messages were rejected.")
        return
    for i, item in enumerate(rejected, 1):
        print(f"  [{i}] key={item.get('key', '?')}  value={item.get('value', '?')}")
        print(f"       reason: {item.get('reason', '?')}")


def _print_trace(trace: list[dict], new_calls: int = 0) -> None:
    shown = trace[-new_calls:] if new_calls else trace
    for entry in shown:
        status = entry.get("status", "?")
        icon   = "✓" if status == "success" else "✗"
        print(f"  Call #{entry.get('call'):>2} {icon}  "
              f"model={entry.get('model')}  "
              f"area={entry.get('area')}  "
              f"tokens={entry.get('total_tokens')}  "
              f"({entry.get('duration_seconds')}s)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    t_start = time.time()

    # --- Load messages from disk ------------------------------------------
    all_messages = load_messages(MESSAGES_FILE)
    if not all_messages:
        print("No messages found in data/messages.txt — nothing to do.")
        return

    # --- Load saved state -------------------------------------------------
    state = load_state()
    is_initial = state["run_count"] == 0

    # --- Deduplicate: find messages not yet processed ---------------------
    known_hashes: set[str] = set(state.get("processed_hashes", []))
    new_messages  = [m for m in all_messages if _hash(m) not in known_hashes]

    if not new_messages:
        print("No new messages found. Add messages to data/messages.txt to trigger an update.")
        if state["plan"]:
            _print_plan(state["plan"])
        return

    mode = "INITIAL RUN" if is_initial else f"INCREMENTAL UPDATE ({len(new_messages)} new message(s))"
    print(f"\n{'='*70}")
    print(f"  Mind to Machine — Event Planning System")
    print(f"  Mode    : {mode}")
    print(f"  Backend : {config.LLM_BACKEND.upper()}  model={config.active_model_info()['model']}")
    print(f"  Messages: {len(new_messages)} to process  ({len(all_messages)} total in file)")
    print(f"{'='*70}\n")

    # --- Cap at 5 model calls for updates ---------------------------------
    MAX_CALLS = 5
    if not is_initial and len(new_messages) > MAX_CALLS:
        print(f"[INFO] Update capped at {MAX_CALLS} messages per run. "
              f"{len(new_messages) - MAX_CALLS} will be processed next run.")
        new_messages = new_messages[:MAX_CALLS]

    # --- Run agent (one LLM call per message) -----------------------------
    # Build a minimal state dict compatible with the agent (claims/rejected/messages).
    agent_state = {
        "claims":   state["claims"],
        "rejected": state["rejected"],
        "messages": state["messages"],
    }

    agent = create_agent()
    call_offset = len(state["trace"])

    # Offset the agent's trace call numbers so they continue from last run.
    agent_state_result, _plan_from_agent = agent.run_messages(new_messages, agent_state)

    # Adjust call numbers in the new trace entries.
    for i, entry in enumerate(agent.trace):
        entry["call"] = call_offset + i + 1

    # --- Merge agent results back into persistent state -------------------
    state["claims"]   = agent_state_result["claims"]
    state["rejected"] = agent_state_result["rejected"]
    state["messages"] = agent_state_result["messages"]

    # Record processed hashes.
    for m in new_messages:
        h = _hash(m)
        if h not in known_hashes:
            state["processed_hashes"].append(h)
            known_hashes.add(h)

    # Append new trace entries.
    state["trace"].extend(agent.trace)

    # --- Generate plan using Stage 3 engine -------------------------------
    new_plan = generate_plan(state)

    # Preserve last valid plan.
    if new_plan.get("valid"):
        state["last_valid_plan"] = deepcopy(new_plan)
    elif state.get("last_valid_plan") is None:
        # No valid plan yet — store the current one anyway.
        state["last_valid_plan"] = deepcopy(new_plan)

    state["plan"] = new_plan

    # --- Update aggregate counters ----------------------------------------
    this_run_tokens   = sum(e.get("total_tokens", 0) for e in agent.trace if e.get("status") == "success")
    this_run_duration = round(time.time() - t_start, 2)
    state["total_tokens"]   = state.get("total_tokens", 0) + this_run_tokens
    state["total_duration"] = round(state.get("total_duration", 0.0) + this_run_duration, 2)
    state["run_count"]      = state.get("run_count", 0) + 1

    # --- Save state -------------------------------------------------------
    save_state(state)

    # --- Print results ----------------------------------------------------
    print()
    _print_plan(new_plan)

    print(f"\n{_sep()}")
    print("  REJECTED CLAIMS")
    print(_sep())
    _print_rejected(state["rejected"])

    print(f"\n{_sep()}")
    print(f"  MODEL TRACE — this run ({len(agent.trace)} call(s))")
    print(_sep())
    _print_trace(state["trace"], new_calls=len(agent.trace))

    summary = agent.trace_summary()
    print(f"\n  Calls this run   : {summary['calls']}")
    print(f"  Tokens this run  : {summary['total_tokens']}")
    print(f"  Total tokens ever: {state['total_tokens']}")
    print(f"  Duration         : {this_run_duration}s")
    print(f"  State saved to   : {STATE_FILE}")

    sys.exit(0 if new_plan.get("valid") else 2)


if __name__ == "__main__":
    run()
