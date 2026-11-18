"""
tests/test_reader.py — Test suite for the Stage 2 AI message reader.

Test categories
---------------
1.  English message extraction                (mocked)
2.  Roman Urdu message extraction             (mocked)
3.  Irrelevant message detection              (mocked)
4.  Correction to an earlier claim            (mocked)
5.  Conflicting information handling          (mocked)
6.  Missing / ambiguous values                (mocked)
7.  Trace structure and token fields          (mocked)
8.  Existing state passed into the reader     (mocked)
9.  Parser: JSON fence stripping              (unit — no model)
10. Parser: malformed JSON                    (unit — no model)
11. Parser: area / status normalisation       (unit — no model)
12. Prompts: build_extraction_messages shape  (unit — no model)

All mocked tests patch agents.reader._dispatch so no real network call is made.
Tests marked REAL_MODEL are skipped automatically when the backend is unreachable.

Run:
    python -m pytest tests/test_reader.py -v
    python -m pytest tests/test_reader.py -v -k "not REAL_MODEL"
"""

from __future__ import annotations

import json
import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or tests/ subdirectory
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from core.schema import empty_state, Claim
from core.parser import parse_llm_response, to_schema_claim, _normalise_area, _normalise_status
from agents.prompts import build_extraction_messages, VALID_AREAS, VALID_STATUSES
from agents.reader import read_messages, extract_claims, trace_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dispatch_response(claims: list[dict], **kwargs) -> tuple[str, str, dict, float]:
    """Build a fake _dispatch return value."""
    doc = {
        "message_meta": {"sender": kwargs.get("sender", "Test Sender"), "date": kwargs.get("date", "1 Jan 2026")},
        "primary_area": kwargs.get("primary_area", "budget"),
        "message_type": kwargs.get("message_type", "fact"),
        "summary": kwargs.get("summary", "Test summary."),
        "affected_areas": kwargs.get("affected_areas", []),
        "claims": claims,
    }
    raw = json.dumps(doc)
    usage = {
        "prompt_tokens":     kwargs.get("prompt_tokens", 42),
        "completion_tokens": kwargs.get("completion_tokens", 18),
        "total_tokens":      kwargs.get("total_tokens", 60),
    }
    return raw, kwargs.get("model", "mock-model"), usage, 0.1


ENGLISH_CLAIM = [
    {"key": "total_budget", "value": 500000, "unit": "PKR",
     "area": "budget", "status": "confirmed", "confidence": 0.95,
     "rationale": "Explicitly stated.", "updates_key": None, "message_id": None}
]

ROMAN_URDU_CLAIM = [
    {"key": "meal_price", "value": 900, "unit": "Rs",
     "area": "food", "status": "confirmed", "confidence": 0.9,
     "rationale": "ab Rs 900 hoga means new price is 900.", "updates_key": "meal_price", "message_id": None}
]

CORRECTION_CLAIM = [
    {"key": "judge_count", "value": 15, "unit": None,
     "area": "judging", "status": "confirmed", "confidence": 0.95,
     "rationale": "Updated from 10 to 15 judges.", "updates_key": "judge_count", "message_id": None}
]

CONFLICTING_CLAIM = [
    {"key": "venue_cost", "value": 80000, "unit": "Rs",
     "area": "venue", "status": "unconfirmed", "confidence": 0.5,
     "rationale": "Conflicts with earlier quote.", "updates_key": None, "message_id": None}
]

AMBIGUOUS_CLAIM = [
    {"key": "team_count", "value": None, "unit": None,
     "area": "schedule", "status": "unconfirmed", "confidence": 0.3,
     "rationale": "Value not stated clearly.", "updates_key": None, "message_id": None}
]


# ===========================================================================
# 1. English message extraction
# ===========================================================================

def test_english_extraction_accepted_claim():
    """A clear English budget message produces an accepted claim in state."""
    msg = "[Message 1 | From: Finance | Date: 5 Jan]\nThe total budget is Rs 500,000."
    dispatch_ret = _make_dispatch_response(ENGLISH_CLAIM, primary_area="budget",
                                           message_type="fact", sender="Finance")
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg])

    assert len(trace) == 1
    assert trace[0]["status"] == "success"
    assert "total_budget" in state["claims"]
    assert state["claims"]["total_budget"]["value"] == 500000
    assert state["claims"]["total_budget"]["area"] == "budget"


def test_english_extraction_trace_fields():
    """TraceEntry has all required fields populated."""
    msg = "[Message 1 | From: Finance | Date: 5 Jan]\nThe total budget is Rs 500,000."
    dispatch_ret = _make_dispatch_response(ENGLISH_CLAIM, prompt_tokens=42,
                                           completion_tokens=18, total_tokens=60)
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        _, trace = read_messages([msg])

    e = trace[0]
    for field in ("call", "model", "area", "input", "output",
                  "prompt_tokens", "completion_tokens", "total_tokens",
                  "duration_seconds", "status", "merge_decision"):
        assert field in e, f"Missing TraceEntry field: {field}"
    assert e["call"] == 1
    assert e["prompt_tokens"] == 42
    assert e["completion_tokens"] == 18
    assert e["total_tokens"] == 60


# ===========================================================================
# 2. Roman Urdu message extraction
# ===========================================================================

def test_roman_urdu_price_update():
    """A Roman Urdu price update ('ab Rs X hoga') is extracted as a confirmed claim."""
    msg = "[Message 2 | From: Caterer | Date: 10 Jan]\nSalam! ab Rs 900 hoga per head."
    dispatch_ret = _make_dispatch_response(ROMAN_URDU_CLAIM, primary_area="food",
                                           message_type="update", sender="Caterer")
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg])

    assert trace[0]["status"] == "success"
    assert "meal_price" in state["claims"]
    assert state["claims"]["meal_price"]["value"] == 900


def test_roman_urdu_question_not_applied():
    """A Roman Urdu question/suggestion is NOT stored in active claims."""
    suggestion_claim = [
        {"key": "dj_cost", "value": 50000, "unit": "Rs",
         "area": "budget", "status": "suggestion", "confidence": 0.6,
         "rationale": "kar sakte hain? is a question, not approval.",
         "updates_key": None, "message_id": None}
    ]
    msg = "[Message 3 | From: Student society | Date: 11 Jan]\nDJ kar sakte hain? Rs 50,000 quote hai."
    dispatch_ret = _make_dispatch_response(suggestion_claim, primary_area="budget",
                                           message_type="suggestion", sender="Student society")
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg])

    assert "dj_cost" not in state["claims"], "Suggestion should not enter active claims"
    # It should appear in rejected
    rejected_keys = [r.get("key") for r in state["rejected"]]
    assert "dj_cost" in rejected_keys


# ===========================================================================
# 3. Irrelevant message detection
# ===========================================================================

def test_irrelevant_message_no_claims():
    """An irrelevant message produces no active claims and no rejected items."""
    msg = "[Message 4 | From: Random | Date: 12 Jan]\nGreat weather today! See you all soon."
    dispatch_ret = _make_dispatch_response(
        [], primary_area="budget", message_type="irrelevant",
        summary="Message is irrelevant to event planning."
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg])

    assert trace[0]["status"] == "success"
    assert len(state["claims"]) == 0
    assert len(state["rejected"]) == 0
    assert state["messages"][0] == msg


def test_irrelevant_message_recorded_in_trace():
    """An irrelevant message still gets a trace entry with the correct message_type."""
    msg = "Nothing useful here."
    dispatch_ret = _make_dispatch_response(
        [], message_type="irrelevant", summary="Irrelevant."
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        _, trace = read_messages([msg])

    assert trace[0]["output"]["message_type"] == "irrelevant"


# ===========================================================================
# 4. Correction to an earlier claim
# ===========================================================================

def test_correction_overwrites_older_claim():
    """A correction dated later than an existing claim should replace it."""
    # Seed state with an earlier claim (date 5 Jan)
    initial_state = empty_state()
    initial_state["claims"]["judge_count"] = {
        "value": 10, "area": "judging",
        "date": "05 Jan 2026", "sender": "Coordinator",
        "confidence": 0.9, "rationale": "Initial count."
    }

    # Correction arrives dated 12 Jan (newer)
    msg = "[Message 5 | From: Coordinator | Date: 12 Jan]\nUpdated: we now have 15 judges confirmed."
    dispatch_ret = _make_dispatch_response(
        CORRECTION_CLAIM, primary_area="judging",
        message_type="correction", sender="Coordinator", date="12 Jan"
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg], existing_state=initial_state)

    assert state["claims"]["judge_count"]["value"] == 15
    assert trace[0]["merge_decision"]["accepted_keys"] == ["judge_count"]


def test_correction_older_date_rejected():
    """A correction dated OLDER than the active claim should be rejected."""
    initial_state = empty_state()
    initial_state["claims"]["judge_count"] = {
        "value": 15, "area": "judging",
        "date": "12 Jan 2026", "sender": "Coordinator",
        "confidence": 0.9, "rationale": "Latest count."
    }

    # Stale message dated 3 Jan (older)
    stale_claim = [{
        "key": "judge_count", "value": 10, "unit": None,
        "area": "judging", "status": "confirmed", "confidence": 0.8,
        "rationale": "Old stale message.", "updates_key": None, "message_id": None
    }]
    msg = "[Message 6 | From: Coordinator | Date: 3 Jan]\nWe have 10 judges."
    dispatch_ret = _make_dispatch_response(
        stale_claim, primary_area="judging",
        message_type="fact", sender="Coordinator", date="3 Jan"
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg], existing_state=initial_state)

    # Value must remain 15 (the newer claim)
    assert state["claims"]["judge_count"]["value"] == 15
    # The stale claim should appear in rejected
    rejected_keys = [r.get("key") for r in state["rejected"]]
    assert "judge_count" in rejected_keys


# ===========================================================================
# 5. Conflicting information
# ===========================================================================

def test_conflicting_claim_marked_unconfirmed_rejected():
    """A claim with status=unconfirmed should be rejected by merge_claims."""
    initial_state = empty_state()
    initial_state["claims"]["venue_cost"] = {
        "value": 120000, "area": "venue",
        "date": "08 Jan 2026", "sender": "Venue A",
        "confidence": 0.95, "rationale": "Confirmed quote."
    }

    msg = "[Message 7 | From: Unknown | Date: 8 Jan]\nVenue cost is Rs 80,000."
    dispatch_ret = _make_dispatch_response(
        CONFLICTING_CLAIM, primary_area="venue",
        message_type="fact", sender="Unknown", date="8 Jan"
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg], existing_state=initial_state)

    # Unconfirmed same-date lower-confidence claim should be rejected
    assert state["claims"]["venue_cost"]["value"] == 120000
    rejected_keys = [r.get("key") for r in state["rejected"]]
    assert "venue_cost" in rejected_keys


# ===========================================================================
# 6. Missing / ambiguous values
# ===========================================================================

def test_ambiguous_value_stored_as_none():
    """A claim where the model returns value=None is stored faithfully."""
    msg = "[Message 8 | From: Organiser | Date: 9 Jan]\nNumber of teams is unclear."
    dispatch_ret = _make_dispatch_response(
        AMBIGUOUS_CLAIM, primary_area="schedule",
        message_type="fact", sender="Organiser",
        summary="Team count ambiguous."
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg])

    # status=unconfirmed → rejected by merge_claims; value not in active claims
    assert "team_count" not in state["claims"]
    rejected_keys = [r.get("key") for r in state["rejected"]]
    assert "team_count" in rejected_keys


def test_empty_message_skipped():
    """An empty string is skipped without generating a trace entry."""
    with patch("agents.reader._dispatch") as mock_dispatch:
        state, trace = read_messages(["", "   ", "\n"])
    mock_dispatch.assert_not_called()
    assert len(trace) == 0


# ===========================================================================
# 7. Model trace and token usage
# ===========================================================================

def test_trace_tokens_come_from_api():
    """Token counts in the trace come from the mocked API — not estimated."""
    msg = "[Message 9 | From: Dean | Date: 10 Jan]\nBudget approved: Rs 650,000."
    dispatch_ret = _make_dispatch_response(
        ENGLISH_CLAIM, prompt_tokens=123, completion_tokens=77, total_tokens=200
    )
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        _, trace = read_messages([msg])

    assert trace[0]["prompt_tokens"]     == 123
    assert trace[0]["completion_tokens"] == 77
    assert trace[0]["total_tokens"]      == 200


def test_trace_error_on_connection_failure():
    """A connection failure produces a trace entry with status=error and zero tokens."""
    msg = "[Message 10 | From: Anyone | Date: 11 Jan]\nSome message."
    with patch("agents.reader._dispatch", side_effect=ConnectionError("refused")):
        state, trace = read_messages([msg])

    assert trace[0]["status"] == "error"
    assert trace[0]["prompt_tokens"]     == 0
    assert trace[0]["completion_tokens"] == 0
    assert trace[0]["total_tokens"]      == 0
    # State should be unmodified (no crash, no phantom claims)
    assert len(state["claims"]) == 0


def test_trace_summary_counts():
    """trace_summary() correctly counts successful vs failed calls."""
    mock_trace = [
        {"status": "success", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"status": "success", "prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        {"status": "error",   "prompt_tokens": 0,  "completion_tokens": 0, "total_tokens": 0},
    ]
    summary = trace_summary(mock_trace)
    assert summary["calls"]             == 3
    assert summary["successful_calls"]  == 2
    assert summary["failed_calls"]      == 1
    assert summary["prompt_tokens"]     == 30
    assert summary["completion_tokens"] == 13
    assert summary["total_tokens"]      == 43


# ===========================================================================
# 8. Existing state passed into the reader
# ===========================================================================

def test_existing_state_preserved():
    """Claims from an existing state are preserved when new messages are processed."""
    initial_state = empty_state()
    initial_state["claims"]["people_count"] = {
        "value": 200, "area": "schedule",
        "date": "01 Jan 2026", "sender": "Dean",
        "confidence": 1.0, "rationale": "Pre-seeded."
    }

    msg = "[Message 11 | From: Finance | Date: 5 Jan]\nBudget is Rs 700,000."
    new_claim = [{
        "key": "total_budget", "value": 700000, "unit": "PKR",
        "area": "budget", "status": "confirmed", "confidence": 0.95,
        "rationale": "Explicitly stated.", "updates_key": None, "message_id": None
    }]
    dispatch_ret = _make_dispatch_response(new_claim, primary_area="budget",
                                           sender="Finance", date="5 Jan")
    with patch("agents.reader._dispatch", return_value=dispatch_ret):
        state, trace = read_messages([msg], existing_state=initial_state)

    # Both the pre-seeded and new claims should be present
    assert "people_count" in state["claims"]
    assert state["claims"]["people_count"]["value"] == 200
    assert "total_budget" in state["claims"]
    assert state["claims"]["total_budget"]["value"] == 700000


def test_existing_state_not_mutated_on_error():
    """The original existing_state dict is not mutated when a call fails."""
    initial_state = empty_state()
    initial_state["claims"]["people_count"] = {
        "value": 150, "area": "schedule",
        "date": "01 Jan 2026", "sender": "Dean",
        "confidence": 1.0, "rationale": "Original."
    }
    original_copy = json.loads(json.dumps(initial_state))  # deep copy via JSON

    msg = "Some message."
    with patch("agents.reader._dispatch", side_effect=RuntimeError("fail")):
        state, _ = read_messages([msg], existing_state=initial_state)

    # initial_state should be unchanged (reader deepcopies before modifying)
    assert initial_state["claims"] == original_copy["claims"]


# ===========================================================================
# Parser unit tests (no model needed)
# ===========================================================================

def test_parser_strips_markdown_fence():
    """parse_llm_response handles JSON wrapped in markdown code fences."""
    inner = json.dumps({
        "message_meta": {"sender": "X", "date": "1 Jan"},
        "primary_area": "food",
        "message_type": "fact",
        "summary": "Test.",
        "affected_areas": ["food"],
        "claims": [],
    })
    fenced = f"```json\n{inner}\n```"
    doc, warnings = parse_llm_response(fenced)
    assert doc["primary_area"] == "food"
    assert warnings == []


def test_parser_malformed_json_returns_empty():
    """parse_llm_response returns an empty doc and a warning on malformed JSON."""
    doc, warnings = parse_llm_response("not json at all {{")
    assert doc["claims"] == []
    assert any("parse error" in w.lower() or "no json" in w.lower() for w in warnings)


def test_parser_area_normalisation():
    """_normalise_area maps aliases correctly."""
    assert _normalise_area("venues")   == "venue"
    assert _normalise_area("catering") == "food"
    assert _normalise_area("JUDGING")  == "judging"
    assert _normalise_area("unknown")  == "budget"   # fallback


def test_parser_status_normalisation():
    """_normalise_status falls back to 'unconfirmed' for unrecognised values."""
    assert _normalise_status("confirmed")  == "confirmed"
    assert _normalise_status("TENTATIVE")  == "tentative"
    assert _normalise_status("gibberish")  == "unconfirmed"


# ===========================================================================
# Prompts unit tests (no model needed)
# ===========================================================================

def test_build_extraction_messages_structure():
    """build_extraction_messages returns exactly two messages with correct roles."""
    msgs = build_extraction_messages("Test message body.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Test message body." in msgs[1]["content"]


def test_extraction_message_contains_message_body():
    """The user turn contains the original message text unchanged."""
    body = "Special chars: & < > \" ' Rs 1,000"
    msgs = build_extraction_messages(body)
    assert body in msgs[1]["content"]


# ===========================================================================
# REAL_MODEL tests — skipped when backend unreachable
# ===========================================================================

def _backend_reachable() -> bool:
    """Quick probe to see if the configured backend is reachable."""
    import requests as _req
    import config as _cfg
    try:
        if _cfg.LLM_BACKEND == "ollama":
            _req.get(_cfg.OLLAMA_BASE_URL + "/api/tags", timeout=3)
        elif _cfg.LLM_BACKEND == "openai":
            if not _cfg.OPENAI_API_KEY or _cfg.OPENAI_API_KEY.startswith("sk-xxxxx"):
                return False
        elif _cfg.LLM_BACKEND == "anthropic":
            if not _cfg.ANTHROPIC_API_KEY:
                return False
        return True
    except Exception:
        return False


REAL_MODEL = pytest.mark.skipif(
    not _backend_reachable(),
    reason="Configured LLM backend is not reachable — skipping real model test."
)


@REAL_MODEL
def test_real_model_english_extraction():
    """
    REAL MODEL TEST — makes an actual API call.
    Sends a simple English message and verifies the trace records
    a real model name and non-zero token counts.
    """
    msg = "[Message 1 | From: Test | Date: 1 Jan]\nThe event will have 100 attendees."
    state, trace = read_messages([msg])

    assert len(trace) == 1
    e = trace[0]
    assert e["status"] == "success", f"Call failed: {e['output']}"
    # Model name must come from the API, not be empty
    assert e["model"], "Model name should not be empty"
    # Token counts must be positive integers from the API
    assert e["prompt_tokens"]     > 0, "prompt_tokens should be > 0"
    assert e["completion_tokens"] > 0, "completion_tokens should be > 0"
    assert e["total_tokens"]      == e["prompt_tokens"] + e["completion_tokens"]
    print(f"\n[REAL_MODEL] model={e['model']} tokens={e['total_tokens']} "
          f"({e['prompt_tokens']}+{e['completion_tokens']})")
