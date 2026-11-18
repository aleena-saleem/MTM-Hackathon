"""
agents/reader.py — AI message reader for the MTM Hackathon project.

Public API
----------
read_messages(messages, existing_state=None)
    -> (SharedState, list[TraceEntry])

extract_claims(messages, existing_state=None)
    -> (SharedState, list[TraceEntry])

Both functions are identical in behaviour; extract_claims is provided as a
convenient alias that matches the spec naming.

Design notes
------------
- One LLM call per message, using the prompt in agents/prompts.py.
- Routes to Ollama, OpenAI, or Anthropic based on config.LLM_BACKEND.
- Token counts come directly from the API response; never estimated.
- If a call fails, the trace entry records status="error" and zero tokens.
- The function never raises on individual message failures; it records the
  error in the trace and continues with the remaining messages.
- No arithmetic is delegated to the LLM.
- Designed to stay within 5 model calls for a typical update flow by
  processing messages individually (1 call each, up to 5 messages at a time).
"""

from __future__ import annotations

import time
import json
import requests
from copy import deepcopy
from typing import Any

import config
from core.schema import SharedState, TraceEntry, empty_state
from core.parser import parse_llm_response, to_schema_claim
from agents.prompts import build_extraction_messages, VALID_AREAS
from planner import merge_claims   # re-use existing conflict/date logic


# ---------------------------------------------------------------------------
# Internal: single LLM call, returns (raw_text, model_name, usage_dict, duration)
# ---------------------------------------------------------------------------

def _call_groq(chat_messages: list[dict]) -> tuple[str, str, dict, float]:
    """Call Groq via the official groq Python SDK. Returns (content, model, usage, seconds)."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    t0 = time.time()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=chat_messages,       # type: ignore[arg-type]
        temperature=0,
        response_format={"type": "json_object"},
    )
    duration = round(time.time() - t0, 2)
    content    = response.choices[0].message.content or ""
    model_name = response.model or config.GROQ_MODEL
    u = response.usage
    usage = {
        "prompt_tokens":     u.prompt_tokens     if u else 0,
        "completion_tokens": u.completion_tokens if u else 0,
        "total_tokens":      u.total_tokens      if u else 0,
    }
    return content, model_name, usage, duration


def _call_ollama(chat_messages: list[dict]) -> tuple[str, str, dict, float]:
    """Call the Ollama /api/chat endpoint. Returns (content, model, usage, seconds)."""
    url = config.OLLAMA_BASE_URL + "/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": chat_messages,
        "options": {"temperature": 0},
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    model_name = data.get("model", config.OLLAMA_MODEL)
    usage = {
        "prompt_tokens":     int(data.get("prompt_eval_count") or 0),
        "completion_tokens": int(data.get("eval_count") or 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return content, model_name, usage, round(time.time() - t0, 2)


def _call_openai(chat_messages: list[dict]) -> tuple[str, str, dict, float]:
    """Call the OpenAI /v1/chat/completions endpoint."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.OPENAI_MODEL,
        "messages": chat_messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    model_name = data.get("model", config.OPENAI_MODEL)
    u = data.get("usage", {})
    usage = {
        "prompt_tokens":     u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "total_tokens":      u.get("total_tokens", 0),
    }
    return content, model_name, usage, round(time.time() - t0, 2)


def _call_anthropic(chat_messages: list[dict]) -> tuple[str, str, dict, float]:
    """Call the Anthropic /v1/messages endpoint."""
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    # Anthropic separates system from human turns.
    system_content = ""
    human_messages = []
    for m in chat_messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            human_messages.append(m)

    payload: dict[str, Any] = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "messages": human_messages,
    }
    if system_content:
        payload["system"] = system_content

    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["content"][0]["text"]
    model_name = data.get("model", config.ANTHROPIC_MODEL)
    u = data.get("usage", {})
    usage = {
        "prompt_tokens":     u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens":      u.get("input_tokens", 0) + u.get("output_tokens", 0),
    }
    return content, model_name, usage, round(time.time() - t0, 2)


def _dispatch(chat_messages: list[dict]) -> tuple[str, str, dict, float]:
    """Route to the backend selected in config.LLM_BACKEND."""
    backend = config.LLM_BACKEND
    if backend == "groq":
        return _call_groq(chat_messages)
    if backend == "openai":
        return _call_openai(chat_messages)
    if backend == "anthropic":
        return _call_anthropic(chat_messages)
    return _call_ollama(chat_messages)


# ---------------------------------------------------------------------------
# Internal: process one message, return (parsed_doc, trace_entry)
# ---------------------------------------------------------------------------

def _process_one(
    raw_message: str,
    call_index: int,
    message_id: int,
) -> tuple[dict | None, TraceEntry]:
    """
    Send one message to the LLM, parse the response, return the parsed doc
    and a fully-populated TraceEntry.

    On any error the trace entry carries status="error" and zero token counts.
    parsed_doc will be None on failure.
    """
    chat_msgs = build_extraction_messages(raw_message)
    input_text = raw_message   # what we actually sent

    try:
        raw_response, model_name, usage, duration = _dispatch(chat_msgs)
    except Exception as exc:
        entry: TraceEntry = {
            "call":               call_index,
            "model":              config.active_model_info()["model"],
            "area":               "reading",
            "input":              input_text,
            "output":             {"error": str(exc)},
            "prompt_tokens":      0,
            "completion_tokens":  0,
            "total_tokens":       0,
            "duration_seconds":   0.0,
            "status":             "error",
            "merge_decision":     {"accepted_keys": [], "rejected": [], "summary": ""},
        }
        return None, entry

    # Parse the raw JSON response.
    parsed_doc, warnings = parse_llm_response(
        raw_response,
        message_id=message_id,
    )

    # Determine area tag for the trace (must be one of the six valid values).
    area_tag = parsed_doc.get("primary_area", "reading")
    if area_tag not in VALID_AREAS:
        area_tag = "reading"

    entry = TraceEntry(
        call              = call_index,
        model             = model_name,
        area              = area_tag,
        input             = input_text,
        output            = {
            "message_meta":   parsed_doc["message_meta"],
            "primary_area":   parsed_doc["primary_area"],
            "message_type":   parsed_doc["message_type"],
            "summary":        parsed_doc["summary"],
            "affected_areas": parsed_doc["affected_areas"],
            "claims":         parsed_doc["raw_claims"],    # raw for auditability
            "parse_warnings": warnings,
        },
        prompt_tokens     = usage["prompt_tokens"],
        completion_tokens = usage["completion_tokens"],
        total_tokens      = usage["total_tokens"],
        duration_seconds  = duration,
        status            = "success",
        merge_decision    = {"accepted_keys": [], "rejected": [], "summary": parsed_doc["summary"]},
    )
    return parsed_doc, entry


# ---------------------------------------------------------------------------
# Internal: apply one parsed_doc into state, update trace entry in-place
# ---------------------------------------------------------------------------

def _apply_to_state(
    parsed_doc: dict,
    raw_message: str,
    state: SharedState,
    trace_entry: TraceEntry,
) -> None:
    """
    Merge the claims from parsed_doc into state using planner.merge_claims.
    Updates trace_entry["merge_decision"] in-place.
    """
    # planner.merge_claims expects the extraction in agent.py's format:
    # {"message_meta": {...}, "claims": [{key, value, area, status, confidence, rationale}]}
    # Build a compatible dict from our normalised claims.
    planner_claims = []
    for nc in parsed_doc.get("claims", []):
        planner_claims.append({
            "key":        nc["key"],
            "value":      nc["value"],
            "area":       nc["area"],
            "status":     nc["status"],
            "confidence": nc["confidence"],
            "rationale":  nc["rationale"],
        })

    extraction = {
        "message_meta": parsed_doc["message_meta"],
        "primary_area": parsed_doc["primary_area"],
        "claims":       planner_claims,
        "summary":      parsed_doc["summary"],
    }

    accepted, rejected = merge_claims(state, extraction, raw_message)

    trace_entry["merge_decision"] = {
        "accepted_keys": accepted,
        "rejected":      [r.get("key", "") if isinstance(r, dict) else str(r)
                          for r in rejected],
        "summary":       parsed_doc["summary"],
        "affected_areas": parsed_doc.get("affected_areas", []),
        "message_type":   parsed_doc.get("message_type", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_messages(
    messages: list[str],
    existing_state: SharedState | None = None,
) -> tuple[SharedState, list[TraceEntry]]:
    """
    Read a list of raw message strings, extract claims from each via the LLM,
    merge them into state, and return the updated state and full trace.

    Args:
        messages:       List of raw message strings (order is preserved).
        existing_state: Optional existing SharedState to update.
                        If None, a fresh empty state is used.

    Returns:
        (state, trace)
        state: Updated SharedState with claims, rejected items, and messages.
        trace: List of TraceEntry, one per message processed.

    Token counts come directly from the API. Zero means the call failed
    or the provider did not return usage data — never an estimate.
    """
    state: SharedState = deepcopy(existing_state) if existing_state else empty_state()
    trace: list[TraceEntry] = []

    # Determine starting call index (useful when appending to an existing trace).
    call_index = 1

    for msg_idx, raw_message in enumerate(messages):
        raw_message = raw_message.strip()
        if not raw_message:
            continue

        message_id = msg_idx + 1
        parsed_doc, entry = _process_one(raw_message, call_index, message_id)
        call_index += 1

        if parsed_doc is not None:
            _apply_to_state(parsed_doc, raw_message, state, entry)
        # On error, state is unchanged for this message; trace records the failure.

        trace.append(entry)

    return state, trace


def extract_claims(
    messages: list[str],
    existing_state: SharedState | None = None,
) -> tuple[SharedState, list[TraceEntry]]:
    """
    Alias for read_messages().

    Provided so callers can choose the name that best describes their intent.
    Both functions are identical in behaviour.
    """
    return read_messages(messages, existing_state)


# ---------------------------------------------------------------------------
# Convenience: token usage summary across a trace
# ---------------------------------------------------------------------------

def trace_summary(trace: list[TraceEntry]) -> dict:
    """
    Return aggregate token usage across a list of trace entries.
    Only counts successful calls to match agent.py's existing behaviour.
    """
    successful = [e for e in trace if e["status"] == "success"]
    return {
        "calls":              len(trace),
        "successful_calls":   len(successful),
        "failed_calls":       len(trace) - len(successful),
        "prompt_tokens":      sum(e["prompt_tokens"]     for e in successful),
        "completion_tokens":  sum(e["completion_tokens"] for e in successful),
        "total_tokens":       sum(e["total_tokens"]      for e in successful),
    }
