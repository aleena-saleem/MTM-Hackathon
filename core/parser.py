"""
core/parser.py — Parse and normalise raw LLM JSON output into SharedState-compatible structures.

Responsibilities:
  1. Parse the JSON string returned by the model (handles malformed output gracefully).
  2. Validate and normalise field values against the schema.
  3. Convert each raw claim into a schema.Claim dict.
  4. Flag structural problems without raising — callers decide whether to abort.

This module does NO arithmetic and makes NO model calls.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from core.schema import Claim
from agents.prompts import VALID_AREAS, VALID_STATUSES, VALID_MESSAGE_TYPES

# ---------------------------------------------------------------------------
# Area aliases — map any variant the model might return to a canonical value
# ---------------------------------------------------------------------------
_AREA_ALIASES: dict[str, str] = {
    "venue": "venue", "venues": "venue", "location": "venue",
    "schedule": "schedule", "scheduling": "schedule", "timing": "schedule",
    "food": "food", "catering": "food", "meals": "food",
    "judging": "judging", "judges": "judging", "evaluation": "judging",
    "budget": "budget", "finance": "budget", "cost": "budget",
    "reading": "budget",          # internal fallback tag
}


def _normalise_area(raw: Any) -> str:
    """Return a valid area string; default to 'budget' if unrecognisable."""
    s = str(raw).lower().strip()
    return _AREA_ALIASES.get(s, "budget")


def _normalise_status(raw: Any) -> str:
    """Return a valid status string; default to 'unconfirmed'."""
    s = str(raw).lower().strip()
    return s if s in VALID_STATUSES else "unconfirmed"


def _normalise_message_type(raw: Any) -> str:
    """Return a valid message_type string; default to 'fact'."""
    s = str(raw).lower().strip()
    return s if s in VALID_MESSAGE_TYPES else "fact"


def _parse_date(raw: Any) -> str:
    """
    Parse a date string into 'DD Mon YYYY' format.
    Returns the raw string unchanged if it cannot be parsed,
    so provenance is preserved.
    """
    if not raw or str(raw).lower() in ("unknown", "none", "null", ""):
        return "Unknown"
    s = str(raw).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%d %b", "%d %B",
                "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            d = datetime.strptime(s, fmt)
            year = d.year if d.year != 1900 else 2026
            return d.replace(year=year).strftime("%d %b %Y")
        except ValueError:
            pass
    return s   # return as-is — planner._date() has its own fallback


def _safe_confidence(raw: Any) -> float:
    """Clamp confidence to [0.0, 1.0]; default to 0.7 on failure."""
    try:
        v = float(raw)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.7


# ---------------------------------------------------------------------------
# Primary parsing entry point
# ---------------------------------------------------------------------------

def parse_llm_response(
    raw_text: str,
    message_id: int | None = None,
    sender_fallback: str = "Unknown",
    date_fallback: str = "Unknown",
) -> tuple[dict, list[str]]:
    """
    Parse the raw JSON string returned by the model.

    Args:
        raw_text:        The model's response string.
        message_id:      Sequential message number (stamped onto each claim).
        sender_fallback: Used if the model omits sender.
        date_fallback:   Used if the model omits date.

    Returns:
        (parsed_doc, warnings)
        parsed_doc keys:
            message_meta   – {sender, date}
            primary_area   – normalised area string
            message_type   – normalised message type string
            summary        – str
            affected_areas – list of valid area strings
            claims         – list of raw claim dicts (not yet Claim TypedDicts)
            raw_claims     – the claims list before normalisation (for trace)
        warnings:
            List of human-readable issues found during parsing.
    """
    warnings: list[str] = []

    # --- Step 1: extract JSON from the response --------------------------------
    # Models sometimes wrap JSON in markdown fences.
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    elif not text.startswith("{"):
        # Try to find the first '{' — the model may have prepended prose.
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
        else:
            warnings.append("Response contains no JSON object.")
            return _empty_parsed_doc(sender_fallback, date_fallback), warnings

    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        warnings.append(f"JSON parse error: {e}")
        return _empty_parsed_doc(sender_fallback, date_fallback), warnings

    if not isinstance(doc, dict):
        warnings.append("Parsed JSON is not a dict.")
        return _empty_parsed_doc(sender_fallback, date_fallback), warnings

    # --- Step 2: normalise top-level fields ------------------------------------
    meta = doc.get("message_meta") or {}
    if not isinstance(meta, dict):
        meta = {}
        warnings.append("message_meta missing or not a dict.")

    sender = str(meta.get("sender") or sender_fallback).strip() or sender_fallback
    date   = _parse_date(meta.get("date") or date_fallback)

    primary_area   = _normalise_area(doc.get("primary_area", "budget"))
    message_type   = _normalise_message_type(doc.get("message_type", "fact"))
    summary        = str(doc.get("summary") or "").strip()

    raw_affected = doc.get("affected_areas") or []
    affected_areas = [_normalise_area(a) for a in raw_affected
                      if _normalise_area(a) in VALID_AREAS]
    if primary_area not in affected_areas:
        affected_areas.insert(0, primary_area)

    # --- Step 3: normalise claims ----------------------------------------------
    raw_claims = doc.get("claims") or []
    if not isinstance(raw_claims, list):
        warnings.append("'claims' field is not a list; treating as empty.")
        raw_claims = []

    normalised_claims: list[dict] = []
    seen_keys: set[str] = set()

    for i, claim in enumerate(raw_claims):
        if not isinstance(claim, dict):
            warnings.append(f"Claim {i} is not a dict; skipped.")
            continue

        key = str(claim.get("key") or "").strip().lower()
        # Replace spaces/hyphens with underscores; strip non-identifier chars.
        key = re.sub(r"[^a-z0-9_]", "_", key).strip("_")
        if not key:
            warnings.append(f"Claim {i} has no key; skipped.")
            continue
        if key in seen_keys:
            warnings.append(f"Duplicate key '{key}' at claim {i}; skipped.")
            continue
        seen_keys.add(key)

        value      = claim.get("value")       # keep as-is; planner handles coercion
        unit       = claim.get("unit")        # str or None
        area       = _normalise_area(claim.get("area", primary_area))
        status     = _normalise_status(claim.get("status", "confirmed"))
        confidence = _safe_confidence(claim.get("confidence", 0.7))
        rationale  = str(claim.get("rationale") or "").strip()
        updates_key = claim.get("updates_key")
        if updates_key:
            updates_key = str(updates_key).strip().lower() or None

        normalised_claims.append({
            "key":         key,
            "value":       value,
            "unit":        unit,
            "area":        area,
            "status":      status,
            "confidence":  confidence,
            "rationale":   rationale,
            "updates_key": updates_key,
            "message_id":  message_id,
            "sender":      sender,
            "date":        date,
        })

    parsed_doc = {
        "message_meta":   {"sender": sender, "date": date},
        "primary_area":   primary_area,
        "message_type":   message_type,
        "summary":        summary,
        "affected_areas": affected_areas,
        "claims":         normalised_claims,
        "raw_claims":     raw_claims,     # preserved for trace
    }
    return parsed_doc, warnings


# ---------------------------------------------------------------------------
# Convert a normalised claim dict to a schema.Claim TypedDict
# ---------------------------------------------------------------------------

def to_schema_claim(norm: dict) -> Claim:
    """
    Convert a normalised claim dict (from parse_llm_response) to a
    schema.Claim TypedDict using the exact field names the schema expects.

    Fields not in Claim (unit, updates_key, message_id) are dropped here;
    they live in the trace entry instead.
    """
    area = norm["area"]
    if area not in VALID_AREAS:
        area = "budget"
    return Claim(
        value      = norm["value"],
        area       = area,            # type: ignore[arg-type]
        date       = norm["date"],
        sender     = norm["sender"],
        confidence = norm["confidence"],
        rationale  = norm["rationale"],
    )


# ---------------------------------------------------------------------------
# Helper: empty document on parse failure
# ---------------------------------------------------------------------------

def _empty_parsed_doc(sender: str, date: str) -> dict:
    return {
        "message_meta":   {"sender": sender, "date": date},
        "primary_area":   "budget",
        "message_type":   "fact",
        "summary":        "",
        "affected_areas": [],
        "claims":         [],
        "raw_claims":     [],
    }
