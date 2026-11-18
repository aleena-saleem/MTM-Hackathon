"""
config.py — Centralised configuration for the MTM Hackathon project.

All credentials and model settings are read from environment variables.
Never put API keys or secrets directly in this file.

Required environment variables
--------------------------------
OLLAMA_BASE_URL   URL of the running Ollama instance (default: http://localhost:11434)
OLLAMA_MODEL      Model tag to use              (default: qwen2.5:3b)

Optional — only needed if you switch to a cloud LLM provider
-------------------------------------------------------------
OPENAI_API_KEY    OpenAI API key
OPENAI_MODEL      OpenAI model name             (default: gpt-4o-mini)

ANTHROPIC_API_KEY Anthropic API key
ANTHROPIC_MODEL   Anthropic model name          (default: claude-3-haiku-20240307)

How to set variables on Windows (PowerShell, current session only)
-------------------------------------------------------------------
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL    = "qwen2.5:3b"

Or add them to a .env file and call load_dotenv() before importing this module.
"""

import os
from dotenv import load_dotenv

# Load a .env file if present. Safe to call even if the file does not exist.
load_dotenv()


# ---------------------------------------------------------------------------
# Ollama (local model — no API key required)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ---------------------------------------------------------------------------
# OpenAI (cloud — requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Anthropic (cloud — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str          = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

# ---------------------------------------------------------------------------
# Active backend selection
# Precedence: if OPENAI_API_KEY is set → openai
#             elif ANTHROPIC_API_KEY is set → anthropic
#             else → ollama (local, no key needed)
# Override by setting LLM_BACKEND explicitly.
# ---------------------------------------------------------------------------
def _detect_backend() -> str:
    if os.getenv("LLM_BACKEND"):
        return os.getenv("LLM_BACKEND", "ollama").lower()
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "ollama"

LLM_BACKEND: str = _detect_backend()


# ---------------------------------------------------------------------------
# Convenience: return the active model name and base URL (for display/logging)
# ---------------------------------------------------------------------------
def active_model_info() -> dict:
    """Return the name and endpoint of the currently configured model."""
    if LLM_BACKEND == "openai":
        return {"backend": "openai", "model": OPENAI_MODEL, "base_url": "https://api.openai.com/v1"}
    if LLM_BACKEND == "anthropic":
        return {"backend": "anthropic", "model": ANTHROPIC_MODEL, "base_url": "https://api.anthropic.com"}
    return {"backend": "ollama", "model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL}


# ---------------------------------------------------------------------------
# Validation helper — called at startup to surface missing credentials early
# ---------------------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of human-readable warnings about missing configuration."""
    warnings: list[str] = []
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_BACKEND=openai but OPENAI_API_KEY is not set.")
    if LLM_BACKEND == "anthropic" and not ANTHROPIC_API_KEY:
        warnings.append("LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set.")
    return warnings
