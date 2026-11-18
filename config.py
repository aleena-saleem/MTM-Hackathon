"""
config.py — Centralised configuration for the MTM Hackathon project.

All credentials and model settings are read from environment variables.
Never put API keys or secrets directly in this file.

Required environment variables (Groq — recommended)
-----------------------------------------------------
GROQ_API_KEY      Groq API key            (get one at console.groq.com/keys)
GROQ_MODEL        Groq model name         (default: llama-3.1-8b-instant)

Optional — Ollama (local, no key required)
------------------------------------------
OLLAMA_BASE_URL   URL of the running Ollama instance (default: http://localhost:11434)
OLLAMA_MODEL      Model tag               (default: qwen2.5:3b)

Optional — OpenAI
------------------
OPENAI_API_KEY    OpenAI API key
OPENAI_MODEL      OpenAI model name       (default: gpt-4o-mini)

Optional — Anthropic
---------------------
ANTHROPIC_API_KEY Anthropic API key
ANTHROPIC_MODEL   Anthropic model name    (default: claude-3-haiku-20240307)

Backend selection precedence (auto-detected unless LLM_BACKEND is set explicitly)
-----------------------------------------------------------------------------------
  groq > openai > anthropic > ollama

Override by setting:  LLM_BACKEND=groq | openai | anthropic | ollama

How to configure on Windows (PowerShell, current session)
----------------------------------------------------------
$env:GROQ_API_KEY = "gsk_..."
$env:GROQ_MODEL   = "llama-3.1-8b-instant"

Or add them to a .env file in the project root (never commit that file).
"""

import os
from dotenv import load_dotenv

# Load a .env file if present. Safe to call even if the file does not exist.
load_dotenv()

# ---------------------------------------------------------------------------
# Groq (cloud — requires GROQ_API_KEY)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str          = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

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
# Precedence: groq > openai > anthropic > ollama
# Override by setting LLM_BACKEND explicitly.
# ---------------------------------------------------------------------------
def _detect_backend() -> str:
    explicit = os.getenv("LLM_BACKEND", "").lower().strip()
    if explicit:
        return explicit
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "ollama"

LLM_BACKEND: str = _detect_backend()


# ---------------------------------------------------------------------------
# Convenience: active model name and endpoint (for display / logging)
# ---------------------------------------------------------------------------
def active_model_info() -> dict:
    """Return the name and endpoint of the currently configured model."""
    if LLM_BACKEND == "groq":
        return {"backend": "groq", "model": GROQ_MODEL, "base_url": "https://api.groq.com/openai/v1"}
    if LLM_BACKEND == "openai":
        return {"backend": "openai", "model": OPENAI_MODEL, "base_url": "https://api.openai.com/v1"}
    if LLM_BACKEND == "anthropic":
        return {"backend": "anthropic", "model": ANTHROPIC_MODEL, "base_url": "https://api.anthropic.com"}
    return {"backend": "ollama", "model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL}


# ---------------------------------------------------------------------------
# Validation — surfaces missing credentials early
# ---------------------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of human-readable warnings about missing configuration."""
    warnings: list[str] = []
    if LLM_BACKEND == "groq" and not GROQ_API_KEY:
        warnings.append("LLM_BACKEND=groq but GROQ_API_KEY is not set.")
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_BACKEND=openai but OPENAI_API_KEY is not set.")
    if LLM_BACKEND == "anthropic" and not ANTHROPIC_API_KEY:
        warnings.append("LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set.")
    return warnings
