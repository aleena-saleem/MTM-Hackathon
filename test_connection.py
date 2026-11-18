"""
test_connection.py — Verify that the configured LLM backend is reachable.

Run:  python test_connection.py

Prints the model name, response text, and actual token usage as reported
by the API. Never fabricates token counts — if a call fails, it says so.
"""

import json, sys, time
import requests
from config import LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_MODEL, \
                   OPENAI_API_KEY, OPENAI_MODEL, \
                   ANTHROPIC_API_KEY, ANTHROPIC_MODEL, \
                   active_model_info, validate

TEST_PROMPT = "Reply with exactly three words: connection test successful."


def test_ollama() -> dict:
    url = OLLAMA_BASE_URL + "/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "options": {"temperature": 0},
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return {
        "backend": "ollama",
        "model": data.get("model", OLLAMA_MODEL),
        "response": data.get("message", {}).get("content", "").strip(),
        "prompt_tokens": int(data.get("prompt_eval_count") or 0),
        "completion_tokens": int(data.get("eval_count") or 0),
        "total_tokens": int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0),
        "duration_seconds": round(time.time() - t0, 2),
    }


def test_openai() -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0,
        "max_tokens": 20,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return {
        "backend": "openai",
        "model": data.get("model", OPENAI_MODEL),
        "response": data["choices"][0]["message"]["content"].strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "duration_seconds": round(time.time() - t0, 2),
    }


def test_anthropic() -> dict:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return {
        "backend": "anthropic",
        "model": data.get("model", ANTHROPIC_MODEL),
        "response": data["content"][0]["text"].strip(),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "duration_seconds": round(time.time() - t0, 2),
    }


def main():
    info = active_model_info()
    warnings = validate()
    if warnings:
        for w in warnings:
            print(f"[CONFIG WARNING] {w}")

    print(f"\nTesting backend : {info['backend']}")
    print(f"Model           : {info['model']}")
    print(f"Endpoint        : {info['base_url']}")
    print(f"Prompt sent     : \"{TEST_PROMPT}\"\n")

    try:
        if LLM_BACKEND == "openai":
            result = test_openai()
        elif LLM_BACKEND == "anthropic":
            result = test_anthropic()
        else:
            result = test_ollama()

        print("SUCCESS")
        print(f"  Model returned  : {result['model']}")
        print(f"  Response        : {result['response']}")
        print(f"  Prompt tokens   : {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Total tokens    : {result['total_tokens']}")
        print(f"  Duration        : {result['duration_seconds']}s")
        print()
        print(json.dumps(result, indent=2))

    except requests.exceptions.ConnectionError as e:
        print(f"\nERROR: Could not connect to {info['base_url']}")
        if info["backend"] == "ollama":
            print("  → Make sure Ollama is running:  ollama serve")
            print(f"  → Make sure the model is pulled: ollama pull {info['model']}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: HTTP {e.response.status_code} from {info['backend']}: {e.response.text[:300]}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
