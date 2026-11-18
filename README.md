# Mind to Machine — Hackathon Planning Agent

A message-driven event planning system for the Mind to Machine hackathon brief. It reads a plain-text pack, uses a local LLM to extract claims, applies deterministic conflict handling, calculates a plan, validates constraints, and exposes the plan and full model trace in Streamlit.

## Architecture

1. **Message reader / extractor (`agent.py`)** — sends every message through one identical Ollama chat call. No per-message flags or manual hints.
2. **Claim store (`planner.py`)** — stores normalized facts with date, sender, confidence and provenance. Newer conflicting claims replace older ones; older conflicting claims are rejected.
3. **Deterministic planner (`planner.py`)** — arithmetic, capacity, budget and judge calculations are Python, not extra LLM calls.
4. **Validation** — checks venue capacity, budget, food quantity, judging capacity, schedule and area coverage.
5. **UI (`app.py`)** — displays plan, checks, rejected claims and per-call trace; exports JSON.

## Run on Windows

### 1. Install Ollama and pull a model
Install Ollama from https://ollama.com/download, open PowerShell, then run:

```powershell
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

Wait for the model prompt, then type `/bye`. Keep Ollama running in the background.

### 2. Set up the project

```powershell
cd mind_to_machine
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch

```powershell
streamlit run app.py
```

The browser opens the dashboard. Click **Run full pack**. Add messages in the right-hand panel and click **Apply update**.

## One-command CLI run

```powershell
python main.py sample_messages.txt
```

The CLI saves `final_plan.json` and `full_trace.json`.

## Important model/trace honesty

The trace records the model identifier and token counters returned by Ollama (`prompt_eval_count`, `eval_count`). If a call fails, it is marked as failed with zero reported token counts; the program does not invent usage numbers. Model timing depends on laptop performance. Run a warm-up and full-pack timing test before submission.

## Add your own message pack

Use the exact message text in a `.txt` file. Separate messages with headers like:

```text
[Message 1 | From: Sender | Date: 18 Nov]
Message body...
```

The app's full-pack path and update path use the same message ingestion method. No message facts should be added to Python source code.

## Before submission

- Test with the sample pack.
- Test one genuine update (e.g. printing cost revised).
- Test an invalid food quantity (less food than attendees) and confirm it is rejected.
- Confirm every run is within the 2-minute full-run / 1-minute update limit.
- Inspect `git status`, commit, push, and submit the exact commit hash.
- Do not modify code after the stated code-freeze time.

## Known limitations to discuss honestly

This is a hackathon prototype. LLM extraction can still make mistakes on novel wording. The deterministic validator protects explicit invariants represented in its rules, but cannot guarantee every possible unseen constraint. Before final judging, add and test domain-specific invariants that the team can explain. Also ensure the judge's exact text-pack delimiter format is supported.
