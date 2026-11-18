"""
Single real Groq extraction test.
One API call only. Reports model, timing, tokens, and extracted claims.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)

import config
from agent import GroqAgent

MSG = (
    "[Message 1 | From: Dean | Date: 9 Nov]\n"
    "Approved: a 4-hour hackathon for 150 students in teams of 3. "
    "The budget is Rs 650,000, all in. Ten staff will be on duty. "
    "Please keep a 15-minute prayer break in the schedule."
)

print("=" * 60)
print("SINGLE GROQ EXTRACTION TEST")
print("=" * 60)
print(f"Backend   : {config.LLM_BACKEND}")
print(f"Model cfg : {config.GROQ_MODEL}")
print(f"Message   : {MSG[:80]}...")
print()

agent = GroqAgent()
t0 = time.time()
extraction = agent.extract(MSG, index=1)
wall_time = round(time.time() - t0, 3)

entry = agent.trace[0]

print(f"1. Model used        : {entry['model']}")
print(f"2. Time taken        : {entry['duration_seconds']}s  (wall: {wall_time}s)")
print(f"3. Prompt tokens     : {entry['prompt_tokens']}")
print(f"4. Completion tokens : {entry['completion_tokens']}")
print(f"5. Total tokens      : {entry['total_tokens']}")
print(f"6. Status            : {entry['status']}")
print()

claims = extraction.get("claims", [])
print(f"Extracted claims ({len(claims)} total):")
for c in claims:
    print(f"  key={c.get('key'):35s} value={str(c.get('value')):20s} "
          f"area={c.get('area'):10s} status={c.get('status'):12s} conf={c.get('confidence')}")

print()
print(f"primary_area  : {extraction.get('primary_area')}")
print(f"message_type  : {extraction.get('message_type')}")
print(f"summary       : {extraction.get('summary')}")
print()
print("SUCCESS" if entry['status'] == 'success' and len(claims) > 0 else "PARTIAL / FAILED")
