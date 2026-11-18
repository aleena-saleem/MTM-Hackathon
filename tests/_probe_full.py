"""Test full extraction prompt on all candidates to find highest ct before 429."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)
import config
from groq import Groq
from agents.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE

client = Groq(api_key=config.GROQ_API_KEY)

# Use the longest realistic message from sample_messages.txt
TEST_MSG = (
    "[Message 2 | From: Margalla Event Halls, email | Date: 3 Nov]\n"
    "Room D seats 80 at Rs 40,000. Hall A seats 200 at Rs 120,000. "
    "Hall B seats 200 at Rs 150,000. Hall C seats 300 at Rs 220,000. "
    "Halls B and C have a standby generator."
)

candidates = ["allam-2-7b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

for m in candidates:
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user",   "content": EXTRACTION_USER_TEMPLATE.format(message=TEST_MSG)},
            ],
            max_tokens=1024,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content or ""
        u = r.usage
        dur = round(time.time() - t0, 2)
        print(f"OK   {m:30s} pt={u.prompt_tokens:4d} ct={u.completion_tokens:4d} total={u.total_tokens:4d} {dur}s")
        # Check JSON parses
        parsed = json.loads(content)
        print(f"     claims: {len(parsed.get('claims',[]))}  area: {parsed.get('primary_area')}")
    except Exception as e:
        print(f"FAIL {m:30s} {str(e)[:140]}")
