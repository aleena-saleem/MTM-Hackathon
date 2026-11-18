"""
Probe openai/gpt-oss-120b on all 8 sample messages with the CURRENT system prompt.
Shows prompt tokens, completion tokens, claims extracted, and any 429s.
"""
import sys, os, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)
import config
from groq import Groq
from agents.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE

client = Groq(api_key=config.GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"

raw = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sample_messages.txt"), encoding="utf-8").read()
messages = [x.strip() for x in re.split(r"(?=\[Message\s+\d+)", raw) if x.strip()]

total_pt = total_ct = 0
print(f"Testing {len(messages)} messages on {MODEL}\n")

for i, msg in enumerate(messages, 1):
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user",   "content": EXTRACTION_USER_TEMPLATE.format(message=msg)},
            ],
            max_tokens=800,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content or ""
        u = r.usage
        dur = round(time.time()-t0, 2)
        try:
            parsed = json.loads(content)
            n_claims = len(parsed.get("claims", []))
            area = parsed.get("primary_area", "?")
        except Exception:
            n_claims = -1
            area = "PARSE_ERR"
        total_pt += u.prompt_tokens
        total_ct += u.completion_tokens
        print(f"  [{i}] pt={u.prompt_tokens:4d} ct={u.completion_tokens:4d} "
              f"claims={n_claims:2d} area={area:10s} {dur}s")
        time.sleep(0.5)   # small gap between calls
    except Exception as e:
        print(f"  [{i}] ERROR: {str(e)[:120]}")

print(f"\nTOTAL: pt={total_pt} ct={total_ct} combined={total_pt+total_ct}")
print(f"Avg per call: pt={total_pt//len(messages)} ct={total_ct//len(messages)}")
