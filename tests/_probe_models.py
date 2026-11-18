"""Probe available models for OTPM limits — temporary test script."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)
import config, time
from groq import Groq

client = Groq(api_key=config.GROQ_API_KEY)
candidates = ["allam-2-7b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

msg = (
    "Extract facts from: 'Budget is Rs 650000'. "
    "Return ONLY valid JSON: "
    '{"primary_area":"budget","claims":[{"key":"total_budget","value":650000,'
    '"area":"budget","status":"confirmed","confidence":0.95,"rationale":"stated"}],'
    '"summary":"Budget is Rs 650000","message_type":"fact","message_meta":{"sender":"Test","date":"1 Jan"}}'
)

for m in candidates:
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": msg}],
            max_tokens=512,
            temperature=0,
        )
        content = r.choices[0].message.content or ""
        u = r.usage
        print(f"OK   {m:30s} pt={u.prompt_tokens:4d} ct={u.completion_tokens:4d} "
              f"{round(time.time()-t0,2):5.2f}s  {repr(content[:50])}")
    except Exception as e:
        print(f"FAIL {m:30s} {str(e)[:100]}")
