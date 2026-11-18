import argparse, json, sys
from agent import OllamaAgent

def load_messages(path):
    text=open(path,encoding="utf-8").read()
    import re
    parts=[x.strip() for x in re.split(r"(?=\[Message\s+\d+)",text) if x.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def main():
    ap=argparse.ArgumentParser(description="Mind to Machine event planning agent")
    ap.add_argument("file", help="Path to message pack .txt")
    ap.add_argument("--url",default="http://localhost:11434")
    ap.add_argument("--model",default="qwen2.5:3b")
    args=ap.parse_args()
    try:
        agent=OllamaAgent(args.url,args.model)
        state,plan=agent.run_messages(load_messages(args.file))
        trace=agent.trace
        with open("final_plan.json","w",encoding="utf-8") as f: json.dump(plan,f,indent=2,ensure_ascii=False)
        with open("full_trace.json","w",encoding="utf-8") as f: json.dump({"calls":trace,"totals":agent.trace_summary(),"rejected":state["rejected"]},f,indent=2,ensure_ascii=False)
        print(json.dumps(plan,indent=2,ensure_ascii=False))
        print("\\n===== MODEL TRACE =====")
        for entry in trace:
            print(json.dumps(entry, indent=2, ensure_ascii=False))
        print("\\nTRACE TOTALS:",json.dumps(agent.trace_summary()))
        print("\\nSaved final_plan.json and full_trace.json")
        sys.exit(0 if plan["valid"] else 2)
    except Exception as e:
        print("ERROR:",e,file=sys.stderr); sys.exit(1)
if __name__=="__main__": main()
