import streamlit as st, json, os
from agent import OllamaAgent
from planner import build_plan

st.set_page_config(page_title="Mind to Machine | Event Planner", page_icon="🧠", layout="wide")
st.title("🧠 Mind to Machine")
st.caption("Multi-message agent system • Constraint-aware 4-hour hackathon planner")
with st.sidebar:
    st.header("Model settings")
    base = st.text_input("Ollama URL", os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
    model = st.text_input("Model", os.getenv("OLLAMA_MODEL","qwen2.5:3b"))
    st.caption("Local model = eligible route for the small-model bonus, subject to organizer rules.")
    if st.button("Reset session"):
        for k in ["state","trace","plan","agent"]: st.session_state.pop(k, None)
        st.rerun()

if "state" not in st.session_state:
    st.session_state.state = {"claims":{}, "rejected":[], "messages":[]}
    st.session_state.trace = []
    st.session_state.plan = None

c1,c2 = st.columns([3,2])
with c1:
    st.subheader("1. Load messages")
    uploaded = st.file_uploader("Upload a plain-text message pack", type=["txt"])
    default_text = ""
    if uploaded:
        default_text = uploaded.getvalue().decode("utf-8", errors="replace")
    else:
        try:
            with open("sample_messages.txt", encoding="utf-8") as f: default_text=f.read()
        except: pass
    text_input = st.text_area("Messages (one message per [Message ...] block)", value=default_text, height=270)
    if st.button("🚀 Run full pack", type="primary", use_container_width=True):
        messages = [x.strip() for x in __import__("re").split(r"(?=\[Message\s+\d+)", text_input) if x.strip()]
        if not messages: st.error("No messages found. Use [Message 1 ...] headers or upload a text file.")
        else:
            try:
                agent = OllamaAgent(base, model)
                state, plan = agent.run_messages(messages, st.session_state.state)
                st.session_state.state, st.session_state.plan = state, plan
                offset = len(st.session_state.trace)
                for item in agent.trace: item["call"] += offset
                st.session_state.trace = st.session_state.trace + agent.trace
                st.success(f"Processed {len(messages)} messages in {len(agent.trace)} model calls.")
            except Exception as e: st.error(str(e))
with c2:
    st.subheader("2. Add an update")
    new_message = st.text_area("Paste one new email / WhatsApp / notice", height=150, placeholder="Paste the new message exactly as received...")
    if st.button("Apply update", use_container_width=True):
        if not new_message.strip(): st.warning("Enter a message first.")
        else:
            try:
                agent = OllamaAgent(base, model)
                state, plan = agent.run_messages([new_message.strip()], st.session_state.state)
                st.session_state.state, st.session_state.plan = state, plan
                offset = len(st.session_state.trace)
                for item in agent.trace: item["call"] += offset
                st.session_state.trace += agent.trace
                st.success("Update processed. Plan recalculated.")
            except Exception as e: st.error(str(e))

if st.session_state.plan:
    p=st.session_state.plan
    st.divider(); st.header("3. Current event plan")
    a,b,c,d=st.columns(4)
    a.metric("Attendees",p["event"]["attendees"])
    b.metric("Teams",p["event"]["teams"])
    c.metric("Budget remaining",f'Rs {p["budget"]["remaining"]:,.0f}')
    d.metric("Plan status","VALID" if p["valid"] else "NEEDS ATTENTION")
    tabs=st.tabs(["Overview","Schedule","Budget & checks","Rejected messages","Model trace","Raw JSON"])
    with tabs[0]:
        x,y=st.columns(2)
        with x:
            st.subheader("Venue")
            st.write(p["venue"])
            st.subheader("Food")
            st.write(p["food"])
        with y:
            st.subheader("Judging")
            st.write(p["judging"])
            st.subheader("Event")
            st.write(p["event"])
    with tabs[1]:
        st.table(p["schedule"])
    with tabs[2]:
        st.write(p["budget"])
        for check in p["checks"]:
            (st.success if check["passed"] else st.error)(f'{ "PASS" if check["passed"] else "FAIL" } — {check["name"]}: {check["detail"]}')
    with tabs[3]:
        if p_rej:=st.session_state.state.get("rejected"):
            for i,item in enumerate(p_rej,1):
                with st.expander(f"Rejected item {i}: {item.get('key','claim')}"):
                    st.write(item.get("reason"))
                    st.code(item.get("message",""))
        else: st.info("No messages have been rejected.")
    with tabs[4]:
        st.caption("Token values are reported by Ollama. Failed calls are explicitly marked and never represented as successful token counts.")
        for entry in st.session_state.trace:
            with st.expander(f'Call #{entry["call"]} • {entry["model"]} • {entry["area"]} • {entry["total_tokens"]} tokens'):
                st.json(entry)
        totals={"calls":len(st.session_state.trace),
                "prompt_tokens":sum(x["prompt_tokens"] for x in st.session_state.trace),
                "completion_tokens":sum(x["completion_tokens"] for x in st.session_state.trace)}
        totals["total_tokens"]=totals["prompt_tokens"]+totals["completion_tokens"]
        st.write("TOTAL TOKENS",totals)
    with tabs[5]:
        st.json(p)
    col1,col2=st.columns(2)
    with col1:
        st.download_button("⬇️ Download final plan",json.dumps(p,indent=2),file_name="final_plan.json",mime="application/json",use_container_width=True)
    with col2:
        st.download_button("⬇️ Download full trace",json.dumps(st.session_state.trace,indent=2),file_name="full_trace.json",mime="application/json",use_container_width=True)
else:
    st.info("Load the sample pack or upload a new pack, then run the agent. The plan and trace will appear here.")
