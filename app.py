"""
Bonus: Streamlit interface for the Nextdot AI Agent pipeline.
Run with:  streamlit run app.py
"""

import json
import streamlit as st
from agent import run_pipeline

st.set_page_config(
    page_title="Nextdot AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Nextdot — AI Support Agent")
st.caption("Mini AI agent that classifies, extracts, replies, and explains its reasoning.")

# ---- Sidebar: sample inputs ------------------------------------------------
SAMPLES = {
    "Input A – Angry Customer": (
        "i ordered the premium plan 3 weeks ago and STILL havent received any confirmation email or "
        "access. this is absolutely ridiculous. i paid Rs 4999 and nobody has responded to my 4 emails. "
        "if this isnt fixed today im disputing the charge with my bank. my name is Rohan Mehta and my "
        "order id is ORD-8821."
    ),
    "Input B – Confused Query": (
        "hi I saw your AI tool on linkedin and I am curious how it works for small businesses? like do "
        "you guys offer any free trial or something? also can it work with whatsapp? just exploring "
        "options for now nothing urgent."
    ),
    "Input C – Positive Feedback": (
        "Just wanted to say the onboarding session last Tuesday was really well done. Priya from your "
        "team was super helpful and patient. The tool is working great so far. Looking forward to the "
        "advanced features you mentioned. Keep it up!"
    ),
}

with st.sidebar:
    st.header("📋 Load Sample")
    sample_choice = st.selectbox("Choose a sample input", ["— none —"] + list(SAMPLES.keys()))

# ---- Main input ------------------------------------------------------------
default_text = SAMPLES.get(sample_choice, "") if sample_choice != "— none —" else ""
message = st.text_area(
    "Customer Message",
    value=default_text,
    height=160,
    placeholder="Paste or type a customer message here…",
)

if st.button("▶  Run Pipeline", type="primary") and message.strip():
    with st.spinner("Analysing message…"):
        result = run_pipeline(message)

    if "error" in result:
        st.error(f"Pipeline error: {result['error']}")
        st.code(result.get("raw_response", ""), language="text")
    else:
        clf  = result["classification"]
        ext  = result["extraction"]
        reply     = result["reply"]
        reasoning = result["reasoning"]

        # Row 1: classification + extraction
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Classification")
            st.metric("Intent",    clf["intent"].capitalize())
            st.metric("Sentiment", clf["sentiment"].capitalize())

        with col2:
            st.subheader("📦 Extraction")
            st.write(f"**Customer Name:** {ext['customer_name'] or '—'}")
            st.write(f"**Issue Type:** {ext['issue_type']}")
            urgency_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
            icon = urgency_colors.get(ext["urgency_level"], "⚪")
            st.write(f"**Urgency:** {icon} {ext['urgency_level']}")
            st.write(f"**Recommended Action:** {ext['recommended_action']}")

        # Row 2: reply
        st.subheader("✉️ Generated Reply")
        st.info(reply)

        # Row 3: reasoning
        st.subheader("🧠 Model Reasoning")
        with st.expander("Why this classification?"):
            st.write(reasoning["classification_rationale"])
        with st.expander("Why this tone?"):
            st.write(reasoning["tone_rationale"])
        with st.expander("How were fields extracted?"):
            st.write(reasoning["extraction_notes"])

        # Raw JSON toggle
        with st.expander("🗂 Raw JSON output"):
            st.json(result)
elif st.button("▶  Run Pipeline", type="primary"):
    st.warning("Please enter a customer message first.")
