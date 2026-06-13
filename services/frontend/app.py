"""Banking Operations Dashboard -- Streamlit frontend for the tracing demo.

Calls the Orchestrator agent via A2A protocol, generating MLflow traces
for every interaction.
"""

import asyncio
import json
import os
import sys
import uuid

import streamlit as st
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.mlflow_bootstrap import ensure_mlflow_initialized, traced_a2a_call

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8000")
MLFLOW_UI_URL = os.environ.get(
    "MLFLOW_UI_URL",
    "https://rh-ai.apps.cluster-9tjvr.9tjvr.sandbox2001.opentlc.com/mlflow",
)

CUSTOMERS = {
    "C-1001": {"name": "Alice Tan", "tier": "Platinum", "occupation": "CFO", "income": "$280,000", "credit_score": 780, "balance": "$125,000", "risk": "Low"},
    "C-1002": {"name": "Bob Chen", "tier": "Gold", "occupation": "Software Engineer", "income": "$120,000", "credit_score": 720, "balance": "$45,000", "risk": "Low"},
    "C-1003": {"name": "Carol Wong", "tier": "Platinum", "occupation": "Business Owner", "income": "$450,000", "credit_score": 810, "balance": "$380,000", "risk": "Low"},
    "C-1004": {"name": "David Lim", "tier": "Silver", "occupation": "Marketing Manager", "income": "$75,000", "credit_score": 650, "balance": "$12,000", "risk": "Medium"},
    "C-1006": {"name": "Frank Lee", "tier": "Silver", "occupation": "Restaurant Owner", "income": "$95,000", "credit_score": 580, "balance": "$8,500", "risk": "High"},
    "C-1007": {"name": "Grace Ho", "tier": "Diamond", "occupation": "Retired Banker", "income": "$350,000", "credit_score": 830, "balance": "$720,000", "risk": "Low"},
}

ensure_mlflow_initialized()

st.set_page_config(page_title="Banking Operations", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #6c63ff;
    }
    .check-pass { color: #28a745; }
    .check-fail { color: #dc3545; }
    .check-warning { color: #ffc107; }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Run async function from sync Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def call_orchestrator(skill: str, params: dict) -> dict:
    """Call the orchestrator via A2A."""
    return run_async(traced_a2a_call(ORCHESTRATOR_URL, skill, params))


# --- Header ---
col_title, col_link = st.columns([4, 1])
with col_title:
    st.title("Banking Operations Dashboard")
with col_link:
    st.markdown(f"[Open MLflow Traces]({MLFLOW_UI_URL}/#/experiments)")

st.divider()

# --- Sidebar: Customer Selection ---
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Customer")
    customer_id = st.selectbox(
        "Select Customer",
        options=list(CUSTOMERS.keys()),
        format_func=lambda x: f"{x} - {CUSTOMERS[x]['name']}",
    )

    customer = CUSTOMERS[customer_id]

    st.markdown(f"### {customer['name']}")
    st.markdown(f"**Tier:** {customer['tier']}")
    st.markdown(f"**Occupation:** {customer['occupation']}")
    st.markdown(f"**Annual Income:** {customer['income']}")
    st.markdown(f"**Credit Score:** {customer['credit_score']}")
    st.markdown(f"**Account Balance:** {customer['balance']}")

    risk_class = f"risk-{customer['risk'].lower()}"
    st.markdown(f"**Risk Category:** <span class='{risk_class}'>{customer['risk']}</span>", unsafe_allow_html=True)

    st.divider()

    if st.button("Run Full Assessment", type="primary", use_container_width=True):
        with st.spinner("Running assessment pipeline..."):
            result = call_orchestrator("assess_customer", {"customer_id": customer_id})
            st.session_state["assessment_result"] = result
            st.session_state["assessment_customer"] = customer_id

# --- Right Panel: Assessment Results ---
with right_col:
    st.subheader("Assessment Results")

    if "assessment_result" in st.session_state:
        result = st.session_state["assessment_result"]
        assessed_id = st.session_state.get("assessment_customer", "")

        if result.get("status") == "error":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.success(f"Assessment complete for {assessed_id}")

            # Customer data
            cust_data = result.get("customer_data", {})
            if cust_data and cust_data.get("profile"):
                profile = cust_data["profile"]
                loans = cust_data.get("loans", [])

                with st.expander("Customer Data Retrieved", expanded=False):
                    st.json(profile)
                    if loans:
                        st.markdown("**Loans:**")
                        for loan in loans:
                            status_icon = "🟢" if loan.get("status") == "active" else "🔴" if loan.get("status") == "delinquent" else "⚪"
                            st.markdown(f"{status_icon} **{loan.get('loan_id')}** - {loan.get('type', 'N/A')} | Outstanding: ${loan.get('outstanding', 0):,.0f} | Rate: {loan.get('interest_rate', 'N/A')}%")

            # Risk Assessment
            risk = result.get("risk_assessment", {})
            if risk:
                st.markdown("#### Risk Assessment")
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    level = risk.get("risk_level", "unknown")
                    color = {"low": "green", "medium": "orange", "high": "red"}.get(level, "gray")
                    st.metric("Risk Level", level.upper())
                with r_col2:
                    st.metric("Risk Score", f"{risk.get('risk_score', 'N/A')}/100")
                with r_col3:
                    st.metric("DTI Ratio", f"{risk.get('debt_to_income_ratio', 'N/A')}%")

                rec = risk.get("recommendation", "")
                if rec:
                    st.info(f"**Recommendation:** {rec}")

            # Compliance Review
            compliance = result.get("compliance_review", {})
            if compliance:
                st.markdown("#### Compliance Review")
                compliant = compliance.get("compliant")
                score = compliance.get("compliance_score", "N/A")

                if compliant is True:
                    st.success(f"Compliant (Score: {score}/100)")
                elif compliant is False:
                    st.error(f"Non-Compliant (Score: {score}/100)")
                else:
                    st.warning("Compliance status unknown")

                checks = compliance.get("checks", [])
                if checks:
                    for check in checks:
                        status = check.get("status", "unknown")
                        icon = {"pass": "✅", "fail": "❌", "warning": "⚠️"}.get(status, "❓")
                        st.markdown(f"{icon} **{check.get('check', 'N/A')}**: {check.get('detail', status)}")

                flags = compliance.get("flags", [])
                if flags:
                    st.markdown("**Flags:**")
                    for flag in flags:
                        st.markdown(f"🚩 {flag}")

                rec = compliance.get("recommendation", "")
                if rec:
                    st.info(f"**Compliance Recommendation:** {rec}")

            # Final response
            final = result.get("final_response", "")
            if final:
                st.markdown("---")
                st.markdown(f"**Summary:** {final}")

    else:
        st.info("Select a customer and click 'Run Full Assessment' to start the pipeline.")

# --- Bottom: Chat ---
st.divider()
st.subheader("Banking Assistant Chat")

if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = f"chat-{uuid.uuid4().hex[:8]}"
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a banking question..."):
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = call_orchestrator("chat", {
                "query": prompt,
                "session_id": st.session_state["chat_session_id"],
                "skill": "chat",
            })
            response = result.get("response", result.get("content", str(result)))

            # Strip thinking tags if present (Qwen3 outputs <think>...</think>)
            if "<think>" in response:
                import re
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            st.markdown(response)
            st.session_state["chat_messages"].append({"role": "assistant", "content": response})
