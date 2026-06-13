"""Banking Operations Dashboard -- Streamlit frontend for the tracing demo."""

import asyncio
import json
import os
import re
import sys
import uuid

import streamlit as st
import mlflow
from mlflow.genai import make_judge
from mlflow.genai.scorers import scorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.mlflow_bootstrap import ensure_mlflow_initialized, traced_a2a_call, get_model_name

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
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def call_orchestrator(skill: str, params: dict) -> dict:
    return run_async(traced_a2a_call(ORCHESTRATOR_URL, skill, params))


def strip_thinking(text: str) -> str:
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return text


def _judge_base_url() -> str:
    base = os.environ.get(
        "OPENAI_BASE_URL",
        "https://qwen3-8b-fp8-dynamic-no-maas-0-test.apps.cluster-9tjvr.9tjvr.sandbox2001.opentlc.com/v1",
    )
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if base.endswith("/v1/"):
        return base + "chat/completions"
    return base


def run_evaluators():
    """Run MLflow evaluators on recent traces. Results attach as assessments in MLflow UI."""
    exp = mlflow.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT_NAME", "banking-demo"))
    if not exp:
        return None, "Experiment not found"

    traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=10)
    if traces.empty:
        return None, "No traces found"

    model_uri = f"openai:/{get_model_name()}"
    base_url = _judge_base_url()

    relevance_judge = make_judge(
        name="relevance_to_query",
        model=model_uri,
        base_url=base_url,
        instructions=(
            "Evaluate whether the agent's response is relevant to the user's query.\n\n"
            "Query: {{ inputs }}\nResponse: {{ outputs }}\n\n"
            "Answer 'yes' if relevant, 'no' if not."
        ),
    )

    compliance_judge = make_judge(
        name="banking_compliance",
        model=model_uri,
        base_url=base_url,
        instructions=(
            "Evaluate whether the response adheres to banking guidelines:\n"
            "1. Must not provide personal financial advice\n"
            "2. Should reference data points in risk assessments\n"
            "3. Must be professional\n"
            "4. Risk levels must be stated as low/medium/high\n\n"
            "Response: {{ outputs }}\n\nAnswer 'pass' or 'fail' with rationale."
        ),
    )

    @scorer
    def latency_check(inputs, outputs, trace):
        from mlflow.entities import Feedback
        if trace and trace.info and trace.info.execution_time_ms is not None:
            ok = trace.info.execution_time_ms < 30_000
            return Feedback(
                name="latency_sla",
                value="pass" if ok else "fail",
                rationale=f"Took {trace.info.execution_time_ms}ms ({'within' if ok else 'exceeds'} 30s SLA).",
            )
        return Feedback(name="latency_sla", value="unknown", rationale="No timing data.")

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[relevance_judge, compliance_judge, latency_check],
    )

    return results, None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_link = st.columns([4, 1])
with col_title:
    st.title("Banking Operations Dashboard")
with col_link:
    st.markdown(f"[Open MLflow Traces]({MLFLOW_UI_URL}/#/experiments)")

st.divider()

# ---------------------------------------------------------------------------
# Customer Selection + Assessment Results
# ---------------------------------------------------------------------------

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
            st.rerun()

    st.divider()

    if st.button("Run Evaluators on Traces", use_container_width=True):
        with st.spinner("Running LLM judges on recent traces (this takes ~20s)..."):
            eval_results, err = run_evaluators()
            if err:
                st.session_state["eval_error"] = err
            else:
                st.session_state["eval_results"] = eval_results
                st.session_state.pop("eval_error", None)
            st.rerun()

with right_col:
    st.subheader("Assessment Results")

    if "assessment_result" in st.session_state:
        result = st.session_state["assessment_result"]
        assessed_id = st.session_state.get("assessment_customer", "")

        if result.get("status") == "error":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.success(f"Assessment complete for {assessed_id}")

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

                level = risk.get("risk_level", "unknown")
                if level == "unknown" and risk.get("raw_assessment"):
                    level = "see details"
                score = risk.get("risk_score", "N/A")
                dti = risk.get("debt_to_income_ratio", "N/A")

                with r_col1:
                    st.metric("Risk Level", str(level).upper())
                with r_col2:
                    st.metric("Risk Score", f"{score}/100")
                with r_col3:
                    st.metric("DTI Ratio", f"{dti}%")

                key_factors = risk.get("key_factors", [])
                if key_factors:
                    st.markdown("**Key Factors:**")
                    for f in key_factors[:5]:
                        st.markdown(f"- {f}")

                rec = risk.get("recommendation", "")
                if rec:
                    st.info(f"**Recommendation:** {rec}")

                if risk.get("raw_assessment"):
                    with st.expander("Raw LLM Assessment", expanded=False):
                        st.code(strip_thinking(risk["raw_assessment"])[:2000])

            # Compliance Review
            compliance = result.get("compliance_review", {})
            if compliance:
                st.markdown("#### Compliance Review")
                compliant = compliance.get("compliant")
                comp_score = compliance.get("compliance_score", "N/A")

                if compliant is True:
                    st.success(f"Compliant (Score: {comp_score}/100)")
                elif compliant is False:
                    st.error(f"Non-Compliant (Score: {comp_score}/100)")
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
                    for flag in flags:
                        st.markdown(f"🚩 {flag}")

                rec = compliance.get("recommendation", "")
                if rec:
                    st.info(f"**Compliance Recommendation:** {rec}")

                if compliance.get("raw_review"):
                    with st.expander("Raw LLM Review", expanded=False):
                        st.code(strip_thinking(compliance["raw_review"])[:2000])

            final = result.get("final_response", "")
            if final:
                st.markdown("---")
                st.markdown(f"**Summary:** {final}")

    else:
        st.info("Select a customer and click 'Run Full Assessment' to start the pipeline.")

    # Evaluator results
    if "eval_error" in st.session_state:
        st.error(f"Evaluator error: {st.session_state['eval_error']}")
    elif "eval_results" in st.session_state:
        st.markdown("---")
        st.markdown("#### Evaluation Results (LLM Judges)")
        st.caption("Assessments attached to traces in MLflow UI → Show assessments")
        metrics = st.session_state["eval_results"].metrics
        if metrics:
            for k, v in metrics.items():
                st.markdown(f"- **{k}**: {v}")
        else:
            st.markdown("Scorers ran but no aggregate metrics returned. Check individual trace assessments in MLflow.")
        st.markdown(f"[View evaluation run in MLflow]({MLFLOW_UI_URL}/#/experiments)")


# ---------------------------------------------------------------------------
# Chat (toggle panel at bottom)
# ---------------------------------------------------------------------------

st.markdown("---")

if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = f"chat-{uuid.uuid4().hex[:8]}"
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

_, chat_toggle_col = st.columns([5, 1])
with chat_toggle_col:
    if st.button("💬 Chat" if not st.session_state["chat_open"] else "✕ Close", use_container_width=True):
        st.session_state["chat_open"] = not st.session_state["chat_open"]
        st.rerun()

if st.session_state["chat_open"]:
    st.subheader("Banking Assistant")
    st.caption("Ask questions about the customer or assessment results")

    chat_container = st.container(height=400)

    with chat_container:
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a banking question...", key="chat_input"):
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

        context_parts = []
        if "assessment_result" in st.session_state:
            r = st.session_state["assessment_result"]
            cid = st.session_state.get("assessment_customer", "")
            context_parts.append(
                f"Context: Latest assessment for {cid}: "
                f"Risk={r.get('risk_assessment', {}).get('risk_level', 'N/A')} "
                f"(score={r.get('risk_assessment', {}).get('risk_score', 'N/A')}), "
                f"Compliant={r.get('compliance_review', {}).get('compliant', 'N/A')} "
                f"(score={r.get('compliance_review', {}).get('compliance_score', 'N/A')}). "
                f"Customer: {json.dumps(r.get('customer_data', {}).get('profile', {}), default=str)[:300]}"
            )

        query = f"{prompt}\n\n{''.join(context_parts)}" if context_parts else prompt

        chat_result = call_orchestrator("chat", {
            "query": query,
            "session_id": st.session_state["chat_session_id"],
            "skill": "chat",
        })
        response = chat_result.get("response", chat_result.get("content", str(chat_result)))
        response = strip_thinking(response)
        st.session_state["chat_messages"].append({"role": "assistant", "content": response})
        st.rerun()

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state["chat_messages"] = []
            st.session_state["chat_session_id"] = f"chat-{uuid.uuid4().hex[:8]}"
            st.rerun()
