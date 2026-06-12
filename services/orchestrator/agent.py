"""Orchestrator Agent -- LangGraph banking workflow with automatic tracing.

Workflow: customer_lookup -> risk_assessment -> compliance_check

Trace chain:
  Orchestrator (LangGraph autolog)
    -> Customer Analyst (A2A + traceparent via traced_a2a_call)
       -> MongoDB MCP (MCP + traceparent)
          -> MongoDB query
    -> Risk Assessor (A2A + traceparent)
       -> Qwen3-8B LLM (openai autolog)
    -> Compliance Reviewer (A2A + traceparent)
       -> Qwen3-8B LLM (openai autolog)
"""

import json
import os
import sys
import uuid
import logging
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow
import mlflow.tracing
from langgraph.graph import StateGraph, START, END

from shared.mlflow_bootstrap import (
    traced_a2a_call,
    update_trace_session,
    get_openai_client,
    get_model_name,
    is_mock_mode,
)

logger = logging.getLogger(__name__)

CUSTOMER_ANALYST_URL = os.environ.get("CUSTOMER_ANALYST_URL", "http://customer-analyst:8001")
RISK_ASSESSOR_URL = os.environ.get("RISK_ASSESSOR_URL", "http://risk-assessor:8002")
COMPLIANCE_REVIEWER_URL = os.environ.get("COMPLIANCE_REVIEWER_URL", "http://compliance-reviewer:8003")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class BankingState(TypedDict, total=False):
    customer_id: str
    query: str
    session_id: str
    customer_data: dict
    risk_assessment: dict
    compliance_review: dict
    final_response: str
    status: str


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

MOCK_RISK_DATA = {
    "C-1001": {"risk_level": "low", "risk_score": 22, "debt_to_income_ratio": 18.5, "recommendation": "Low risk. Suitable for additional credit up to $200K."},
    "C-1006": {"risk_level": "high", "risk_score": 78, "debt_to_income_ratio": 67.4, "recommendation": "High risk. Delinquent business loan, high DTI ratio. Deny new credit."},
    "C-1004": {"risk_level": "medium", "risk_score": 52, "debt_to_income_ratio": 40.3, "recommendation": "Medium risk. Near DTI limit. Recommend small credit only with monitoring."},
}

MOCK_COMPLIANCE_DATA = {
    "C-1001": {"compliant": True, "compliance_score": 92, "checks": [{"check": "KYC", "status": "pass"}, {"check": "AML", "status": "pass"}, {"check": "DTI", "status": "pass"}], "recommendation": "Approved. All regulatory requirements met."},
    "C-1006": {"compliant": False, "compliance_score": 38, "checks": [{"check": "KYC", "status": "pass"}, {"check": "AML", "status": "warning"}, {"check": "DTI", "status": "fail"}], "flags": ["DTI exceeds 43% regulatory limit", "Delinquent loan L-2008"], "recommendation": "Denied. DTI ratio exceeds regulatory limit. Delinquent loan must be resolved."},
    "C-1004": {"compliant": True, "compliance_score": 71, "checks": [{"check": "KYC", "status": "pass"}, {"check": "AML", "status": "pass"}, {"check": "DTI", "status": "warning"}], "recommendation": "Conditional approval. DTI near limit; monitor closely."},
}


async def customer_lookup_node(state: BankingState) -> BankingState:
    """Call Customer Analyst to retrieve profile and loan history."""
    customer_id = state.get("customer_id", "C-1001")

    try:
        if is_mock_mode():
            result = _mock_customer_data(customer_id)
        else:
            result = await traced_a2a_call(
                CUSTOMER_ANALYST_URL,
                "get_customer_profile",
                {"customer_id": customer_id},
            )
        return {**state, "customer_data": result}
    except Exception as exc:
        logger.error("customer_lookup failed: %s", exc)
        return {**state, "customer_data": {}, "status": "failed"}


async def risk_assessment_node(state: BankingState) -> BankingState:
    """Call Risk Assessor with the customer data."""
    customer_id = state.get("customer_id", "C-1001")

    try:
        if is_mock_mode():
            result = MOCK_RISK_DATA.get(customer_id, MOCK_RISK_DATA["C-1001"])
        else:
            result = await traced_a2a_call(
                RISK_ASSESSOR_URL,
                "assess_risk",
                {"customer_data": state.get("customer_data", {})},
            )
        return {**state, "risk_assessment": result}
    except Exception as exc:
        logger.error("risk_assessment failed: %s", exc)
        return {**state, "risk_assessment": {}, "status": "failed"}


async def compliance_check_node(state: BankingState) -> BankingState:
    """Call Compliance Reviewer with customer data + risk assessment."""
    customer_id = state.get("customer_id", "C-1001")

    try:
        if is_mock_mode():
            result = MOCK_COMPLIANCE_DATA.get(customer_id, MOCK_COMPLIANCE_DATA["C-1001"])
        else:
            result = await traced_a2a_call(
                COMPLIANCE_REVIEWER_URL,
                "review_compliance",
                {
                    "assessment_data": {
                        "customer": state.get("customer_data", {}),
                        "risk": state.get("risk_assessment", {}),
                    }
                },
            )

        risk = state.get("risk_assessment", {})
        final = (
            f"Customer {customer_id} assessment complete.\n"
            f"Risk: {risk.get('risk_level', 'N/A')} "
            f"(score: {risk.get('risk_score', 'N/A')})\n"
            f"Compliance: {'Approved' if result.get('compliant') else 'Flagged'} "
            f"(score: {result.get('compliance_score', 'N/A')})"
        )
        return {**state, "compliance_review": result, "final_response": final, "status": "completed"}
    except Exception as exc:
        logger.error("compliance_check failed: %s", exc)
        return {**state, "compliance_review": {}, "status": "failed"}


def _check_failed(state: BankingState) -> str:
    return "end" if state.get("status") == "failed" else "continue"


# ---------------------------------------------------------------------------
# Compiled LangGraph workflow
# ---------------------------------------------------------------------------

def build_assessment_workflow():
    workflow = StateGraph(BankingState)
    workflow.add_node("customer_lookup", customer_lookup_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_edge(START, "customer_lookup")
    workflow.add_conditional_edges("customer_lookup", _check_failed, {"continue": "risk_assessment", "end": END})
    workflow.add_conditional_edges("risk_assessment", _check_failed, {"continue": "compliance_check", "end": END})
    workflow.add_edge("compliance_check", END)
    return workflow.compile()


_workflow = build_assessment_workflow()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class OrchestratorAgent:

    def __init__(self) -> None:
        self._conversation_history: dict[str, list[dict]] = {}

    async def handle_skill(self, skill: str, params: dict) -> dict:
        dispatch = {
            "assess_customer": self._skill_assess_customer,
            "chat": self._skill_chat,
        }
        handler = dispatch.get(skill, self._skill_assess_customer)
        return await handler(params)

    @mlflow.trace(name="orchestrator.assess_customer", span_type="AGENT")
    async def _skill_assess_customer(self, params: dict) -> dict:
        customer_id = params.get("customer_id", "C-1001")
        session_id = params.get("session_id", str(uuid.uuid4()))

        update_trace_session({
            "mlflow.trace.session": session_id,
            "workflow": "assess_customer",
            "customer_id": customer_id,
        })

        initial_state: BankingState = {
            "customer_id": customer_id,
            "session_id": session_id,
        }

        result = await _workflow.ainvoke(initial_state)

        return {
            "customer_id": customer_id,
            "customer_data": result.get("customer_data", {}),
            "risk_assessment": result.get("risk_assessment", {}),
            "compliance_review": result.get("compliance_review", {}),
            "final_response": result.get("final_response", ""),
            "session_id": session_id,
        }

    @mlflow.trace(name="orchestrator.chat", span_type="AGENT")
    async def _skill_chat(self, params: dict) -> dict:
        query = params.get("query", params.get("text", ""))
        session_id = params.get("session_id", str(uuid.uuid4()))

        with mlflow.tracing.context(session_id=session_id):
            update_trace_session({
                "mlflow.trace.session": session_id,
                "workflow": "chat",
            })

            history = self._conversation_history.setdefault(session_id, [])
            history.append({"role": "user", "content": query})

            if is_mock_mode():
                response = f"[Mock] Regarding your banking inquiry: {query}"
            else:
                client = get_openai_client()
                messages = [
                    {"role": "system", "content": (
                        "You are a helpful banking assistant. You help customers understand "
                        "their accounts, loans, and credit status. Be professional and concise."
                    )},
                    *history[-10:],
                ]
                completion = await client.chat.completions.create(
                    model=get_model_name(),
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                response = completion.choices[0].message.content or ""

            history.append({"role": "assistant", "content": response})

        return {"response": response, "session_id": session_id, "turn": len(history) // 2}


# ---------------------------------------------------------------------------
# Mock customer data (used by customer_lookup_node in mock mode)
# ---------------------------------------------------------------------------

_MOCK_CUSTOMERS = {
    "C-1001": {"name": "Alice Tan", "tier": "platinum", "credit_score": 780, "annual_income": 280000, "account_balance": 125000, "risk_category": "low"},
    "C-1006": {"name": "Frank Lee", "tier": "silver", "credit_score": 580, "annual_income": 95000, "account_balance": 8500, "risk_category": "high"},
    "C-1004": {"name": "David Lim", "tier": "silver", "credit_score": 650, "annual_income": 75000, "account_balance": 12000, "risk_category": "medium"},
}

_MOCK_LOANS = {
    "C-1001": [{"loan_id": "L-2001", "type": "mortgage", "outstanding": 420000, "status": "active"}, {"loan_id": "L-2002", "type": "auto", "outstanding": 12000, "status": "active"}],
    "C-1006": [{"loan_id": "L-2008", "type": "business", "outstanding": 185000, "status": "delinquent"}, {"loan_id": "L-2009", "type": "personal", "outstanding": 28000, "status": "active"}],
    "C-1004": [{"loan_id": "L-2005", "type": "personal", "outstanding": 14200, "status": "active"}, {"loan_id": "L-2006", "type": "credit_line", "outstanding": 8500, "status": "active"}],
}


def _mock_customer_data(customer_id: str) -> dict:
    profile = _MOCK_CUSTOMERS.get(customer_id, _MOCK_CUSTOMERS["C-1001"])
    loans = _MOCK_LOANS.get(customer_id, [])
    return {
        "customer_id": customer_id,
        "profile": profile,
        "loans": loans,
        "total_outstanding": sum(l.get("outstanding", 0) for l in loans),
    }
