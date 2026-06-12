"""Compliance Reviewer Agent -- LLM-based regulatory compliance checks.

Traces: @mlflow.trace creates AGENT span, openai.autolog() captures the LLM call.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow

from shared.mlflow_bootstrap import get_openai_client, get_model_name, is_mock_mode

SYSTEM_PROMPT = """You are a banking compliance officer. Review the customer profile and risk assessment against regulatory requirements.

Check for:
1. KYC (Know Your Customer) completeness
2. AML (Anti-Money Laundering) red flags
3. Debt-to-income ratio regulatory limits (typically max 43%)
4. Concentration risk (single borrower exposure limits)
5. Internal policy adherence

Respond with a JSON object containing:
- "compliant": true/false
- "compliance_score": 1-100 (100 = fully compliant)
- "checks": list of {"check": name, "status": "pass"/"fail"/"warning", "detail": explanation}
- "flags": list of any regulatory flags or concerns
- "recommendation": final compliance recommendation"""

MOCK_REVIEW = {
    "compliant": True,
    "compliance_score": 92,
    "checks": [
        {"check": "KYC Verification", "status": "pass", "detail": "Customer identity verified, 12-year relationship history."},
        {"check": "AML Screening", "status": "pass", "detail": "No suspicious transaction patterns detected."},
        {"check": "Debt-to-Income Ratio", "status": "pass", "detail": "DTI at 18.5%, well below 43% regulatory limit."},
        {"check": "Concentration Risk", "status": "pass", "detail": "Total exposure within single-borrower limits."},
        {"check": "Internal Policy", "status": "warning", "detail": "Customer has two active loans; monitor for overextension."},
    ],
    "flags": [],
    "recommendation": "Approved. Customer meets all regulatory requirements. Minor note on multiple active facilities.",
}


class ComplianceReviewerAgent:

    @mlflow.trace(name="compliance_reviewer.review", span_type="AGENT")
    async def review_compliance(self, params: dict) -> dict:
        if is_mock_mode():
            return MOCK_REVIEW

        assessment_data = params.get("assessment_data", params)
        if isinstance(assessment_data, str):
            data_str = assessment_data
        else:
            data_str = json.dumps(assessment_data, indent=2, default=str)

        client = get_openai_client()
        response = await client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Review the following for regulatory compliance:\n\n{data_str}"},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_review": content, "compliant": None}
