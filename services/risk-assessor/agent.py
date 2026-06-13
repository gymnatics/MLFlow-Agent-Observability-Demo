"""Risk Assessor Agent -- LLM-based credit risk analysis.

Traces: @mlflow.trace creates AGENT span, openai.autolog() captures the LLM call.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow

from shared.mlflow_bootstrap import get_openai_client, get_model_name, is_mock_mode

SYSTEM_PROMPT = """You are a senior credit risk analyst at a bank. Given customer profile data and loan history, provide a structured risk assessment.

Respond with a JSON object containing:
- "risk_level": "low", "medium", or "high"
- "risk_score": 1-100 (100 = highest risk)
- "debt_to_income_ratio": calculated percentage
- "key_factors": list of 3-5 risk factors identified
- "recommendation": brief recommendation for the loan committee
- "max_additional_credit": estimated maximum additional credit in dollars"""

MOCK_ASSESSMENT = {
    "risk_level": "low",
    "risk_score": 22,
    "debt_to_income_ratio": 18.5,
    "key_factors": [
        "Strong credit score (780)",
        "Low debt-to-income ratio",
        "Long banking relationship (12 years)",
        "Stable employment as CFO",
        "Consistent payment history on existing loans",
    ],
    "recommendation": "Customer presents low credit risk. Suitable for additional credit facilities up to $200K.",
    "max_additional_credit": 200000,
}


class RiskAssessorAgent:

    @mlflow.trace(name="risk_assessor.assess", span_type="AGENT")
    async def assess_risk(self, params: dict) -> dict:
        if is_mock_mode():
            return MOCK_ASSESSMENT

        customer_data = params.get("customer_data", params)
        if isinstance(customer_data, str):
            data_str = customer_data
        else:
            data_str = json.dumps(customer_data, indent=2, default=str)

        client = get_openai_client()
        response = await client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze the credit risk for this customer:\n\n{data_str}"},
            ],
            temperature=0.2,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_assessment": content, "risk_level": "unknown"}
