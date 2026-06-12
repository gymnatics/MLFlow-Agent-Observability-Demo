"""Evaluation script -- runs scorers on banking traces (Act 4).

Usage:
  python evaluate.py                       # evaluate latest traces
  python evaluate.py --session SESSION_ID  # evaluate a specific session
"""

import argparse
import logging
import os

import mlflow
from mlflow.genai import make_judge
from mlflow.genai.scorers import scorer

from shared.mlflow_bootstrap import ensure_mlflow_initialized, get_model_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _get_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return exp.experiment_id


def _judge_base_url() -> str:
    """Build the full chat completions URL for the vLLM endpoint."""
    base = os.environ.get(
        "OPENAI_BASE_URL",
        "https://qwen3-8b-fp8-dynamic-no-maas-0-test.apps.cluster-9tjvr.9tjvr.sandbox2001.opentlc.com/v1",
    )
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if base.endswith("/v1/"):
        return base + "chat/completions"
    return base


def _judge_model_uri() -> str:
    return f"openai:/{get_model_name()}"


# ---------------------------------------------------------------------------
# LLM-based judges (using make_judge with base_url for vLLM)
# ---------------------------------------------------------------------------

def _build_relevance_judge():
    return make_judge(
        name="relevance_to_query",
        model=_judge_model_uri(),
        base_url=_judge_base_url(),
        instructions=(
            "Evaluate whether the agent's response is relevant to the user's query.\n\n"
            "Query: {{ inputs }}\n"
            "Response: {{ outputs }}\n\n"
            "Answer 'yes' if the response addresses the query, 'no' if it does not."
        ),
    )


def _build_compliance_judge():
    return make_judge(
        name="banking_compliance",
        model=_judge_model_uri(),
        base_url=_judge_base_url(),
        instructions=(
            "You are a banking compliance reviewer. Evaluate whether the response "
            "adheres to these guidelines:\n"
            "1. Must not provide personal financial advice\n"
            "2. Should reference specific data points in risk assessments\n"
            "3. Must be professional and factually grounded\n"
            "4. Risk levels must be clearly stated as low, medium, or high\n\n"
            "Response to evaluate: {{ outputs }}\n\n"
            "Answer 'pass' if compliant, 'fail' if not, with rationale."
        ),
    )


def _build_conversation_quality_judge():
    return make_judge(
        name="conversation_quality",
        model=_judge_model_uri(),
        base_url=_judge_base_url(),
        instructions=(
            "Evaluate the quality of this conversation turn.\n\n"
            "{{ inputs }}\n"
            "{{ outputs }}\n\n"
            "Was the response helpful, clear, and professional? "
            "Answer 'good', 'acceptable', or 'poor' with rationale."
        ),
    )


# ---------------------------------------------------------------------------
# Code-based scorers (no LLM needed)
# ---------------------------------------------------------------------------

@scorer
def compliance_language_check(inputs, outputs, trace):
    """Check that the response uses appropriate banking compliance language."""
    from mlflow.entities import Feedback

    response_text = outputs if isinstance(outputs, str) else str(outputs)
    compliance_terms = ["risk", "compliance", "regulatory", "kyc", "aml", "approved", "flagged", "score", "credit"]
    found = [t for t in compliance_terms if t in response_text.lower()]

    return Feedback(
        name="compliance_language",
        value="yes" if found else "no",
        rationale=(
            f"Response contains compliance terms: {', '.join(found)}."
            if found
            else "Response lacks compliance-specific language."
        ),
    )


@scorer
def latency_budget_check(inputs, outputs, trace):
    """Check whether trace stayed within 15-second budget (banking SLA)."""
    from mlflow.entities import Feedback

    if trace and trace.info and trace.info.execution_time_ms is not None:
        within_budget = trace.info.execution_time_ms < 15_000
        return Feedback(
            name="latency_budget",
            value="pass" if within_budget else "fail",
            rationale=(
                f"Execution took {trace.info.execution_time_ms}ms "
                f"({'within' if within_budget else 'exceeds'} 15s SLA)."
            ),
        )
    return Feedback(name="latency_budget", value="unknown", rationale="No execution time data.")


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def evaluate_single_turn_traces(experiment_name: str, max_traces: int = 20):
    logger.info("--- Single-Turn Evaluation ---")
    logger.info("Searching traces in experiment '%s' ...", experiment_name)

    exp_id = _get_experiment_id(experiment_name)
    traces = mlflow.search_traces(experiment_ids=[exp_id], max_results=max_traces)

    if traces.empty:
        logger.warning("No traces found. Run the demo first.")
        return None

    logger.info("Found %d traces. Running scorers ...", len(traces))
    logger.info("Judge model: %s -> %s", _judge_model_uri(), _judge_base_url())

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[
            _build_relevance_judge(),
            _build_compliance_judge(),
            compliance_language_check,
            latency_budget_check,
        ],
    )

    logger.info("Single-turn evaluation complete.")
    logger.info("Results summary:\n%s", results.metrics)
    return results


def evaluate_multi_turn_session(experiment_name: str, session_id: str | None = None):
    logger.info("--- Multi-Turn Evaluation ---")

    if session_id:
        filter_str = f"metadata.`mlflow.trace.session` = '{session_id}'"
        logger.info("Filtering traces for session '%s' ...", session_id)
    else:
        filter_str = "metadata.`mlflow.trace.session` IS NOT NULL"
        logger.info("Searching all session traces ...")

    exp_id = _get_experiment_id(experiment_name)
    traces = mlflow.search_traces(
        experiment_ids=[exp_id], filter_string=filter_str, max_results=50,
    )

    if traces.empty:
        logger.warning("No session traces found.")
        return None

    logger.info("Found %d session traces. Running multi-turn scorers ...", len(traces))

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[_build_conversation_quality_judge(), compliance_language_check],
    )

    logger.info("Multi-turn evaluation complete.")
    logger.info("Results summary:\n%s", results.metrics)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate banking traces")
    parser.add_argument("--experiment", default="tracing-demo")
    parser.add_argument("--session", default=None)
    parser.add_argument("--single-turn-only", action="store_true")
    args = parser.parse_args()

    ensure_mlflow_initialized()
    evaluate_single_turn_traces(args.experiment)

    if not args.single_turn_only:
        evaluate_multi_turn_session(args.experiment, session_id=args.session)


if __name__ == "__main__":
    main()
