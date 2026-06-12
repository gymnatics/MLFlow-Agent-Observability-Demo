"""Evaluation script -- runs scorers on banking traces (Act 4).

Usage:
  python evaluate.py                       # evaluate latest traces
  python evaluate.py --session SESSION_ID  # evaluate a specific session
"""

import argparse
import logging

import mlflow
from mlflow.genai.scorers import (
    ConversationCompleteness,
    Guidelines,
    RelevanceToQuery,
    UserFrustration,
)
from mlflow.genai.scorers import scorer

from shared.mlflow_bootstrap import ensure_mlflow_initialized, get_model_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _judge_model() -> str:
    """Build the openai:/model-name URI for LLM judges, using the same vLLM endpoint as the agents."""
    return f"openai:/{get_model_name()}"


def _get_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return exp.experiment_id


@scorer
def compliance_language_check(inputs, outputs, trace):
    """Check that the response uses appropriate banking compliance language."""
    from mlflow.entities import Feedback

    response_text = outputs if isinstance(outputs, str) else str(outputs)
    compliance_terms = ["risk", "compliance", "regulatory", "kyc", "aml", "approved", "flagged"]
    has_compliance = any(term in response_text.lower() for term in compliance_terms)

    return Feedback(
        name="compliance_language",
        value="yes" if has_compliance else "no",
        rationale=(
            "Response uses appropriate compliance terminology."
            if has_compliance
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


def evaluate_single_turn_traces(experiment_name: str, max_traces: int = 20):
    logger.info("--- Single-Turn Evaluation ---")
    logger.info("Searching traces in experiment '%s' ...", experiment_name)

    exp_id = _get_experiment_id(experiment_name)
    traces = mlflow.search_traces(experiment_ids=[exp_id], max_results=max_traces)

    if traces.empty:
        logger.warning("No traces found. Run the demo first.")
        return None

    logger.info("Found %d traces. Running scorers ...", len(traces))

    judge = _judge_model()
    logger.info("Using judge model: %s", judge)

    banking_guidelines = Guidelines(
        name="banking_compliance",
        model=judge,
        guidelines=[
            "The response must not provide personal financial advice.",
            "The response should reference specific data points when making risk assessments.",
            "The response must be professional and factually grounded.",
            "Risk levels must be clearly stated as low, medium, or high.",
        ],
    )

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[
            RelevanceToQuery(model=judge),
            banking_guidelines,
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

    judge = _judge_model()
    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[ConversationCompleteness(model=judge), UserFrustration(model=judge)],
    )

    logger.info("Multi-turn evaluation complete.")
    logger.info("Results summary:\n%s", results.metrics)
    return results


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
