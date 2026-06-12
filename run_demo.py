"""Demo runner -- executes the full banking tracing demo (Acts 1-6).

Usage:
  # Mock mode (no LLM / MongoDB needed)
  USE_MOCK=1 MLFLOW_TRACKING_URI=http://localhost:5000 python run_demo.py

  # Live mode (agents running as services)
  MLFLOW_TRACKING_URI=https://... python run_demo.py
"""

import asyncio
import json
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))

import mlflow
import mlflow.tracing

from shared.mlflow_bootstrap import ensure_mlflow_initialized, is_mock_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70


async def act1_single_request_tracing():
    """Act 1: Full pipeline trace -- customer lookup -> risk -> compliance."""
    logger.info(SEPARATOR)
    logger.info("ACT 1: Multi-Agent Span Tracing")
    logger.info("Assessing customer C-1001 through the full pipeline")
    logger.info(SEPARATOR)

    if is_mock_mode():
        from services.orchestrator.agent import OrchestratorAgent
        agent = OrchestratorAgent()
        result = await agent.handle_skill("assess_customer", {"customer_id": "C-1001"})
    else:
        from shared.mlflow_bootstrap import traced_a2a_call
        result = await traced_a2a_call(
            os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
            "assess_customer",
            {"customer_id": "C-1001"},
        )

    logger.info("Result: %s", json.dumps(result, indent=2, default=str)[:600])
    logger.info(">> Open MLflow UI -> Traces to see: Orchestrator -> Customer Analyst -> MongoDB MCP -> Risk Assessor -> Compliance Reviewer\n")
    return result


async def act2_different_risk_profiles():
    """Act 2: Trace different agent types with varying risk profiles."""
    logger.info(SEPARATOR)
    logger.info("ACT 2: Distributed Tracing Across Agent Types")
    logger.info("Assessing high-risk customer C-1006 (delinquent loans)")
    logger.info(SEPARATOR)

    if is_mock_mode():
        from services.orchestrator.agent import OrchestratorAgent
        agent = OrchestratorAgent()
        result = await agent.handle_skill("assess_customer", {"customer_id": "C-1006"})
    else:
        from shared.mlflow_bootstrap import traced_a2a_call
        result = await traced_a2a_call(
            os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
            "assess_customer",
            {"customer_id": "C-1006"},
        )

    logger.info("Result: %s", json.dumps(result, indent=2, default=str)[:600])
    logger.info(">> Compare trace with Act 1 -- different risk levels, different LLM reasoning\n")
    return result


async def act3_multi_turn_session():
    """Act 3: Multi-turn banking conversation with session_id."""
    logger.info(SEPARATOR)
    logger.info("ACT 3: Multi-Turn Conversation Tracing")
    logger.info("Running a 3-turn banking inquiry with the same session_id")
    logger.info(SEPARATOR)

    session_id = f"banking-session-{uuid.uuid4().hex[:8]}"
    turns = [
        "What is the current loan status for customer C-1004?",
        "What is their credit score and risk category?",
        "Would they qualify for a new personal loan of $20,000?",
    ]

    for i, turn in enumerate(turns, 1):
        logger.info("Turn %d: %s", i, turn)

        with mlflow.tracing.context(session_id=session_id):
            if is_mock_mode():
                from services.orchestrator.agent import OrchestratorAgent
                agent = OrchestratorAgent()
                result = await agent.handle_skill("chat", {
                    "query": turn,
                    "session_id": session_id,
                })
            else:
                from shared.mlflow_bootstrap import traced_a2a_call
                result = await traced_a2a_call(
                    os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
                    "chat",
                    {"query": turn, "session_id": session_id, "skill": "chat"},
                )

        response = result.get("response", result.get("final_response", str(result)))
        logger.info("Response: %s", str(response)[:200])
        logger.info("")

    logger.info("Session ID: %s", session_id)
    logger.info(">> MLflow UI -> filter by session to see all 3 turns grouped\n")
    return session_id


async def act4_evaluators(session_id: str | None = None):
    """Act 4: Evaluators and Scorers on banking traces."""
    logger.info(SEPARATOR)
    logger.info("ACT 4: Evaluators and Scorers")
    logger.info("Running built-in + custom scorers on collected traces")
    logger.info(SEPARATOR)

    from evaluate import evaluate_single_turn_traces, evaluate_multi_turn_session

    evaluate_single_turn_traces("tracing-demo", max_traces=10)

    if session_id:
        evaluate_multi_turn_session("tracing-demo", session_id=session_id)

    logger.info(">> Check MLflow UI -> Traces -> Assessments tab for scorer results")
    logger.info(">> For production: schedule evaluate.py via CronJob or pipeline for continuous monitoring")
    logger.info(">> Note: MLflow 3.11+ adds server-side automatic evaluation (scorer.register() + .start())\n")


async def act5_metrics_and_search():
    """Act 5: Trace search, filtering, and aggregation."""
    logger.info(SEPARATOR)
    logger.info("ACT 5: Metrics, Thresholds, and Monitoring")
    logger.info("Demonstrating trace search, filtering, and aggregation")
    logger.info(SEPARATOR)

    exp = mlflow.get_experiment_by_name("tracing-demo")
    if exp is None:
        logger.warning("Experiment 'tracing-demo' not found.")
        return

    traces_df = mlflow.search_traces(
        experiment_ids=[exp.experiment_id],
        max_results=50,
    )

    if traces_df.empty:
        logger.warning("No traces found.")
        return

    logger.info("Total traces: %d", len(traces_df))

    if "info.execution_time_ms" in traces_df.columns:
        avg_latency = traces_df["info.execution_time_ms"].mean()
        max_latency = traces_df["info.execution_time_ms"].max()
        logger.info("Average latency: %.0fms", avg_latency)
        logger.info("Max latency: %.0fms", max_latency)

        slow_threshold = 10000
        slow_traces = traces_df[traces_df["info.execution_time_ms"] > slow_threshold]
        if not slow_traces.empty:
            logger.warning("ALERT: %d traces exceeded %dms threshold!", len(slow_traces), slow_threshold)

    if "info.state" in traces_df.columns:
        error_traces = traces_df[traces_df["info.state"] == "ERROR"]
        if not error_traces.empty:
            logger.warning("ALERT: %d traces in ERROR state!", len(error_traces))

    logger.info(">> For production: integrate with Prometheus/AlertManager via OpenShift monitoring\n")


async def act6_architecture_summary():
    """Act 6: Architecture Explanation."""
    logger.info(SEPARATOR)
    logger.info("ACT 6: Architecture Summary")
    logger.info(SEPARATOR)
    logger.info("""
Trace Chain Architecture:

  User Request
    -> Orchestrator (LangGraph + @mlflow.trace + langchain.autolog)
       -> Customer Analyst (A2A + traceparent via traced_a2a_call)
          -> MongoDB MCP (FastMCP _meta context propagation + @mlflow.trace)
             -> MongoDB query
       -> Risk Assessor (A2A + traceparent)
          -> Qwen3-8B LLM (openai.autolog captures call automatically)
       -> Compliance Reviewer (A2A + traceparent)
          -> Qwen3-8B LLM (openai.autolog captures call automatically)

  A2A propagation:  TracingMiddleware extracts traceparent from HTTP headers
  MCP propagation:  FastMCP injects/extracts traceparent via MCP _meta fields
  Both connect spans into one trace via W3C TraceContext (OpenTelemetry).

  SDK:          mlflow[kubernetes,genai]>=3.11
  Auth:         kubernetes-namespaced SA tokens on RHOAI 3.4
  Collection:   Direct to MLflow server REST API, no OTel collector needed
""")


async def main():
    ensure_mlflow_initialized()

    logger.info("Banking MLflow Tracing Demo")
    logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Mock mode: %s", is_mock_mode())
    logger.info("")

    await act1_single_request_tracing()
    await act2_different_risk_profiles()
    session_id = await act3_multi_turn_session()
    await act4_evaluators(session_id)
    await act5_metrics_and_search()
    await act6_architecture_summary()

    logger.info(SEPARATOR)
    logger.info("Demo complete! Open MLflow UI to explore traces.")
    logger.info(SEPARATOR)


if __name__ == "__main__":
    asyncio.run(main())
