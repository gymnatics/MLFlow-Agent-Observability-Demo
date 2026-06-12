# Banking MLflow Tracing Demo on RHOAI 3.4

A banking-domain multi-agent system demonstrating MLflow 3.x tracing, evaluation, and observability on Red Hat OpenShift AI 3.4. Built to answer **Question 6** of the Enterprise Agentic Platform requirements.

## What It Does

A user asks "Assess credit risk for customer C-1001" and the system:
1. **Customer Analyst** looks up the customer profile and loan history from MongoDB (via MCP)
2. **Risk Assessor** uses Qwen3-8B to evaluate credit risk
3. **Compliance Reviewer** uses Qwen3-8B to check regulatory compliance

All spans are connected in one trace via W3C traceparent propagation.

## Trace Chain

```
User Request
  └── Orchestrator (LangGraph workflow)
       ├── Customer Analyst (A2A + traceparent)
       │    └── MongoDB MCP (FastMCP + traceparent)
       │         └── MongoDB query
       ├── Risk Assessor (A2A + traceparent)
       │    └── Qwen3-8B LLM call (autolog)
       └── Compliance Reviewer (A2A + traceparent)
            └── Qwen3-8B LLM call (autolog)
```

## Project Structure

```
services/
  orchestrator/          # LangGraph orchestrator (A2A, port 8000)
  customer-analyst/      # Calls MongoDB MCP for customer data (A2A, port 8001)
  risk-assessor/         # LLM-based credit risk analysis (A2A, port 8002)
  compliance-reviewer/   # LLM-based compliance checks (A2A, port 8003)
  mongodb-mcp/           # FastMCP server wrapping MongoDB (port 8090)
shared/
  mlflow_bootstrap.py    # TracingMiddleware + traced_a2a_call + autolog setup
k8s/
  base/
    agents/              # Deployment + Service per agent
    mcp/                 # MongoDB + MongoDB MCP deployments
    configmap.yaml       # Environment config
    rbac.yaml            # SA + mlflow-integration RoleBinding
  mlflow-cr.yaml         # MLflow CR for RHOAI 3.4
run_demo.py              # Full demo runner (Acts 1-6)
evaluate.py              # Evaluation with scorers
```

## Quick Start (Local with Mock Data)

```bash
pip install -r requirements.txt

export MLFLOW_TRACKING_URI="https://<dashboard>/mlflow"
export MLFLOW_TRACKING_TOKEN="$(oc whoami --show-token)"
export MLFLOW_WORKSPACE="mlflow-tracing-demo"
export MLFLOW_TRACKING_INSECURE_TLS=true
export USE_MOCK=1

python run_demo.py
```

## Deploy on OpenShift

```bash
# 1. Create namespace
oc new-project mlflow-tracing-demo

# 2. Apply infrastructure
oc apply -f k8s/base/rbac.yaml
oc apply -f k8s/base/configmap.yaml
oc apply -f k8s/base/mcp/mongodb.yaml

# 3. Wait for MongoDB, then deploy MCP
oc apply -f k8s/base/mcp/mongodb-mcp.yaml

# 4. Deploy agents
oc apply -f k8s/base/agents/

# 5. Seed database
oc exec deploy/mongodb-mcp -- python seed_data.py
```

## Customer Question 6 Coverage

| Requirement | How Demo Addresses It |
|---|---|
| Trace data as spans | Nested trace tree: Orchestrator -> Agents -> MCP -> DB |
| Multi-turn conversations | Session grouping with `session_id` (Act 3) |
| Different agent types | Customer lookup, risk analysis, compliance review |
| Evaluators on traces | Built-in + custom scorers (Act 4) |
| Custom dashboard | MLflow search/filter, aggregated metrics (Act 5) |
| Threshold and alerts | Latency SLA checks + Prometheus integration |
| Trace collection SDK | MLflow SDK only, no OTel collector needed |
| Central trace collection | MLflow server = trace store + UI |
