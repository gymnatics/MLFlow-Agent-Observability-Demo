#!/bin/bash
set -e

# ============================================================================
# Banking MLflow Tracing Demo -- Deploy Script
#
# Prerequisites:
#   - oc logged in to your RHOAI 3.4 cluster
#   - MLflow operator enabled and MLflow CR deployed
#   - A vLLM model serving endpoint (Qwen3 or similar)
#
# Usage:
#   ./deploy.sh                    # Full deploy (build + apply)
#   ./deploy.sh --build-only       # Just build images
#   ./deploy.sh --apply-only       # Just apply k8s manifests (images must exist)
#   ./deploy.sh --teardown         # Remove everything
# ============================================================================

NAMESPACE="${NAMESPACE:-mlflow-tracing-demo}"
GIT_REPO="${GIT_REPO:-https://github.com/gymnatics/MLFlow-Agent-Observability-Demo.git}"
SERVICES="mongodb-mcp customer-analyst risk-assessor compliance-reviewer orchestrator banking-dashboard"

echo "=== Banking MLflow Tracing Demo ==="
echo "Namespace: $NAMESPACE"
echo "Git repo:  $GIT_REPO"
echo ""

# --- Teardown ---
if [[ "$1" == "--teardown" ]]; then
    echo "Tearing down..."
    oc delete -k k8s/base/ -n "$NAMESPACE" 2>/dev/null || true
    for svc in $SERVICES; do
        oc delete bc "$svc" -n "$NAMESPACE" 2>/dev/null || true
        oc delete is "${svc}atest" -n "$NAMESPACE" 2>/dev/null || true
    done
    oc delete project "$NAMESPACE" 2>/dev/null || true
    echo "Done. Namespace $NAMESPACE deleted."
    exit 0
fi

# --- Create namespace ---
if ! oc get project "$NAMESPACE" &>/dev/null; then
    echo "Creating namespace $NAMESPACE..."
    oc new-project "$NAMESPACE" --display-name="MLflow Tracing Demo"
else
    oc project "$NAMESPACE"
fi

# --- Build images ---
if [[ "$1" != "--apply-only" ]]; then
    echo ""
    echo "=== Building container images ==="
    for svc in $SERVICES; do
        if oc get bc "$svc" -n "$NAMESPACE" &>/dev/null; then
            echo "  Starting build: $svc"
            oc start-build "$svc" -n "$NAMESPACE" --wait 2>/dev/null || true
        else
            echo "  Creating BuildConfig: $svc"
            dockerfile_path="services/$svc/Dockerfile"
            [[ "$svc" == "banking-dashboard" ]] && dockerfile_path="services/frontend/Dockerfile"
            oc new-build "$GIT_REPO" \
                --name="$svc" \
                --strategy=docker \
                --dockerfile="$(cat "$dockerfile_path")" \
                --to="${svc}atest:latest" \
                -n "$NAMESPACE" 2>/dev/null
        fi
    done

    echo ""
    echo "Waiting for all builds to complete..."
    for svc in $SERVICES; do
        latest_build=$(oc get builds -n "$NAMESPACE" -l buildconfig="$svc" --sort-by='.metadata.creationTimestamp' -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
        if [[ -n "$latest_build" ]]; then
            echo -n "  $latest_build: "
            oc wait "build/$latest_build" --for=jsonpath='{.status.phase}'=Complete --timeout=600s -n "$NAMESPACE" 2>/dev/null && echo "Complete" || echo "Failed"
        fi
    done
fi

# --- Apply manifests ---
if [[ "$1" != "--build-only" ]]; then
    echo ""
    echo "=== Applying Kubernetes manifests ==="
    oc apply -k k8s/base/ -n "$NAMESPACE"

    echo ""
    echo "=== Waiting for deployments ==="
    for dep in mongodb mongodb-mcp customer-analyst risk-assessor compliance-reviewer orchestrator banking-dashboard; do
        echo -n "  $dep: "
        oc rollout status "deploy/$dep" -n "$NAMESPACE" --timeout=120s 2>/dev/null && echo "" || echo "TIMEOUT"
    done

    # Seed MongoDB
    echo ""
    echo "=== Seeding MongoDB ==="
    sleep 5
    oc exec deploy/mongodb-mcp -n "$NAMESPACE" -- python seed_data.py 2>/dev/null || echo "  (already seeded)"
fi

# --- Print access info ---
echo ""
echo "=== Deployment Complete ==="
DASHBOARD_URL=$(oc get route banking-dashboard -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
echo ""
echo "  Dashboard:  https://$DASHBOARD_URL"
echo "  MLflow UI:  $(oc get mlflow mlflow -o jsonpath='{.status.url}' 2>/dev/null)"
echo "  Namespace:  $NAMESPACE"
echo ""
echo "  To port-forward orchestrator locally:"
echo "    oc port-forward svc/orchestrator 9000:8000 -n $NAMESPACE"
echo ""
echo "  To run evaluators:"
echo "    python evaluate.py --experiment banking-demo"
echo ""
