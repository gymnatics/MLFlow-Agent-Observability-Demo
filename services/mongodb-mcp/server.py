"""MongoDB MCP Server -- FastMCP wrapper for banking customer database.

Provides tools for retrieving customer profiles and loan history.

Trace context propagation:
  FastMCP handles this natively via MCP _meta fields. When the client
  (Customer Analyst) calls call_tool(), FastMCP injects traceparent into
  _meta. On the server side, FastMCP extracts _meta and restores the
  OTel context. The @mlflow.trace decorators on each tool then create
  child spans under that restored context.

  Set FASTMCP_TELEMETRY_MODE=propagation_only to suppress FastMCP's own
  OTel spans and let @mlflow.trace be the sole span source.
"""
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastmcp import FastMCP
from pymongo import MongoClient

import mlflow
from shared.mlflow_bootstrap import ensure_mlflow_initialized

logger = logging.getLogger(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongodb:27017")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "banking_crm")

ensure_mlflow_initialized()

mcp = FastMCP("Banking MongoDB MCP")

_client = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    return _client[MONGODB_DATABASE]


@mcp.tool()
@mlflow.trace(name="mcp.get_customer", span_type="TOOL")
def get_customer(customer_id: str) -> dict:
    """Look up a customer profile by ID (e.g., C-1001)."""
    db = _get_db()
    doc = db.customers.find_one({"customer_id": customer_id}, {"_id": 0})
    if doc is None:
        return {"error": f"Customer {customer_id} not found"}
    return doc


@mcp.tool()
@mlflow.trace(name="mcp.search_customers", span_type="TOOL")
def search_customers(
    tier: str = "",
    min_credit_score: int = 0,
    min_balance: float = 0,
    limit: int = 10,
) -> list[dict]:
    """Search customers by tier, credit score, or balance."""
    db = _get_db()
    query: dict = {}
    if tier:
        query["tier"] = tier
    if min_credit_score > 0:
        query["credit_score"] = {"$gte": min_credit_score}
    if min_balance > 0:
        query["account_balance"] = {"$gte": min_balance}

    docs = list(db.customers.find(query, {"_id": 0}).limit(limit))
    return docs


@mcp.tool()
@mlflow.trace(name="mcp.get_loan_history", span_type="TOOL")
def get_loan_history(customer_id: str) -> list[dict]:
    """Retrieve all loan records for a customer."""
    db = _get_db()
    docs = list(db.loans.find({"customer_id": customer_id}, {"_id": 0}))
    return docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    from seed_data import seed
    try:
        seed()
    except Exception as e:
        logger.warning("Seed failed (DB may not be ready yet): %s", e)

    mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
