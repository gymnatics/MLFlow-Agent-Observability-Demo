"""Orchestrator A2A Server -- coordinates banking workflow via LangGraph."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities,
    SecurityScheme, HTTPAuthSecurityScheme,
)
from starlette.routing import Route
from starlette.responses import JSONResponse

from shared.mlflow_bootstrap import ensure_mlflow_initialized, TracingMiddleware
from agent_executor import OrchestratorExecutor

ensure_mlflow_initialized()

host = "0.0.0.0"
port = int(os.environ.get("PORT", 8000))

agent_card = AgentCard(
    name="Banking Orchestrator",
    description="Coordinates customer analysis, risk assessment, and compliance review workflows using LangGraph",
    url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="assess_customer",
            name="Assess Customer",
            description="Full customer assessment: lookup profile, evaluate risk, check compliance",
            tags=["customer", "risk", "compliance", "workflow"],
            examples=["Assess the credit risk and compliance for customer C-1001"],
        ),
        AgentSkill(
            id="chat",
            name="Chat",
            description="Multi-turn banking conversation with session context",
            tags=["chat", "conversation", "multi-turn"],
            examples=["What is the loan status for customer C-1006?"],
        ),
    ],
    securitySchemes={
        "Bearer": SecurityScheme(root=HTTPAuthSecurityScheme(
            type="http", scheme="bearer", bearerFormat="JWT",
            description="OAuth 2.0 JWT token",
        ))
    },
)

http_handler = DefaultRequestHandler(
    agent_executor=OrchestratorExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(agent_card=agent_card, http_handler=http_handler)
app = server.build()
app.add_middleware(TracingMiddleware)


async def health_check(request):
    return JSONResponse({"status": "healthy", "agent": "Banking Orchestrator"})


app.routes.insert(0, Route("/.well-known/agent-card.json", server._handle_get_agent_card, methods=["GET"]))
app.routes.insert(1, Route("/healthz", health_check, methods=["GET"]))
app.routes.insert(1, Route("/readyz", health_check, methods=["GET"]))

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
