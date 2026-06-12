"""Compliance Reviewer A2A Server -- LLM-based regulatory compliance checks."""
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
from agent_executor import ComplianceReviewerExecutor

ensure_mlflow_initialized()

host = "0.0.0.0"
port = int(os.environ.get("PORT", 8003))

agent_card = AgentCard(
    name="Compliance Reviewer",
    description="Validates customer risk assessments against banking regulations and internal policies",
    url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="review_compliance",
            name="Review Compliance",
            description="Check a risk assessment against regulatory requirements",
            tags=["compliance", "regulation", "review"],
            examples=["Review this risk assessment for KYC and AML compliance"],
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
    agent_executor=ComplianceReviewerExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(agent_card=agent_card, http_handler=http_handler)
app = server.build()
app.add_middleware(TracingMiddleware)


async def health_check(request):
    return JSONResponse({"status": "healthy", "agent": "Compliance Reviewer"})


app.routes.insert(0, Route("/.well-known/agent-card.json", server._handle_get_agent_card, methods=["GET"]))
app.routes.insert(1, Route("/healthz", health_check, methods=["GET"]))
app.routes.insert(1, Route("/readyz", health_check, methods=["GET"]))

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
