"""Compliance Reviewer A2A Executor."""
import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_task, new_agent_text_message

from agent import ComplianceReviewerAgent


class ComplianceReviewerExecutor(AgentExecutor):

    def __init__(self):
        self.agent = ComplianceReviewerAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        try:
            params = json.loads(user_input)
        except (json.JSONDecodeError, TypeError):
            params = {"assessment_data": user_input}

        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Checking compliance...", task.context_id, task.id),
        )

        result = await self.agent.review_compliance(params)

        parts: list[Part] = [Part(root=TextPart(text=json.dumps(result)))]
        await updater.add_artifact(parts)
        await updater.update_status(
            TaskState.completed,
            message=new_agent_text_message("Compliance review complete.", task.context_id, task.id),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")
