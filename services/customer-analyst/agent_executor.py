"""Customer Analyst A2A Executor."""
import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_task, new_agent_text_message

from agent import CustomerAnalystAgent


class CustomerAnalystExecutor(AgentExecutor):

    def __init__(self):
        self.agent = CustomerAnalystAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        try:
            params = json.loads(user_input)
            skill = params.pop("skill", "get_customer_profile")
        except (json.JSONDecodeError, TypeError):
            params = {"customer_id": user_input}
            skill = "get_customer_profile"

        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"Looking up customer data...", task.context_id, task.id,
            ),
        )

        result = await self.agent.handle_skill(skill, params)

        parts: list[Part] = [Part(root=TextPart(text=json.dumps(result)))]
        await updater.add_artifact(parts)
        await updater.update_status(
            TaskState.completed,
            message=new_agent_text_message("Customer data retrieved.", task.context_id, task.id),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")
