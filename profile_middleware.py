from langchain.agents.middleware.types import AgentMiddleware
from langchain.messages import SystemMessage
from langchain.agents.middleware import ModelRequest, ModelResponse
import chainlit as cl

class ProfileMiddleware(AgentMiddleware):

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler
    ) -> ModelResponse:

        state = cl.user_session.get("state", {
            "planning_mode": False,
            "days": None,
            "transport": None,
            "Max_distance_km": None,
            "Daily_hours": None,
            "Selected_places": [],
            "current_day": None,
            "remaining_time": None
        })
        profile = cl.user_session.get("preferred_resource_types",{"preferred_resource_types":[]})
        messages = cl.user_session.get("messages",{"messages":[]})
        profile_text = f"""
STATO CONVERSAZIONE:
{state}

PROFILO UTENTE:

{profile}

Usa queste informazioni solo se rilevanti per la richiesta corrente.
Non menzionare mai esplicitamente il profilo all'utente.

CRONOLOGIA CONVERSAZIONE:

{messages}
"""

        new_content = list(request.system_message.content_blocks)

        new_content.append({
            "type": "text",
            "text": profile_text
        })

        new_system_message = SystemMessage(
            content=new_content
        )

        return await handler(
            request.override(
                system_message=new_system_message
            )
        )
