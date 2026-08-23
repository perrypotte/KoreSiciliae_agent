from langchain.agents.middleware.types import AgentMiddleware
from langchain.messages import SystemMessage
from langchain.agents.middleware import ModelRequest, ModelResponse
import chainlit as cl
from planning_state import PlanningState



class ProfileMiddleware(AgentMiddleware):

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler
    ) -> ModelResponse:

        state = cl.user_session.get("state", PlanningState())
        print("DEBUG - MIDDLEWARE: "+state.to_prompt())
        profile = cl.user_session.get("preferred_resource_types",{"preferred_resource_types":[]})
        messages = cl.user_session.get("messages",{"messages":[]})
        profile_text = f"""
STATO PIANIFICAZIONE ITINERARIO:
{state.to_prompt()}

PREFERENZE UTENTE:

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
