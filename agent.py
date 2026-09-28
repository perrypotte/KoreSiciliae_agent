import asyncio
import json
import os
from pathlib import Path
import uuid
import chainlit as cl
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from profile_middleware import ProfileMiddleware
from tools import ALL_TOOLS
from planning_state import PlanningState
from langchain_community.callbacks import get_openai_callback

model=ChatOpenAI(
    #LOCAL HOST
    base_url="http://127.0.0.1:8888/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="unsloth/Qwen3.5-9B-GGUF",
    # # model="unsloth/gemma-4-12B-it-qat-GGUF",
    #OPENROUTER PAID
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # model="qwen/qwen3.5-9b",
    # model="meta-llama/llama-3.3-70b-instruct",
    # model="qwen/qwen-2.5-72b-instruct",
    # model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.2,
    top_p=0.4,
    # max_tokens=2500,
    # I parametri custom di OpenRouter vanno inseriti in extra_body
    streaming=True,
    stream_usage=True,
    reasoning={
    "effort": "low",  # Default None; can be "low", "medium", or "high"
    "summary": "concise",  # Can be "auto", "concise", or "detailed"
},
    # reasoning_effort="minimal"
)

def load_prompt(filename: str) -> str:
    return Path(f"prompts/{filename}").read_text(encoding="utf-8")

system_prompt = load_prompt("agent.txt")
profile_updater_sys_prompt=load_prompt("profile_updater.txt")


# Create the agent with a model and tools
agent = create_agent(
    # model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b",
    #                 temperature=0.2, #default 1 basso per determinismo (modifica la distribuzione di probabilità delle parole successive)
    #                 top_p=0.4, #default 0.95, considera solo le parole la cui somma delle probabilità è >= a top_p 
    #                 max_tokens=2500, #default 16384
    #                 reasoning_budget=400, #default 16384
    #                 chat_template_kwargs={"enable_thinking":True}),
    #FREE MODEL
    # model=ChatOpenRouter(
        # model="nvidia/nemotron-3-super-120b-a12b:free",
        # temperature=0.2,
        # top_p=0.4,
        # max_tokens=2500,
        # reasoning={"effort":"medium","summary":"concise"},
    # ),
    #PAID MODEL
    model=model,
    tools=ALL_TOOLS,
    middleware=[
        ProfileMiddleware()
    ],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt =system_prompt
)

@cl.on_chat_start
async def start():
    state=PlanningState()
    preferred_resource_types = {
        "preferred_resource_types": []
    }
    messages={
        "messages": [],
    }
    cl.user_session.set("token_usage",0)
    cl.user_session.set("state", state)
    cl.user_session.set("preferred_resource_types", preferred_resource_types)
    cl.user_session.set("messages", messages)
    #TODO: streammare il messaggio di benvenuto in tempo reale
    await cl.Message(
        AIMessage(
            content=f"""Ciao! 👋 Sono il tuo assistente per scoprire Enna e il suo territorio.

Posso aiutarti a:

🔎 trovare luoghi, attrazioni e attività fornendo una piccola descrizione;
🍽️ proporti esperienze e attività coerenti con le tue preferenze;
🗺️ creare un itinerario personalizzato, organizzato giorno per giorno;
📍 tenere conto di distanze, tempi di percorrenza e tempo disponibile;
🏁 considerare, se vuoi, un indirizzo di partenza e un punto di arrivo della giornata  (Limitato alla Sicilia);
ℹ️ darti informazioni specifiche sui luoghi presenti nella mia base di conoscenza.

Per iniziare a creare un itinerario, ti chiederò solo le informazioni necessarie, come quanti giorni hai a disposizione, i tuoi interessi e i tuoi vincoli di spostamento.

Se non indichi un punto di partenza o di arrivo, utilizzerò il centro di Enna come riferimento.

Durante la pianificazione potrai decidere passo passo cosa inserire nell'itinerario e modificare se necessario la distanza massima che puoi percorrere per andare alla tappa successiva.

Vuoi una mano a creare un itinerario o preferisci chiedermi qualcosa su Enna?
 
Iniziamo!""",
        ).content
    ).send()


@cl.action_callback("agent_info_place")
async def ui_bridge_get_place_info(action: cl.Action):
    place_document_id=action.payload.get("id")
    messages = cl.user_session.get("messages", {"messages": []})
    history = messages["messages"]
    # Costruisci l'istruzione per l'agente
    prompt = f"Recupera informazioni sul documento con Id=\"{place_document_id}\""
    inputs = {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
    # Esegue la stessa identica pipeline di streaming dell'agente!
    # MEMO: tolto dai file di traduzione il prefisso: "Utilizzato"
    async with cl.Step("Recupero informazioni..."):
        await run_agent_pipeline(inputs,history)

@cl.action_callback("agent_add_place")
async def ui_bridge_add_place(action: cl.Action):
    place_title = action.payload.get("title")
    place_id = action.payload.get("id")
    place_duration = action.payload.get("duration", 0.0)
    place_travel_time = action.payload.get("travel_time", 0.0)
    place_travel_time_to_accomodation= action.payload.get("travel_time_hours_to_final_destination",0.0)
    messages = cl.user_session.get("messages", {"messages": []})
    history = messages["messages"]
    # Costruisci l'istruzione per l'agente
    prompt = f"Aggiungi {place_title} (ID: {place_id}) (durata: {place_duration} ore) (tempo di viaggio: {round(place_travel_time,2)} ore) (tempo per tornare all'alloggio: {round(place_travel_time_to_accomodation,2)} ore) all'itinerario"
    inputs = {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
    # Esegue la stessa identica pipeline di streaming dell'agente!
    # MEMO: tolto dai file di traduzione il prefisso: "Utilizzato"
    async with cl.Step("Aggiungo il luogo all'itinerario..."):
        await run_agent_pipeline(inputs,history)

# 1. FUNZIONE CENTRALE: contiene la tua logica di chiamata all'agente e streaming
async def run_agent_pipeline(inputs: dict,history: list):
    messages = cl.user_session.get("messages", {"messages": []})
    msg = cl.Message(content="")
    has_streamed_tokens = False
    has_documents = False
    element = None
    # 2. Avvolgi l'esecuzione nel context manager del callback
    with get_openai_callback() as cb:
        async for message_chunk, metadata in agent.astream(
            inputs,
            stream_mode="messages"
        ):
            # 1. STREAMING DEI TOKEN DELL'AGENTE (AIMessageChunk)
            if isinstance(message_chunk, AIMessageChunk):
                # Salta se il chunk è una chiamata tool
                if getattr(message_chunk, "tool_call_chunks", None) or getattr(message_chunk, "tool_calls", None):
                    continue

                # Estrae solo il testo finale pulito, ignorando message_chunk.reasoning
                token_text = getattr(message_chunk, "text", None)
                
                # Fallback di sicurezza se la proprietà text non è presente
                if token_text is None:
                    content = message_chunk.content
                    if isinstance(content, str):
                        token_text = content
                    elif isinstance(content, list):
                        token_text = "".join(
                            b.get("text", "") for b in content 
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        token_text = ""

                # Filtro rapido per eventuali frammenti grezzi residui
                if not token_text or "ool_call>" in token_text:
                    continue

                if not has_streamed_tokens:
                    await msg.send()
                    has_streamed_tokens = True
                
                await msg.stream_token(token_text)

            # 2. GESTIONE DEI TOOL / DOCUMENTI (ToolMessage)
            elif isinstance(message_chunk, ToolMessage):
                docs = getattr(message_chunk, "artifact", None)
                if docs and isinstance(docs, list) and message_chunk.name == "search_places":
                    has_documents = True
                    places = []
                    for doc in docs:
                        md = doc.metadata
                        places.append({
                            "id": md["document_id"],
                            "title": md.get("title", "Senza titolo"),
                            "image": md.get("image_url"),
                            "url": md.get("official_url") or md.get("url"),
                            "distance": md.get("distance_km_from_last_selected"),
                            "travel_time": md.get("travel_time_hours_from_last_selected"),
                            "travel_time_hours_to_final_destination": md.get("travel_time_hours_to_final_destination"),
                            "visit_duration": md.get("visit_duration"),
                            "planning": md.get("planning")
                        })

                    carousel_id = f"carousel_{uuid.uuid4().hex[:8]}"
                    element = cl.CustomElement(
                        name="PlaceCarousel",
                        props={
                            "places": places,
                            "elementId": carousel_id
                        }
                    )

    # 3. CHIUSURA E SALVATAGGIO IN CRONOLOGIA
    if has_streamed_tokens:
        await msg.update()
        history = messages.get("messages", [])
        history.append(AIMessage(content=msg.content))
        messages["messages"] = history
        cl.user_session.set("messages", messages)

    if has_documents and element:
        await cl.Message(content="", elements=[element]).send()

    # Al termine dello streaming, il callback contiene il totale aggregato
    # (inclusi i token spesi per eventuali chiamate ai tool intermedie)
    # print("\n--- STATISTICHE TOKEN ---")
    # print(f"Prompt Tokens:     {cb.prompt_tokens}")
    # print(f"Completion Tokens: {cb.completion_tokens}")
    # print(f"Total Tokens:      {cb.total_tokens}")
    # print(f"Costo Stimato:     ${cb.total_cost:.6f}")
    cl.user_session.set("token_usage", cl.user_session.get("token_usage", 0) + cb.total_tokens)
    print(f"Token totali accumulati in sessione: {cl.user_session.get('token_usage', 0)}")

@cl.on_message
async def main(message: cl.Message):
    # Input to start the loop
    inputs = {
        "messages": [
            # {"role": "user", "content": "Give me information about Arancino Experience?"},
            {"role": "user", "content": message.content},
        ]
    }

    elements = []

    messages = cl.user_session.get("messages", {"messages": []})
    history = messages["messages"]
    history.append(
        HumanMessage(content=message.content)
    )

    #TODO: ogni tanto pacca l'output non so perché
    #FREE MODEL
    #model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0,top_p=0.20,chat_template_kwargs={"enable_thinking":False})

    #PAID MODEL
    model=ChatOpenAI(
    base_url="http://127.0.0.1:8888/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    # model="unsloth/gemma-4-12B-it-qat-GGUF",
    model="unsloth/Qwen3.5-9B-GGUF",
    #OPENROUTER PAID
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # model="qwen/qwen3.5-9b",
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # # model="nvidia/nemotron-3-super-120b-a12b",
    # # model="meta-llama/llama-3.3-70b-instruct",
    # model="qwen/qwen-2.5-72b-instruct",
    temperature=0.2,
    top_p=0.4,
    reasoning={
        "effort": "None",  # Default None; can be "low", "medium", or "high"
        "summary": "concise",  # Can be "auto", "concise", or "detailed"
    },
    # reasoning_effort="minimal" NON FUNZIONA MA NELLA DOCS C'E'
)
    conversation = [
    {"role": "system", "content":profile_updater_sys_prompt},
    {
        "role": "user",
        "content": f"""
                        PROFILO CORRENTE:
                        {json.dumps(cl.user_session.get("preferred_resource_types", {}).get("preferred_resource_types", []), ensure_ascii=False)}
                        Ultimo messaggio dell'utente:
                        {message.content}.
                    """
    },
        ]
    try:
        response = model.invoke(conversation)
        # print(response[0]["text"])
        print(response)  # AIMessage("J'adore créer des applications.")
        print("Profilo prima: "+ str(cl.user_session.get("preferred_resource_types")))
        
        try:
            print("Profilo aggiornato: "+ str(response.content[1]["text"])) #Quando c'è reasoning è nel primo posto
            cl.user_session.set("preferred_resource_types", json.loads(response.content[1]["text"]))
        except Exception as e:
            print("Profilo aggiornato: "+ str(response.content[0]["text"])) #Quando c'è reasoning è nel primo posto
            cl.user_session.set("preferred_resource_types", json.loads(response.content[0]["text"]))
        # preferred_resource_types = response.content
        print("Profilo dopo: "+ str(cl.user_session.get("preferred_resource_types")))
        # Prepara un messaggio Chainlit vuoto che aggiornerai via token
        await run_agent_pipeline(inputs,history)
    except Exception as e:
        msg = cl.Message(
            content="⚠️ Servizio momentaneamente sovraccarico. Riprova tra poco ad inviare il messaggio."
        )
        import traceback
        traceback.print_exc()
        # print(e)
        await msg.send()
        await asyncio.sleep(30)
        await msg.remove()
        return