import json
from pathlib import Path
import uuid
import chainlit as cl
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from profile_middleware import ProfileMiddleware
from tools import ALL_TOOLS

def load_prompt(filename: str) -> str:
    return Path(f"prompts/{filename}").read_text(encoding="utf-8")

system_prompt = load_prompt("agent.txt")
profile_updater_sys_prompt=load_prompt("profile_updater.txt")


# Create the agent with a model and tools
agent = create_agent(
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b",
                    temperature=0.2, #default 1 basso per determinismo (modifica la distribuzione di probabilità delle parole successive)
                    top_p=0.4, #default 0.95, considera solo le parole la cui somma delle probabilità è >= a top_p 
                    max_tokens=2500, #default 16384
                    reasoning_budget=400, #default 16384
                    chat_template_kwargs={"enable_thinking":True}),
    tools=ALL_TOOLS,
    middleware=[
        ProfileMiddleware()
    ],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt =system_prompt
)

@cl.on_chat_start
async def start():
    state={
        "planning_mode":False,
        "days": None,
        "transport":None,
        "Max_distance_km":None,
        "Daily_hours":None,
        "Selected_places":[],
        "starting_place":None,
        "ending_place":None,
        "current_day":None,
        "remaining_time":None,
        # "preferred_resource_types": []
    }
    preferred_resource_types = {
        "preferred_resource_types": []
    }
    messages={
        "messages": [],
    }
    cl.user_session.set("state", state)
    cl.user_session.set("preferred_resource_types", preferred_resource_types)
    cl.user_session.set("messages", messages)
    #TODO: streammare il messaggio di benvenuto in tempo reale
    await cl.Message(
        AIMessage(
            content=f"""Ciao! 👋 Sono il tuo assistente per scoprire Enna e il suo territorio.

Posso aiutarti a:

🔎 trovare luoghi, attrazioni e attività in base ai tuoi interessi;
🍽️ proporti esperienze e attività coerenti con le tue preferenze;
🗺️ creare un itinerario personalizzato, organizzato giorno per giorno;
📍 tenere conto di distanze, tempi di percorrenza e tempo disponibile;
🏁 considerare, se vuoi, un indirizzo di partenza e un punto di arrivo della giornata;
ℹ️ darti informazioni specifiche sui luoghi presenti nella mia base di conoscenza.

Per iniziare a creare un itinerario, ti chiederò solo le informazioni necessarie, come quanti giorni hai a disposizione, i tuoi interessi e i tuoi vincoli di spostamento.

Se non indichi un punto di partenza o di arrivo, utilizzerò il centro di Enna come riferimento.

Durante la pianificazione potrai anche modificare i tuoi vincoli, aggiungere o rimuovere le attività e decidere passo dopo passo cosa inserire nell'itinerario.

Dimmi semplicemente cosa ti piacerebbe fare o scoprire a Enna e iniziamo!""",
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
    prompt = f"Aggiungi {place_title} (ID: {place_id}) (durata: {place_duration} ore) (tempo di viaggio: {place_travel_time} min) (tempo per tornare all'alloggio: {place_travel_time_to_accomodation}) all'itinerario"
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
    has_documents=False
    async for message_chunk, metadata in agent.astream(
        inputs,
        stream_mode="messages" #Per lo streaming dei token in tempo reale
    ):
        # 1. STREAMING DEI TOKEN DELL'AGENTE (AIMessageChunk)
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
            # Invia il primo token ed evidenzia che il messaggio è attivo
            if not has_streamed_tokens:
                await msg.send()
                has_streamed_tokens = True
            
            # Streamma il token in tempo reale nell'interfaccia
            await msg.stream_token(message_chunk.content)

        # 2. GESTIONE DEI TOOL / DOCUMENTI (ToolMessage)
        elif isinstance(message_chunk, ToolMessage):
            # Verifichiamo se il tool ha restituito documenti nell'artifact
            docs = getattr(message_chunk, "artifact", None)
            
            if docs and isinstance(docs, list):
                if message_chunk.name == "search_places":
                    has_documents=True
                    places = []
                    for doc in docs:
                        md = doc.metadata
                        # Genera un ID univoco per QUESTA risposta/carosello
                        carousel_id = f"carousel_{uuid.uuid4().hex[:8]}"
                        places.append({
                                "id": md["document_id"],
                                "title": md.get("title", "Senza titolo"),
                                #"image": md.get("image"),
                                "image": md["image_url"],
                                "url":md["official_url"] if md["official_url"]!="" else md["url"],
                                #"category": ", ".join(md.get("resource_types", [])),
                                "distance": md.get("distance_km_from_last_selected"),
                                "travel_time": md.get("travel_time_hours_from_last_selected"),
                                #"distance_km_to_final_destination":md.get("distance_km_to_final_destination"),
                                "travel_time_hours_to_final_destination":md.get("travel_time_hours_to_final_destination"),
                                "visit_duration": md.get("visit_duration"),
                                "planning":md.get("planning")
                            })


                        element = cl.CustomElement(
                            name="PlaceCarousel",
                            props={
                                "places": places,
                                "elementId": carousel_id # <--- ID UNIVOCO PER MESSAGGIO
                            }
                        )

    # 3. CHIUSURA E SALVATAGGIO IN CRONOLOGIA
    if has_streamed_tokens:
        await msg.update()
        
        # Salva l'output finale nella tua user session / storia
        history.append(AIMessage(content=msg.content))
        messages["messages"] = history
        cl.user_session.set("messages", messages)
        if has_documents:
            await cl.Message(
                            content="",
                            elements=[element]
                            ).send()



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

    #PERSISTENZA A LIVELLO DI SESSIONE DELLA CHAT
    # inputs = {
    #     "messages": history
    # }

    #TODO: ogni tanto pacca l'output non so perché
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0,top_p=0.20,chat_template_kwargs={"enable_thinking":False})
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

    response = model.invoke(conversation)
    # print(response)  # AIMessage("J'adore créer des applications.")

    print("Profilo prima: "+ str(cl.user_session.get("preferred_resource_types")))
    print("Profilo aggiornato: "+ str(response.content))
    cl.user_session.set("preferred_resource_types", json.loads(response.content))
    # preferred_resource_types = response.content
    print("Profilo dopo: "+ str(cl.user_session.get("preferred_resource_types")))
    # Prepara un messaggio Chainlit vuoto che aggiornerai via token
    await run_agent_pipeline(inputs,history)