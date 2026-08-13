import json
import os
import textwrap
from typing import Literal
import chainlit as cl
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_postgres import PGVector
import getpass
from profile_middleware import ProfileMiddleware
from sparse_retriever import SparseRetriever
from route_matrix_repository import RouteMatrixRepository, RouteMatrixEntry
from geocoding import geocode_address
from valhalla_tool import calculate_routes, calculate_routes_matrix

load_dotenv()

if os.getenv("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI PKey (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="koreSiciliae_resources_v6",
    connection=os.getenv("DATABASE2_URL"),
)

route_matrix_repository = RouteMatrixRepository()

sparse_retriever = SparseRetriever(
    collection_id=os.getenv('COLLECTION_ID', 'a8d572b6-ac8b-4d21-9132-6d2c808e2d6a')
)


@tool()
def get_place_info(document_id:str):
    """Ottiene i dettagli di una singola risorsa/luogo
Args:
document_id (str): ID univoco del documento di cui si vuole recuperare i dettagli."""
    document=vector_store.similarity_search(query="prova",k=1,filter={"document_id":document_id})
    return document

@tool()
def add_place_to_itinerary(document_id: str, place_name: str,visit_duration: float,travel_time: float=0.0,travel_time_to_accomodation: float=0.0):
    """Aggiunge un luogo selezionato all'itinerario dell'utente per un giorno specifico.
Args:
document_id (str): ID univoco del documento del luogo da aggiungere (uuid).
place_name (str): Nome del luogo da aggiungere.
visit_duration (float): Durata della visita in ore.
travel_time (float): Tempo di viaggio in ore dalla tappa precedente.
travel_time_to_accomodation (float): Tempo necessario per tornare all'alloggio a partire da qui"""
    #TODO fai finta al momento perché il controllo è fatto in search places  poi lo ritocchiamo
    if cl.user_session.get("state", {}).get("remaining_time")>= visit_duration + travel_time:
        remaining_time = cl.user_session.get("state", {}).get("remaining_time") - (visit_duration + travel_time)
        cl.user_session.get("state", {})["remaining_time"] = remaining_time
        cl.user_session.get("state", {}).get("Selected_places", []).append({
            "document_id": document_id,
            "place_name": place_name,
            "visit_duration": visit_duration,
            "travel_time": travel_time,
            "travel_time_to_accomodation": travel_time_to_accomodation,
            "day": cl.user_session.get("state", {}).get("current_day", 1)
        })
        return f"Il luogo '{place_name}' è stato aggiunto all'itinerario per il giorno {cl.user_session.get('state', {}).get('current_day', 1)}. Tempo rimanente per il giorno: {cl.user_session.get('state', {}).get('remaining_time', 0)} ore."
    else:
        return f"Non c'è abbastanza tempo rimanente per aggiungere '{place_name}' all'itinerario. Tempo necessario: {visit_duration + travel_time} ore, tempo rimanente: {cl.user_session.get('state', {}).get('remaining_time', 0)} ore. Tempo per tornare all'alloggio partendo dall'ultima tappa: {cl.user_session.get("state", {}).get("Selected_places", [])[-1].get("travel_time_to_accomodation")}"

@tool()
def finish_day():
    """Segna il giorno corrente come completato e passa al giorno successivo.
Questo accade sotto richiesta dell'utente oppure quando il tempo rimanente per il giorno corrente è esaurito."""
    state = cl.user_session.get("state", {})
    if state.get("current_day") is not None:
        state["current_day"] += 1
        state["remaining_time"] = state.get("Daily_hours", 0)
        state["starting_place"]=state["ending_place"]
        cl.user_session.set("state", state)
        #DEBUG
        tool_message=f"Giorno {state['current_day'] - 1} completato. Passando al giorno {state['current_day']}. Ricorda all'utente di specificare il luogo con indirizzo dove terminerà la giornata se necessario (Hotel, B&B,ecc...)"
        print(tool_message)
        return tool_message
    else:
        return "Errore: Giorno corrente non impostato."

@tool()
def update_planning_constraints(days: int, transport: Literal["auto", "pedestrian"], Max_distance_km: float, Daily_hours: float,current_day: int = 1,starting_place: str=None,ending_place:str=None):
    """Aggiorna i vincoli di pianificazione nella sessione dell'utente.
Args:
days (int): Numero di giorni per il viaggio.
transport (str): Mezzo di trasporto preferito tra auto e camminare a piedi ("auto" o "pedestrian").
Max_distance_km (float): Raggio massimo di ricerca in chilometri.
Daily_hours (float): Ore giornaliere disponibili per le attività.
current_day (int): Giorno corrente dell'itinerario.
starting_place (str): Luogo da cui parte la prima giornata (Es. Stazione centrale di Enna).
ending_place (str): Luoco in cui termina la giornata corrente (Es. Via esempio 17, Enna)."""

    cl.user_session.set("state", {
        "planning_mode": True,
        "days": days,
        "transport": transport,
        "Max_distance_km": Max_distance_km,
        "Daily_hours": Daily_hours,
        "Selected_places": [],
        "starting_place": geocode_address(starting_place) if starting_place!=None else geocode_address("Enna"),
        "ending_place": geocode_address(ending_place) if ending_place!=None else geocode_address("Enna"),
        "current_day": current_day,
        "remaining_time": Daily_hours,
    })
    print("Stato aggiornato:", cl.user_session.get("state", {}))
    return f"Vincoli di pianificazione aggiornati: {days} giorni, {transport} mezzo di trasporto, {Max_distance_km} km raggio massimo, {Daily_hours} ore giornaliere."

# $in non funziona per come vengono salvati gli array di resource_types su postgres, quindi bisogna costruire un filtro OR manualmente
# Non voglio farlo fare all'agente perché aggiungo complessità inutile e rischio di errori
# Aggiustare resource_types_or come nome del campo
def build_filter(f):
    conditions = []

    #TODO: aggiustare le condizioni qua che ormai non sono più necessarie con la nuova struttura dei documenti
    if "section_header" in f:
        conditions.append({"section_header": f["section_header"]})

    if "title" in f:
        conditions.append({"title": f["title"]})

    if "resource_types" in f:
        conditions.append({
            "$or": [
                {"resource_types": v}
                for v in f["resource_types"]
            ]
        })

    if len(conditions) == 0:
        return {}

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

@tool(response_format="content_and_artifact")
def search_places(query: str,filter: dict = None):
    """Retrieve information to help user find places to insert into their itinerary."""
    print("Raw filter input:", filter)
    raw_filter = filter
    filter=build_filter(filter) if filter else {}
    print(f"\n\nTool called with query: {query} and filter: {filter}\n\n")

    retrieved_docs = vector_store.similarity_search(query, k=10,filter=filter)
    dense_docs=retrieved_docs

    sparse_docs = sparse_retriever.invoke(
    query=query,
    k=10,
    resource_types=raw_filter
    )

    # for doc in sparse_docs[0:2]:
    #     print(f"\n\nSparse doc: {doc}\n\n")

    fused_docs = sparse_retriever.reciprocal_rank_fusion([dense_docs, sparse_docs])
    # print(len(fused_docs))

    # TODO: diventano 5 i documenti rerankati, che sia un limite dell'api?
    client = NVIDIARerank(
        model="nv-rerank-qa-mistral-4b:1",
        top_n=10 
        # api_key=os.getenv("NVIDIA_API_KEY"),
        )

    reranked_fused_docs = client.compress_documents(
        query=query,
        # documents=[Document(page_content=passage) for passage in fused_docs],
        documents=fused_docs,
        )
    # print(len(reranked_fused_docs))

   # ============================================================
   # 1. Recuperiamo gli ID dei candidati
   # ============================================================
    reranked_reranked_fused_docs = reranked_fused_docs

    candidate_ids = [
        doc.metadata["document_id"]
        for doc in reranked_reranked_fused_docs
    ]

    remaining_time=cl.user_session.get("state", {}).get("remaining_time")
    max_distance_km=cl.user_session.get("state", {}).get("Max_distance_km")
    # ============================================================
    # 2. Recuperiamo il punto di partenza della prossima tratta
    # ============================================================
    selected_places = cl.user_session.get("state",{}).get("Selected_places")
    current_day=cl.user_session.get("state",{}).get("current_day")
    print(f"Current day {current_day}\n")
    print(f"Last place day: {selected_places[-1].get("day")}") if selected_places else None
    if selected_places and current_day==selected_places[-1].get("day"):
        # Abbiamo già selezionato almeno un luogo:
        # la posizione corrente è l'ultimo luogo selezionato.
        last_selected = selected_places[-1]

        current_id = last_selected["document_id"]

        print(f"Current position = last selected place: {current_id}")

    else:
        # Nessun luogo ancora selezionato:
        # partiamo dall'indirizzo iniziale dinamico dell'utente.
        starting_point = cl.user_session.get("state", {}).get("starting_place")

        # Non dovrebbe accadere teoricamente
        if not starting_point:
            print("Starting point non disponibile")
            reranked_fused_docs = []
        else:
            current_id = starting_point.get("id", "starting_point")#Prende default la seconda stringa

            print(
                f"Current position = dynamic starting point: "
                f"{starting_point}"
            )


    # ============================================================
    # 3. Calcoliamo current -> candidati
    # ============================================================
    current_to_candidates = {}
    transport=cl.user_session.get("state", {}).get("transport")
    if selected_places and current_day==selected_places[-1].get("day"):

        # --------------------------------------------------------
        # CASO NORMALE:
        # ultimo POI selezionato -> candidati
        #
        # Questi dati li abbiamo già nella route_matrix DB.
        # --------------------------------------------------------

        selected_place_id = selected_places[-1]["document_id"]
        print(
            f"\nGetting DB route matrix entries "
            f"from {selected_place_id}"
        )

        route_matrix_entries = route_matrix_repository.get_matrix(
            source_ids=[selected_place_id],
            target_ids=candidate_ids,
            mode=transport,
        )

        for entry in route_matrix_entries:
            current_to_candidates[entry.target_id] = {
                "distance_km": entry.distance_km,
                "time_seconds": entry.time_seconds,
            }

    else:

        # --------------------------------------------------------
        # PRIMA TAPPA:
        # starting point dinamico -> candidati
        #
        # Non possiamo usare la route_matrix DB perché lo starting
        # point non è un POI conosciuto.
        # --------------------------------------------------------

        starting_point = cl.user_session.get("state", {}).get("starting_place")

        if starting_point:

            start_location = {
                "id": "starting_point",
                "lat": starting_point["lat"],
                "lon": starting_point["lon"],
            }

            candidate_locations = []

            for doc in reranked_reranked_fused_docs:

                md = doc.metadata

                candidate_locations.append({
                    "id": md["document_id"],
                    "lat": md["latitude"],
                    "lon": md["longitude"],
                })

            runtime_routes = calculate_routes(
                source=start_location,
                targets=candidate_locations,
                mode=transport,
            )

            for route in runtime_routes:
                current_to_candidates[route["target_id"]] = {
                    "distance_km": route["distance_km"],
                    "time_seconds": route["time_seconds"],
                }


    # ============================================================
    # 4. Calcoliamo candidato -> destinazione finale
    # ============================================================

    final_destination = cl.user_session.get("state", {}).get("ending_place")

    candidate_to_final = {}


    if final_destination:

        final_location = {
            "id": "final_destination",
            "lat": final_destination["lat"],
            "lon": final_destination["lon"],
        }
        candidate_locations = []

        for doc in reranked_reranked_fused_docs:

            md = doc.metadata

            candidate_locations.append({
                "id": md["document_id"],
                "lat": md["latitude"],
                "lon": md["longitude"],
            })

        # N sorgenti -> 1 destinazione finale
        print("LOG: Calcolo viaggio di ritorno all'alloggio...")
        runtime_routes = calculate_routes_matrix(
            sources=candidate_locations,
            targets=[final_location],
            mode=transport,
        )

        for route in runtime_routes:
            candidate_to_final[route["source_id"]] = {
                "distance_km": route["distance_km"],
                "time_seconds": route["time_seconds"],
            }
    # ============================================================
    # 5. Filtriamo i candidati
    # ============================================================

    filtered_docs = []

    for doc in reranked_reranked_fused_docs:

        document_id = doc.metadata["document_id"]

        # --------------------------------------------------------
        # Dati current -> candidato
        # --------------------------------------------------------

        current_route = current_to_candidates.get(document_id)

        if not current_route:
            print(
                f"Skipping {document_id}: "
                f"nessuna rotta current -> candidato"
            )
            continue

        current_distance = current_route["distance_km"]
        current_time_seconds = current_route["time_seconds"]


        # --------------------------------------------------------
        # Dati candidato -> destinazione finale
        # --------------------------------------------------------

        final_route = candidate_to_final.get(document_id)

        if not final_route:
            print(
                f"Skipping {document_id}: "
                f"nessuna rotta candidato -> destinazione finale"
            )
            continue

        final_distance = final_route["distance_km"]
        final_time_seconds = final_route["time_seconds"]


        # --------------------------------------------------------
        # Durata visita
        # --------------------------------------------------------

        import re

        raw_duration = doc.metadata.get("visit_duration", 0)

        match = re.search(r"\d+(?:\.\d+)?", str(raw_duration))

        visit_duration = float(match.group()) if match else 0.0


        # --------------------------------------------------------
        # Tempo totale necessario per rendere la tappa fattibile
        #
        # current
        #    ↓
        # candidate
        #    ↓ visita
        # candidate
        #    ↓
        # final destination
        # --------------------------------------------------------

        travel_to_candidate_hours = current_time_seconds / 3600
        travel_to_final_hours = final_time_seconds / 3600

        total_required_time = (
            travel_to_candidate_hours
            + visit_duration
            + travel_to_final_hours
        )


        # --------------------------------------------------------
        # Controllo tempo
        # --------------------------------------------------------

        if total_required_time > remaining_time:

            print(
                f"Skipping {document_id}: "
                f"required={total_required_time:.2f}h, "
                f"remaining={remaining_time:.2f}h"
            )

            continue


        # --------------------------------------------------------
        # Controllo distanza
        #
        # Qui assumiamo che Max_distance_km sia il massimo
        # consentito per una singola tratta.
        # --------------------------------------------------------

        if max_distance_km is not None:

            if current_distance > max_distance_km:

                print(
                    f"Skipping {document_id}: "
                    f"current -> candidate = "
                    f"{current_distance:.2f} km > "
                    f"{max_distance_km} km"
                )

                continue

            if final_distance > max_distance_km:

                print(
                    f"Skipping {document_id}: "
                    f"candidate -> final = "
                    f"{final_distance:.2f} km > "
                    f"{max_distance_km} km"
                )

                continue


        # --------------------------------------------------------
        # Arricchiamo il documento
        # --------------------------------------------------------

        doc.metadata["distance_km_from_last_selected"] = current_distance

        doc.metadata["travel_time_hours_from_last_selected"] = (
            travel_to_candidate_hours
        )

        doc.metadata["distance_km_to_final_destination"] = final_distance

        doc.metadata["travel_time_hours_to_final_destination"] = (
            travel_to_final_hours
        )

        doc.metadata["total_required_time_hours"] = total_required_time

        doc.metadata["visit_duration_hours"] = visit_duration


        filtered_docs.append(doc)


    reranked_fused_docs = filtered_docs

    #DEBUG
    #print(reranked_fused_docs[0])
    #print(f"documenti: {len(reranked_fused_docs)}")
    #TODO: Caso semplificato, documenti non compatibili ai vincoli di pianificazione
    if(len(reranked_fused_docs)==0):
        return ("Non sono presenti documenti che rispettano la distanza o i limiti di tempo insieme ai luoghi di arrivo e partenza dell'utente. Proponi all'utente di estendere la distanza massima", [])
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_fused_docs
    )
    return serialized, reranked_fused_docs

# Create the agent with a model and tools
agent = create_agent(
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b",
                    temperature=0.3, #default 1 basso per determinismo (modifica la distribuzione di probabilità delle parole successive)
                    top_p=0.5, #default 0.95, considera solo le parole la cui somma delle probabilità è >= a top_p 
                    max_tokens=2500, #default 16384
                    reasoning_budget=400, #default 16384
                    chat_template_kwargs={"enable_thinking":True}),
    tools=[search_places, add_place_to_itinerary, update_planning_constraints, finish_day,get_place_info],
    middleware=[
        ProfileMiddleware()
    ],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt = """
Sei un semplice assistente virtuale che può attingere da una Knowledge Base basata su luoghi e attività nel territorio di Enna.
Le informazioni sull'utente vengono aggiornate dinamicamente durante la conversazione.
Fornisci informazioni sull'utente esclusivamente se disponibili nello stato della conversazione e se richieste esplicitamente.
Quando l'utente chiede informazioni generiche su luoghi o attività, utilizza il tool search_places per trovare documenti pertinenti.
Quando utile, considera anche le preferenze dell'utente presenti nello stato della conversazione.
Quando l'utente manifesta la volontà di creare un itinerario, osserva lo STATO DELLA CONVERSAZIONE e chiedi esclusivamente le informazioni mancanti necessarie alla pianificazione.
Informa l'utente che può indicare opzionalmente un indirizzo di partenza e un indirizzo per la fine della giornata. Se non vengono indicati, considera il centro di Enna come punto di partenza e come punto di arrivo della giornata.
Quando l'utente fornisce informazioni relative ai vincoli di pianificazione, aggiorna IMMEDIATAMENTE lo stato utilizzando il tool update_planning_constraints. Dopo l'aggiornamento, verifica nuovamente lo stato e chiedi eventuali informazioni ancora mancanti.
L'utente può successivamente modificare i singoli vincoli di pianificazione (STATO DELLA CONVERSAZIONE), se la pianificazione è già attiva allora l'utente può modificare un singolo valore.
Prima di procedere con la creazione dell'itinerario, assicurati che l'utente abbia espresso le proprie preferenze. Le preferenze dell'utente corrispondono ai tag/categorie delle attività che considera interessanti.
Quando l'utente ha espresso le proprie preferenze e tutti i vincoli necessari sono disponibili, utilizza search_places per proporre luoghi pertinenti.
Formula una query breve in linguaggio naturale, semanticamente utile e coerente con le preferenze dell'utente. Utilizza inoltre un filtro resource_types coerente con le categorie delle preferenze, ad esempio {"resource_types": ["Attivita_Degustazioni", "Attrazioni_Musei_e_mostre"]}.
Quando search_places restituisce pochi risultati (il massimo è 10) prova 1-2 volte a cambiare la query rimanendo sempre sul tema corrente riguardo le categorie e le preferenze, dopo di ciò se le proposte rimangono sempre poche proponi all'utente di chiedere esplicitamente cosa vorrebbe fare o cambiare i vincoli di pianificazione. 
Quando l'utente chiede di aggiungere un luogo all'itinerario, utilizza il tool add_place_to_itinerary con tutti i parametri corretti disponibili.
Quando l'utente chiede di terminare il giorno corrente, utilizza il tool finish_day e procedi con la pianificazione delle attività del giorno successivo.
Dopo che l'utente ha selezionato almeno un luogo, saranno disponibili informazioni relative a distanza e tempo di percorrenza tra i luoghi selezionati e quelli proposti da search_places. Utilizza queste informazioni per verificare la compatibilità delle proposte con il tempo rimanente e con i vincoli di distanza e tempo dell'itinerario.
Se nessuno dei luoghi restituiti da search_places rispetta i vincoli di distanza o il tempo rimanente per il giorno corrente, informa brevemente l'utente della situazione. Non inventare mai distanze, tempi o altre informazioni.
Quando mostri all'utente l'output derivato da search_places, lascia descrivere l'output all'interfaccia grafica: non fare l'elenco dei documenti ottenuti dal tool perché sarà l'interfaccia grafica a mostrarli all'utente. L'utente potrà utilizzare i pulsanti dell'interfaccia per aggiungere una proposta all'itinerario o richiedere informazioni su un luogo specifico.
Puoi invece mostrare un breve riepilogo dello stato dell'itinerario in corso.
Quando l'utente richiede informazioni specifiche su un luogo, utilizza esclusivamente le informazioni disponibili nella Knowledge Base e nei tool. Non inventare informazioni mancanti.
In ogni fase osserva lo STATO DELLA CONVERSAZIONE e non chiedere nuovamente informazioni già disponibili.
Quando ragioni o rispondi, sii conciso, diretto e schematico. Non aggiungere spiegazioni o dettagli inutili. Non inventare informazioni e, se non sei sicuro di qualcosa, ammettilo chiaramente.
"""
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
            content=f"""Ciao! Sono il tuo assistente virtuale per la base di conoscenza di Kore Siciliae.
Posso aiutarti a trovare informazioni su luoghi, attrazioni, attività, esperienze e negozi.
Inoltre posso anche aiutarti a creare un itinerario di viaggio.
### PRIMO PASSO: Inserisci le tue preferenze e interessi.""",
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
                                "visit_duration": md.get("visit_duration")
                            })


                        element = cl.CustomElement(
                            name="PlaceCarousel",
                            props={
                                "places": places
                            }
                        )


                    

                # Old presentation of retrieved documents in the chat, now replaced by cards
                # else:
                #     elements = []
                #     for i, doc in enumerate(docs):
                #         md = doc.metadata
                        
                #         # Usiamo textwrap per pulire il markdown senza indentazioni anomale
                #         content = textwrap.dedent(f"""
                #             # {md.get('title', 'Senza Titolo')}

                #             **Categorie:** {", ".join(md.get("resource_types", []))}

                #             ---

                #             {doc.page_content}
                #         """).strip()

                #         elements.append(
                #             cl.Text(
                #                 name=f"Documento {i+1}",
                #                 content=content,
                #                 display="side"
                #             )
                #         )
                    
                #     # Invia un messaggio separato con la lista dei documenti nel pannello laterale
                #     await cl.Message(
                #         content=f"📑 **Trovati {len(docs)} documenti di riferimento.**",
                #         elements=elements
                #     ).send()

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
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0,top_p=0.95,chat_template_kwargs={"enable_thinking":False})
    conversation = [
    {"role": "system", "content": """Sei un Profile Updater.

Il tuo unico compito è mantenere aggiornato il profilo temporaneo della conversazione.

Riceverai sempre:

- il profilo corrente della conversazione;
- l'ultimo messaggio dell'utente.

Il profilo rappresenta esclusivamente le preferenze espresse durante questa conversazione e NON deve essere considerato permanente.

Il profilo ha il seguente formato:

{
    "preferred_resource_types": [
        ...
    ]
}

Le uniche categorie ammesse sono:

- Attivita_Attività_culturali
- Attivita_Corsi_e_laboratori
- Attivita_Degustazioni
- Attivita_Escursioni
- Attivita_Sport
- Attivita_Visite_guidate
- Attrazioni_Castelli_e_fortezze
- Attrazioni_Giardini_monumentali
- Attrazioni_Luoghi_di_culto
- Attrazioni_Musei_e_mostre
- Attrazioni_Paesaggio_e_natura
- Attrazioni_Palazzi_e_monumenti
- Attrazioni_Parchi_e_oasi_naturali
- Attrazioni_Siti_archeologici
- Shopping_Artigianato
- Shopping_Cibo_e_vino
- Shopping_Gioielli

Devi seguire rigorosamente queste regole.

1. Aggiorna il profilo SOLO quando il messaggio esprime chiaramente una preferenza, un interesse, un gusto oppure una preferenza negativa.

Esempi:

- "Mi piace il vino."
- "Sono interessato ai musei."
- "Adoro le degustazioni."
- "Preferisco le escursioni."
- "Non mi interessano i castelli."

2. NON modificare il profilo quando il messaggio contiene soltanto una richiesta di informazioni.

Esempi:

- "Parlami del Castello di Lombardia."
- "Quali musei ci sono?"
- "A che ora apre il museo?"
- "Consigliami un ristorante."

Queste NON rappresentano preferenze permanenti della conversazione.

3. Se il messaggio modifica una preferenza precedente, aggiorna il profilo.

Esempio:

Profilo:
{
    "preferred_resource_types": [
        "Shopping_Cibo_e_vino"
    ]
}

Messaggio:
"In realtà preferisco visitare musei."

Nuovo profilo:
{
    "preferred_resource_types": [
        "Attrazioni_Musei_e_mostre"
    ]
}

4. Non inventare preferenze.

5. Non dedurre preferenze implicite da una singola domanda.

6. Mantieni il profilo invariato se il messaggio non contiene informazioni utili (riguardanti preferenze esplicite dell'utente).

7. Il profilo deve contenere solo categorie appartenenti all'elenco fornito.

8. Non aggiungere spiegazioni.

9. Restituisci esclusivamente il nuovo profilo o il profilo invariato in formato JSON valido.

10. Non trasformare mai il profilo in una lista di preferenze, ma mantienilo sempre come un oggetto JSON con la chiave "preferred_resource_types".

Nient'altro.
"""},

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