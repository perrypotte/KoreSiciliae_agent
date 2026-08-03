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
    collection_name="koreSiciliae_resources_v5",
    connection=os.getenv("DATABASE2_URL"),
)

route_matrix_repository = RouteMatrixRepository()

sparse_retriever = SparseRetriever(
    collection_id=os.getenv('COLLECTION_ID', 'a8d572b6-ac8b-4d21-9132-6d2c808e2d6a')
)

@tool()
def add_place_to_itinerary(document_id: str, place_name: str,visit_duration: float,travel_time: float=0.0):
    """Aggiunge un luogo selezionato all'itinerario dell'utente per un giorno specifico.
Args:
document_id (str): ID univoco del documento del luogo da aggiungere (uuid).
place_name (str): Nome del luogo da aggiungere.
visit_duration (float): Durata della visita in ore.
travel_time (float): Tempo di viaggio in ore."""
    #TODO fai finta al momento
    if cl.user_session.get("state", {}).get("remaining_time")>= visit_duration + travel_time:
        remaining_time = cl.user_session.get("state", {}).get("remaining_time") - (visit_duration + travel_time)
        cl.user_session.get("state", {})["remaining_time"] = remaining_time
        cl.user_session.get("state", {}).get("Selected_places", []).append({
            "document_id": document_id,
            "place_name": place_name,
            "visit_duration": visit_duration,
            "travel_time": travel_time,
            "day": cl.user_session.get("state", {}).get("current_day", 1)
        })
        return f"Il luogo '{place_name}' è stato aggiunto all'itinerario per il giorno {cl.user_session.get('state', {}).get('current_day', 1)}. Tempo rimanente per il giorno: {cl.user_session.get('state', {}).get('remaining_time', 0)} ore."
    else:
        return f"Non c'è abbastanza tempo rimanente per aggiungere '{place_name}' all'itinerario. Tempo necessario: {visit_duration + travel_time} ore, tempo rimanente: {cl.user_session.get('state', {}).get('remaining_time', 0)} ore."

@tool()
def finish_day():
    """Segna il giorno corrente come completato e passa al giorno successivo.
Questo accade sotto richiesta dell'utente oppure quando il tempo rimanente per il giorno corrente è esaurito."""
    state = cl.user_session.get("state", {})
    if state.get("current_day") is not None:
        state["current_day"] += 1
        state["remaining_time"] = state.get("Daily_hours", 0)
        cl.user_session.set("state", state)
        return f"Giorno {state['current_day'] - 1} completato. Passando al giorno {state['current_day']}."
    else:
        return "Errore: Giorno corrente non impostato."

@tool()
def update_planning_constraints(days: int, transport: Literal["auto", "pedestrian"], Max_distance_km: float, Daily_hours: float,current_day: int = 1):
    """Aggiorna i vincoli di pianificazione nella sessione dell'utente.
Args:
days (int): Numero di giorni per il viaggio.
transport (str): Mezzo di trasporto preferito ("auto" o "pedestrian").
Max_distance_km (float): Raggio massimo di ricerca in chilometri.
Daily_hours (float): Ore giornaliere disponibili per le attività.
current_day (int): Giorno corrente dell'itinerario."""

    cl.user_session.set("state", {
        "planning_mode": True,
        "days": days,
        "transport": transport,
        "Max_distance_km": Max_distance_km,
        "Daily_hours": Daily_hours,
        "Selected_places": [],
        "current_day": current_day,
        "remaining_time": Daily_hours,
    })
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
    
    if len(cl.user_session.get("state",{}).get("Selected_places")) > 0:
        print(f"\n\nGetting route matrix entries for selected places")
        # In questo caso le proposte devono essere arricchite/filtrate con distanza e tempo di percorrenza dal luogo precedente
        selected_places_ids = [place["document_id"] for place in cl.user_session.get("state",{}).get("Selected_places")]
        print(f"Selected places IDs: {selected_places_ids}")
        reranked_fused_docs_ids = [doc.metadata["document_id"] for doc in reranked_fused_docs]
        print(f"Reranked fused docs IDs: {reranked_fused_docs_ids}")
        route_matrix_entries = route_matrix_repository.get_matrix(
            source_ids=selected_places_ids,
            target_ids=reranked_fused_docs_ids,
            mode=cl.user_session.get("state", {}).get("transport"))
        print(f"Retrieved {len(route_matrix_entries)} route matrix entries for selected places")
        print(f"Route matrix entries: {route_matrix_entries}")
        filtred_reranked_fused_docs = []
        for doc in reranked_fused_docs:
            for entry in route_matrix_entries:
                if doc.metadata["document_id"] == entry.target_id and entry.source_id in selected_places_ids:
                    #TODO: Sistemare meglio i controlli di distanza  tipo usare solo l'ultimo luogo selezionato??ecc
                    if entry.distance_km > cl.user_session.get("state", {}).get("Max_distance_km"):
                        print(f"Skipping {doc.metadata['document_id']} due to distance {entry.distance_km} km exceeding max {cl.user_session.get('state', {}).get('Max_distance_km')} km")
                        continue
                    if entry.time_seconds > cl.user_session.get("state", {}).get("remaining_time") * 3600:
                        print(f"Skipping {doc.metadata['document_id']} due to travel time {entry.time_seconds/3600} hours exceeding remaining time {cl.user_session.get('state', {}).get('remaining_time')} hours")
                        continue
                    #TODO: Anche qua se si vuole controllo su distanza precisa
                    doc.metadata["distance_km"] = entry.distance_km
                    doc.metadata["travel_time_hours"] = entry.time_seconds /3600
                    filtred_reranked_fused_docs.append(doc)
                    break

        reranked_fused_docs = filtred_reranked_fused_docs

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
                    reasoning_budget=2500, #default 16384
                    chat_template_kwargs={"enable_thinking":True}),
    tools=[search_places, add_place_to_itinerary, update_planning_constraints, finish_day],
    middleware=[
        ProfileMiddleware()
    ],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt = """
Sei un semplice assistente virtuale che può attingere da una Knowledge Base basata su luoghi e attività nel territorio di Enna, le cui informazioni sull'utente saranno aggiornate dinamicamente durante la conversazione.
Fornisci le informazioni sull'utente se disponibili sotto richiesta.

Se l'utente chiede informazioni generiche su luoghi, utilizza il tool di ricerca per trovare documenti pertinenti anche utilizzando le preferenze dell'utente.

Osserva lo stato della conversazione, se l'utente chiede di voler creare un itinerario in base a questo chiedi all'utente le informazioni mancanti nello stato.
Osserva le preferenze dell'utente e assicurati che l'utente esprima le sue preferenze prima di procedere con la creazione del'itinerario.
Una volta che l'utente ha espresso le sue preferenze e vincoli avendo richiesto la creazione dell'itinerario procedi a proporre luoghi in base alle preferenze utilizzando una query in linguaggio naturale e concisa
e filtro tipo {"resource_types": ["Attivita_Degustazioni", "Attrazioni_Musei_e_mostre"]} per il tool di ricerca.

Quando l'utente chiede di aggiungere un luogo all'itinerario, utilizza il tool add_place_to_itinerary con i parametri corretti.
Quando l'utente chiede di terminare il giorno corrente, utilizza il tool finish_day e procedi con le prossime attività.

Una volta che l'utente ha selezionato almeno un luogo, avrai accesso alle informazioni di distanza e tempo di percorrenza tra i luoghi selezionati e quelli proposti dal tool di ricerca.
In questo caso può capitare che non ci siano luoghi proposti dal tool di ricerca che rispettino i vincoli di distanza e tempo rimanente per il giorno corrente. Avverti l'utente di questa situazione nel caso accada.
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
    msg = cl.Message(content="")
    has_streamed_tokens = False

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
                elements = []
                for i, doc in enumerate(docs):
                    md = doc.metadata
                    
                    # Usiamo textwrap per pulire il markdown senza indentazioni anomale
                    content = textwrap.dedent(f"""
                        # {md.get('title', 'Senza Titolo')}

                        **Categorie:** {", ".join(md.get("resource_types", []))}

                        ---

                        {doc.page_content}
                    """).strip()

                    elements.append(
                        cl.Text(
                            name=f"Documento {i+1}",
                            content=content,
                            display="side"
                        )
                    )
                
                # Invia un messaggio separato con la lista dei documenti nel pannello laterale
                await cl.Message(
                    content=f"📑 **Trovati {len(docs)} documenti di riferimento.**",
                    elements=elements
                ).send()

    # 3. CHIUSURA E SALVATAGGIO IN CRONOLOGIA
    if has_streamed_tokens:
        await msg.update()
        
        # Salva l'output finale nella tua user session / storia
        history.append(AIMessage(content=msg.content))
        messages["messages"] = history
        cl.user_session.set("messages", messages)