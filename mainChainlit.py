import json
from typing import List
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_postgres import PGVector
import getpass
import os
from dotenv import load_dotenv

import chainlit as cl

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
    collection_name="koreSiciliae_resources_v4",
    connection=os.getenv("DATABASE2_URL"),
)


import os

from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
    
engine = create_engine(os.getenv("DATABASE2_URL"))

#TODO : usare un id di collection "koreSiciliae_resources_v4"
collection_id = "b6740afb-61ce-4bcd-bb36-2450cef9f190"
def load_documents():
    docs = []

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id},
        )

        for row in rows:
            docs.append(
                Document(
                    page_content=row.document,
                    metadata=row.cmetadata
                )
            )

    return docs

all_documents = load_documents()

from collections import defaultdict

def reciprocal_rank_fusion(rankings, k=20): #TODO Studiare bene gli effetti di k 
    scores = defaultdict(float)
    documents = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc.metadata["document_id"]      # oppure altro identificatore univoco
            #doc_id = hash(doc.page_content)

            scores[doc_id] += 1 / (k + rank + 1)
            documents[doc_id] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [documents[doc_id] for doc_id, _ in ranked]


# $in non funziona per come vengono salvati gli array di resource_types su postgres, quindi bisogna costruire un filtro OR manualmente
# Non voglio farlo fare all'agente perché aggiungo complessità inutile e rischio di errori
# Aggiustare resource_types_or come nome del campo
def build_filter(f):
    conditions = []

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

def filter_documents_by_resource_types(
    documents,
    resource_types: list[str] | None
):
    """
    Filtra i Document mantenendo solo quelli che hanno
    almeno un resource_type presente nella lista richiesta.
    """

    if not resource_types:
        return documents

    filtered_docs = []
    for doc in documents:
        doc_types = doc.metadata.get("resource_types", [])

        #DEBUG
        #print(f"Type of doc_types: {type(doc_types)}, Value: {doc_types}")
        
        # nel caso resource_types sia salvato come stringa singola
        if isinstance(doc_types, str):
            doc_types = [doc_types]

        if any(rt in resource_types for rt in doc_types):
            filtered_docs.append(doc)

    return filtered_docs

@tool(response_format="content_and_artifact")
def retrieve_context(query: str,filter: dict = None):
    """Retrieve information to help answer a query."""
    print("Raw filter input:", filter)
    raw_filter = filter
    filter=build_filter(filter) if filter else {}
    print(f"\n\nTool called with query: {query} and filter: {filter}\n\n")
    retrieved_docs = vector_store.similarity_search(query, k=10,filter=filter)
    dense_docs=retrieved_docs

    # filtro BM25 in base ai resource_types 
    # TODO: SEMPRE TUTTO IN MEMORIA E INTERO DATASET RIESPLORATO OGNI VOLTA PER FILTRARE, VALUTARE COMPUTAZIONALMENTE E SPAZIALMENTE
    bm25_documents = filter_documents_by_resource_types(
        all_documents,
        raw_filter.get("resource_types")
    )
    #DEBUG
    #print("Numero documenti prima:", len(all_documents))
    #print("Numero documenti BM25:", len(bm25_documents))
    bm25 = BM25Retriever.from_documents(bm25_documents)
    bm25.k = 10
    
    sparse_docs = bm25.invoke(query)#query
    #print(f"\n\nSparse docs: {sparse_docs}\n\n")
    # print(f"\n\nDense docs: {dense_docs[0]}\n\n")
    # print(f"\n\nSparse docs: {sparse_docs[0]}\n\n")
    
    fused_docs = reciprocal_rank_fusion(
        [dense_docs, sparse_docs]
    )[:5]
    
    #retrieved_docs = vector_store.similarity_search(query, k=4,filter={"$and":[{"section_header":"Periodi e orari di apertura"},{"$or":[{"resource_types":"Attivita_Degustazioni"},{"resource_types":"Shopping_Cibo_e_vino"}]}]})
    #print(f"\n\nRetrieved DEBUG{(retrieved_docs)} documents.\n\n")
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in fused_docs
    )

    return serialized, fused_docs

from langchain.agents.middleware.types import AgentMiddleware
from langchain.messages import SystemMessage
from langchain.agents.middleware import ModelRequest, ModelResponse

class ProfileMiddleware(AgentMiddleware):

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler
    ) -> ModelResponse:

        profile = cl.user_session.get("state", {}).get("profile", {})

        profile_text = f"""
USER PROFILE

{profile}

Usa queste informazioni solo se rilevanti per la richiesta corrente.
Non menzionare mai esplicitamente il profilo all'utente.
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

# Create the agent with a model and tools
agent = create_agent(
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b"),
    tools=[retrieve_context],
    middleware=[
        ProfileMiddleware()
    ],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt = """
Sei un agente intelligente di recupero delle informazioni per una base di conoscenza strutturata relativa a luoghi, attrazioni, attività, esperienze e negozi.

La tua unica fonte di informazioni fattuali è lo strumento di recupero. Non fare affidamento sulle tue conoscenze per rispondere a domande riguardanti la base di conoscenza.

========================
COMPORTAMENTO GENERALE
========================

- Rispondi sempre alla richiesta dell'utente.
- Non porre mai domande di chiarimento o di follow-up.
- Non inventare mai informazioni.
- Non fare mai supposizioni.
- Usa lo strumento di recupero ogni volta che sono necessarie informazioni presenti nella base di conoscenza.
- Puoi utilizzare lo strumento di recupero più volte, se necessario.
- Non fermarti dopo il primo recupero se sono necessarie ulteriori ricerche per rispondere in modo completo alla richiesta dell'utente.

========================
SELEZIONE DEI RESOURCE TYPE
========================

Prima di ogni recupero, determina quali resource_types hanno maggiori probabilità di contenere le informazioni richieste.

Resource type disponibili:

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

Linee guida:

- Non cercare di individuare una sola categoria "migliore".
- Seleziona tutti i resource_types che potrebbero ragionevolmente contenere informazioni pertinenti.
- Seleziona da 1 a 5 resource_types.
- Inserisci i resource_types selezionati nel parametro filter della chiamata allo strumento di recupero come lista.
- Se l'utente richiede informazioni basate sui propri gusti o preferenze, seleziona i resource_types indicati nel profilo della conversazione.

Esempi:

Cibo, ristoranti, vino, prodotti tipici:
- Shopping_Cibo_e_vino
- Attivita_Degustazioni

Attività all'aperto:
- Attivita_Escursioni
- Attivita_Sport
- Attrazioni_Paesaggio_e_natura
- Attrazioni_Parchi_e_oasi_naturali

Castelli:
- Attrazioni_Castelli_e_fortezze

Musei:
- Attrazioni_Musei_e_mostre

Chiese e monasteri:
- Attrazioni_Luoghi_di_culto

Edifici storici e monumenti:
- Attrazioni_Palazzi_e_monumenti

Servizi di emergenza, ospedali, farmacie, parcheggi, stazioni di polizia:
- Altro

Informazioni vaghe o richieste basate su preferenze personali:
- Seleziona i resource_types indicati nel profilo della conversazione.

Esempio specifico:

Utente: "Dimmi dove posso mangiare la pizza a Catania."

filter:
{"resource_types": ["Shopping_Cibo_e_vino", "Attivita_Degustazioni"]}

========================
RICHIESTE CON PIÙ RISORSE
========================

L'utente potrebbe chiedere informazioni su più luoghi, attrazioni, attività o negozi nella stessa richiesta.

In questi casi:

- Identifica tutte le risorse richieste.
- Recupera le informazioni per ciascuna di esse.
- Effettua ulteriori chiamate allo strumento di recupero ogni volta che è necessario.
- Non fermarti dopo aver recuperato informazioni su una sola risorsa.
- Combina tutte le informazioni recuperate in un'unica risposta coerente.

Esempi:

"Confronta il Castello Ursino e il Monastero dei Benedettini."

Recupera le informazioni su entrambe le risorse prima di rispondere.

"Suggeriscimi musei e chiese a Catania."

Recupera sia i musei sia le chiese prima di rispondere.

========================
UTILIZZO DELLE INFORMAZIONI RECUPERATE
========================

Ogni risorsa recuperata contiene già tutte le informazioni disponibili.

- Usa esclusivamente le informazioni contenute nelle risorse recuperate.
- Se più risorse contengono informazioni utili, integrale in modo naturale.
- Ignora le informazioni non pertinenti alla richiesta dell'utente.
- Se le informazioni recuperate non sono sufficienti, effettua un nuovo recupero invece di fare supposizioni.
- Se non è possibile recuperare informazioni pertinenti, dichiara chiaramente di non aver trovato le informazioni richieste.

========================
RISPOSTA FINALE
========================

Genera un'unica risposta in linguaggio naturale.

La risposta deve essere:

- accurata
- completa
- concisa
- facile da leggere
- direttamente focalizzata sulla domanda dell'utente

Non menzionare mai:

- il recupero delle informazioni
- gli strumenti
- i filtri
- i metadati
- i resource_types
- gli embeddings
- la ricerca vettoriale
- il ragionamento interno
- la pianificazione

Non produrre mai:

- JSON
- elenchi di decisioni interne
- spiegazioni del tuo ragionamento
- analisi
- pensieri
- piani

La risposta finale deve contenere solo il testo destinato all'utente, senza essere troncata.
"""
)

@cl.on_chat_start
async def start():
    state = {
    "messages": [],
    "profile": {
        "preferred_resource_types": []
    }
}
    cl.user_session.set("state", state)


@cl.on_message
async def main(message: cl.Message):
    # Input to start the loop
    inputs = {
        "messages": [
            #{"role": "user", "content": "What is Arancino Experience?"},
            # {"role": "user", "content": "Give me information about Arancino Experience?"},
            {"role": "user", "content": message.content},
            #{"role": "user", "content": "Is there some place where i can see bees in Enna territory?"},
            #{"role": "user", "content": "At what time does Arancino Experience operate?"},
            #{"role": "user", "content": "Can you suggest three places where i can eat in Enna territory?"},
            #{"role": "user", "content": "Can you suggest places where i can buy souveniers or some wearable items in Enna territory?"},
            #{"role": "user", "content": "Can you give me some information about Arancino Experience?"},
            #{"role": "user", "content": "test, don't answer"},

        ]
    }

    elements = []

    state = cl.user_session.get("state")
    history = state["messages"]
    history.append(
        HumanMessage(content=message.content)
    )

    #PERSISTENZA A LIVELLO DI SESSIONE DELLA CHAT
    # inputs = {
    #     "messages": history
    # }

    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0)
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

6. Mantieni il profilo invariato se il messaggio non contiene informazioni utili.

7. Il profilo deve contenere solo categorie appartenenti all'elenco fornito.

8. Non aggiungere spiegazioni.

9. Restituisci esclusivamente il nuovo profilo o il profilo invariato in formato JSON valido.

Nient'altro.
"""},

    {
        "role": "user",
        "content": f"""
PROFILO CORRENTE:
{json.dumps(cl.user_session.get("state", {}).get("profile", {}), ensure_ascii=False)}
Ultimo messaggio dell'utente:
{message.content}.
"""
    },
        ]

    response = model.invoke(conversation)
    # print(response)  # AIMessage("J'adore créer des applications.")

    print(cl.user_session.get("state", {}).get("profile", {}))
    state["profile"] = response.content
    print(state["profile"])
    async for chunk in agent.astream(
       inputs
    , stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"User: {latest_message.content}")
                
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
                await cl.Message(
                    content=f"Agent: {latest_message.content}",
                ).send()
                history.append(
                    AIMessage(content=latest_message.content)
                )
                state["messages"] = history
                cl.user_session.set("state", state)
            else:
                # print(f"Altro: {latest_message}")
                docs=latest_message.artifact
                for i, doc in enumerate(docs):

                    md = doc.metadata

                    content = f"""
                    # {md['title']}

                    **Categorie**
                    {", ".join(md["resource_types"])}

                    ---

                    {doc.page_content}
                    """

                    elements.append(
                        cl.Text(
                            name=f"Documento {i+1}",
                            content=content,
                            display="side"
                        )
                    )
                await cl.Message(
                        content=f"Trovati {len(docs)} documenti.",
                        elements=elements
                    ).send()
                # await cl.Message(
                #     content=f"Altro: {latest_message.content}",
                # ).send()

    # result = agent.invoke(
    # inputs,
    # #config=config,
    # )

    # print(f"Final result: {result}")
    # await cl.Message(
    #     content=f"Response: {result}",
    # ).send()



# Stream the loop's progress
# for chunk in agent.stream(inputs, stream_mode="updates"):
#     print(chunk)


        
    # elif latest_message.tool_calls:
    #     print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")