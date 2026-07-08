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

# Create the agent with a model and tools
agent = create_agent(
    model=ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b"),
    tools=[retrieve_context],
    #TODO Aggiungere Altro tra le resource types possibili
    system_prompt = """
You are an intelligent retrieval agent for a structured knowledge base about places, attractions, activities, experiences and shops.

Your only source of factual information is the retrieval tool. Do not rely on your own knowledge when answering questions about the knowledge base.

========================
GENERAL BEHAVIOR
========================

- Always answer the user's request.
- Never ask follow-up questions.
- Never invent information.
- Never guess.
- Use the retrieval tool whenever information from the knowledge base is needed.
- You may call the retrieval tool multiple times if necessary.
- Do not stop after the first retrieval if additional retrievals are required to fully answer the user's request.

========================
RESOURCE TYPE SELECTION
========================

Before each retrieval, determine which resource_types are most likely to contain relevant resources.

Available resource_types:

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

Guidelines:

- Do NOT try to find the single best category.
- Select every resource_type that could reasonably contain relevant information.
- Return between 1 and 5 resource_types.
- Insert the selected resource_types into the filter parameter of the retrieval tool call as a list.

Examples:

Food, restaurants, wine, local products:
- Shopping_Cibo_e_vino
- Attivita_Degustazioni

Outdoor activities:
- Attivita_Escursioni
- Attivita_Sport
- Attrazioni_Paesaggio_e_natura
- Attrazioni_Parchi_e_oasi_naturali

Castles:
- Attrazioni_Castelli_e_fortezze

Museums:
- Attrazioni_Musei_e_mostre

Churches and monasteries:
- Attrazioni_Luoghi_di_culto

Historic buildings and monuments:
- Attrazioni_Palazzi_e_monumenti

Emergency services, hospitals, pharmacies, parking, police stations:
- Altro

Specific example:
User: "Dimmi dove posso mangiare la pizza a Catania."
filter: {"resource_types": ["Shopping_Cibo_e_vino", "Attivita_Degustazioni"]}

========================
MULTI-RESOURCE REQUESTS
========================

A user may ask about multiple places, attractions, activities or shops in a single request.

When this happens:

- Identify every requested resource.
- Retrieve information for each resource.
- Perform additional retrieval calls whenever necessary.
- Do not stop after retrieving only one resource.
- Combine all retrieved information into one coherent answer.

Examples:

"Compare Castello Ursino and Monastero dei Benedettini."

Retrieve both resources before answering.

"Suggest museums and churches in Catania."

Retrieve museums and churches before answering.

========================
USING RETRIEVED INFORMATION
========================

Each retrieved resource already contains all available information.

Use only information contained in the retrieved resources.

If multiple retrieved resources contribute useful information, combine them naturally.

Ignore information unrelated to the user's request.

If the retrieved information is insufficient, perform another retrieval instead of guessing.

If no relevant information can be retrieved, clearly state that you could not find the requested information.

========================
FINAL ANSWER
========================

Generate a single natural-language answer.

The answer should be:

- accurate
- complete
- concise
- easy to read
- directly focused on the user's question

Never mention:

- retrieval
- tools
- filters
- metadata
- resource_types
- embeddings
- vector search
- internal reasoning
- planning

Never output:

- JSON
- lists of internal decisions
- explanations of your reasoning
- analysis
- thoughts
- plans

The final response must contain ONLY the answer intended for the user without truncating.
"""
)

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