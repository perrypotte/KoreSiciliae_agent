import asyncio
import os
from dotenv import load_dotenv
from urllib.parse import urljoin
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from langchain_postgres import PGVector
from markdownify import markdownify as md
import httpx
import getpass
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import re
from collections import Counter
import numpy as np

BASE_URL = "https://www.koresiciliae.it/WhatToDo"


# =========================
# CONFIG
# =========================

#RESOURCE_ITEM_SELECTOR = ".col-md-4.col-xs-12.resource-item.k-listview-item"

#RESOURCE_LINK_SELECTOR = "a.fullwidth.link-reset"

#NEXT_BUTTON_SELECTOR = 'a[title="Vai all\'ultima pagina"]'

RESOURCE_BASE_URL = "https://www.koresiciliae.it/search/resource?id="

CONTENT_SELECTOR = "div.content-tab.content-tab-active"

RESOURCE_TYPE_SELECTOR = ".mgtop5.search-tag.tag-large"

TITLE_SELECTOR = ".resource-title h1.mgtop15"

#devo prenderli dal file .env come si fa?
load_dotenv()
if not os.getenv("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="koreSiciliae_resources_v3",
    connection=os.getenv("DATABASE_URL"),
)


# =========================
# HELPERS
# =========================


def normalize_resource_types(resource_type_string):

    resource_type_string = resource_type_string.strip()

    parts = [
        p.strip()
        for p in resource_type_string.split(",")
        if p.strip()
    ]

    if not parts:
        return []

    first_part = parts[0]

    prefix = ""

    if "/" in first_part:

        macro, primary = [
            p.strip()
            for p in first_part.split("/", maxsplit=1)
        ]

        macro = (
            macro
            .replace("à", "a")
            .replace("è", "e")
            .replace("é", "e")
            .replace("ì", "i")
            .replace("ò", "o")
            .replace("ù", "u")
            .replace(" ", "_")
        )

        normalized = [
            f"{macro}_{primary.replace(' ', '_')}"
        ]

        prefix = f"{macro}_"

    else:

        normalized = [
            first_part.replace(" ", "_")
        ]

    for p in parts[1:]:
        normalized.append(
            prefix + p.replace(" ", "_")
        )

    return normalized

def extract_text_features(text):
    
    words = re.findall(r"\w+", text.lower())

    num_words = len(words)

    unique_words = len(set(words))

    avg_word_length = (
        sum(len(w) for w in words) / num_words
        if num_words else 0
    )

    top_words = Counter(words).most_common(20)

    return {
        "num_words": num_words,
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "top_words": top_words
    }

def html_to_markdown(html: str) -> str:
    return md(
        html,
        heading_style="ATX",
        strip=["script", "style"]
    ).strip()

def parse_resource_page(html: str, url: str):
    #TODO OCCHIO HANNOS SCRITTO openResoursePage INVECE DI openResourcePage nella funzione di onclick per aprire la risorsa
    soup = BeautifulSoup(html, "html.parser")

    content_div = soup.select_one(CONTENT_SELECTOR)

    resource_type_el = soup.select_one(RESOURCE_TYPE_SELECTOR)

    title_el = soup.select_one(TITLE_SELECTOR)

    resource_type = (
        resource_type_el.get_text(" ", strip=True)
        if resource_type_el else ""
    )

    resource_types = [
    t.strip()
    for t in resource_type.split(",")
    if t.strip()
]

    title = (
        title_el.get_text(" ", strip=True)
        if title_el else ""
    )

    markdown_content = ""

    if content_div:
        markdown_content = html_to_markdown(
            str(content_div)
        )

    text_features = extract_text_features(markdown_content)

    resource_types = normalize_resource_types(resource_type)

    return {
        "url": url,
        "resource_types": resource_types,
        "title": title,
        "markdown": markdown_content,
        **text_features
                }


# =========================
# MAIN CRAWLER
# =========================

async def crawl_resources():

    resource_urls = set()

    #TODO: dopo vari esperimenti, sembra che la lista sia fatta con Kendo UI + POST AJAX ed in questo caso non conviene usare crawl4ai perché
    # porta sempre alla pagina originale senza aggiornare il contenuto (sembrerebbe dopo vari tentativi).
    # La cosa migliore è fare direttamente le chiamate POST con il payload giusto e ricavare gli url delle risorse dagli ID contenuti nel json
    # della response. 
    resource_urls = set()

    async with httpx.AsyncClient(timeout=30) as client:

        #TODO: non ho capito perché basta iterare su queste due per avere tutte le risorse. Specialmente perché il pageSize è 9.
        for page in range(1, 2):

            payload = {
                "page": page,
                "pageSize": 9,
                "categoryId": 0,
                "subcategoryId": 0,
                "favorite": True,
            }

            print(f"\n[PAGE {page}]")

            response = await client.post(
                BASE_URL+"/ResourcesRead",
                json=payload
            )

            response.raise_for_status()

            json_data = response.json()

            # DEBUG
            # print(json_data)

            # Json start with "Data"
            resources = json_data.get("Data", [])

            print(f"Risorse trovate: {len(resources)}")

            for resource in resources:

                resource_id = resource.get("Id")

                if not resource_id:
                    continue

                resource_url = (
                    f"{RESOURCE_BASE_URL}{resource_id}"
                )

                resource_urls.add(resource_url)

        # =========================
        # RESOURCE PAGES
        # =========================

        documents = []

    #TODO: potevo usare direttamente BeautifulSoup? Crawl4AI è più immediato nelle configurazioni?
    async with AsyncWebCrawler() as crawler:

        for idx, resource_url in enumerate(resource_urls):

            print(f"\n[{idx+1}/{len(resource_urls)}] {resource_url}")

            try:
                result = await crawler.arun(url=resource_url)

                data = parse_resource_page(
                    result.html,
                    resource_url
                )

                documents.append(data)

            except Exception as e:
                print("Errore:", e)


    type_counter = Counter()

    for doc in documents:
        type_counter.update(doc["resource_types"])

    word_counts = [doc["num_words"] for doc in documents]

    stats = {
        "type_counter": dict(type_counter),
        "word_counts_stats": {
            "mean": float(np.mean(word_counts)),
            "variance": float(np.var(word_counts)),
            "std_dev": float(np.std(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "median": float(np.median(word_counts))
        }
    }

    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Statistiche salvate in stats.json")
    return documents


# =========================
# CHUNKING
# =========================

def build_chunks(documents):

    # splitter Markdown per intercettare i '####'
    # TODO: nelle risorse pare che siano tutti "####"
    headers_to_split_on = [("####", "Sezione")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Mantiene i cancelletti nel testo del chunk
    )

    # Se le sezioni sono troppo grandi, ulteriore splitter basato su dimensione chunk
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = []

    for doc in documents:
        # Prima dividiamo secondo la logica markdown
        md_sections = markdown_splitter.split_text(doc["markdown"])

        global_chunk_counter = 0

        for section in md_sections:
            # Per ogni sezione markdown poi ulteriore split se necessario
            text_chunks = recursive_splitter.split_text(section.page_content)

            for chunk in text_chunks:
                # Recuperiamo l'eventuale nome della sezione trovato dal Markdown Splitter
                # Se la sezione non aveva un titolo ####, di default mettiamo una stringa vuota o Generico
                sezione_nome = section.metadata.get("Sezione", "Generico")

                chunks.append({
                    "chunk": chunk,
                    "chunk_id": global_chunk_counter,
                    "url": doc["url"],
                    "title": doc["title"],
                    "resource_types": doc["resource_types"],
                    # AGGIUNTA: Iniettiamo nei metadati del dizionario la sezione logica di appartenenza
                    "section_header": sezione_nome 
                })
                
                global_chunk_counter += 1

    return chunks


# =========================
# RUN
# =========================

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    #TODO: vedere come togliere i metadati default come source che prende il path sul mio pc
    metadata["chunk_id"] = record.get("chunk_id")
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    metadata["resource_types"] = record.get("resource_types")
    metadata["section_header"] = record.get("section_header")
    return metadata

async def main():

    #TODO: Ho fatto solo le risorse. Per la home e altre pagine la struttura va cambiata leggermente perché sono organizzate in modo diverso.
    documents = await crawl_resources()

    chunks = build_chunks(documents)

    print(f"\nChunks creati: {len(chunks)}")

    with open("resources_chunks3.json", "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    from langchain_community.document_loaders import JSONLoader

    loader = JSONLoader(
        file_path="resources_chunks3.json",
        jq_schema='.[]',
        content_key='chunk',
        text_content=False,
        metadata_func=metadata_func
    )

    docs = loader.load()

    #Insert chunks into PGVector
    document_ids = vector_store.add_documents(documents=docs, ids=None)  # Let PGVector generate IDs
    #print(docs[0])


if __name__ == "__main__":
    asyncio.run(main())