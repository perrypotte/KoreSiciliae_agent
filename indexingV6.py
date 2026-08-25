import asyncio
import os
from dotenv import load_dotenv
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from langchain_postgres import PGVector
from markdownify import markdownify as md
from langchain_text_splitters import MarkdownHeaderTextSplitter
import httpx
import getpass
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import re
from collections import Counter
import numpy as np


# ============================================================
# BASE URL
# ============================================================

BASE_URL = "https://www.koresiciliae.it/WhatToDo"


# ============================================================
# CONFIG
# ============================================================

RESOURCE_BASE_URL = "https://www.koresiciliae.it/search/resource?id="

CONTENT_SELECTOR = "div.content-tab.content-tab-active"

RESOURCE_TYPE_SELECTOR = ".mgtop5.search-tag.tag-large"

TITLE_SELECTOR = ".resource-title h1.mgtop15"


# ============================================================
# ENV
# ============================================================

load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = getpass.getpass(
        "Enter API key for NVIDIA: "
    )


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b"
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="koreSiciliae_resources_v7",
    connection=os.getenv("DATABASE2_URL"),
)


# ============================================================
# HELPERS
# ============================================================

def normalize_resource_types(resource_type_string):

    resource_type_string = resource_type_string.strip()

    if not resource_type_string:
        return ["Altro"]

    parts = [
        p.strip()
        for p in resource_type_string.split(",")
        if p.strip()
    ]

    if not parts:
        return ["Altro"]

    first_part = parts[0]

    prefix = ""

    if "/" in first_part:

        macro, primary = [
            p.strip()
            for p in first_part.split(
                "/",
                maxsplit=1
            )
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

    words = re.findall(
        r"\w+",
        text.lower()
    )

    num_words = len(words)

    unique_words = len(
        set(words)
    )

    avg_word_length = (
        sum(
            len(w)
            for w in words
        ) / num_words
        if num_words
        else 0
    )

    top_words = Counter(
        words
    ).most_common(20)

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
        strip=[
            "script",
            "style"
        ]
    ).strip()


# ============================================================
# DURATA MEDIA VISITA
# ============================================================

def extract_visit_duration(soup):

    label = soup.find(
        "span",
        class_="weight-700",
        string=lambda text: (
            text
            and "Durata media della visita" in text
        )
    )

    if not label:
        return None

    value_span = label.find_next_sibling(
        "span"
    )

    if not value_span:
        return None

    value = value_span.get_text(
        " ",
        strip=True
    )

    return value if value else None


def extract_opening_periods(markdown_content):

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    sections = splitter.split_text(
        markdown_content
    )

    for section in sections:

        header_values = [
            value.lower().strip()
            for key, value in section.metadata.items()
            if key.startswith("header_")
        ]

        if any(
            "periodi e orari di apertura" in header
            for header in header_values
        ):

            return section.page_content.strip()

    return None


# ============================================================
# COORDINATE
# ============================================================

def extract_coordinates(html):

    pattern = re.compile(
        r"L\.latLng\s*\(\s*"
        r"parseFloat\(['\"]([-+]?\d+(?:\.\d+)?)['\"]\)"
        r"\s*,\s*"
        r"parseFloat\(['\"]([-+]?\d+(?:\.\d+)?)['\"]\)"
    )

    match = pattern.search(
        html
    )

    if not match:
        return None, None

    latitude = float(
        match.group(1)
    )

    longitude = float(
        match.group(2)
    )

    return latitude, longitude


# ============================================================
# IMAGE URL
# ============================================================

def extract_image_url(soup):

    image_div = soup.select_one(
        ".iframe-main-img"
    )

    if not image_div:
        return ""

    style = image_div.get(
        "style",
        ""
    )

    match = re.search(
        r"background-image\s*:\s*url\(['\"]?(.*?)['\"]?\)",
        style,
        re.IGNORECASE
    )

    if not match:
        return ""

    return match.group(1).strip()


# ============================================================
# SITO UFFICIALE
# ============================================================

def extract_official_url(soup):

    # Prima prova a cercare direttamente
    # un link con testo "Sito ufficiale"

    official_link = soup.find(
        "a",
        href=True,
        string=lambda text: (
            text
            and "Sito ufficiale" in text
        )
    )

    if official_link:
        return official_link.get(
            "href",
            ""
        ).strip()

    # Fallback:
    # cerca tutti i tag <a> e controlla
    # il testo contenuto al loro interno.

    for a in soup.find_all(
        "a",
        href=True
    ):

        text = a.get_text(
            " ",
            strip=True
        )

        if "Sito ufficiale" in text:

            return a.get(
                "href",
                ""
            ).strip()

    return ""


# ============================================================
# PARSE RESOURCE PAGE
# ============================================================

def parse_resource_page(
    html: str,
    url: str
):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    # ========================================================
    # CONTENUTO PRINCIPALE
    # ========================================================

    content_div = soup.select_one(
        CONTENT_SELECTOR
    )


    # ========================================================
    # TIPO RISORSA
    # ========================================================

    resource_type_el = soup.select_one(
        RESOURCE_TYPE_SELECTOR
    )

    resource_type = (
        resource_type_el.get_text(
            " ",
            strip=True
        )
        if resource_type_el
        else ""
    )

    resource_types = normalize_resource_types(
        resource_type
    )


    # ========================================================
    # TITOLO
    # ========================================================

    title_el = soup.select_one(
        TITLE_SELECTOR
    )

    title = (
        title_el.get_text(
            " ",
            strip=True
        )
        if title_el
        else ""
    )


    # ========================================================
    # HTML -> MARKDOWN
    # ========================================================

    markdown_content = ""

    if content_div:

        markdown_content = html_to_markdown(
            str(content_div)
        )


    # ========================================================
    # DURATA MEDIA VISITA
    #
    # Viene cercata direttamente
    # nell'HTML tramite lo span.
    # ========================================================

    visit_duration = extract_visit_duration(
        soup
    )


    # ========================================================
    # PERIODO E ORARI DI APERTURA
    #
    # È già contenuto nel Markdown
    # della pagina.
    # ========================================================

    period = extract_opening_periods(
        markdown_content
    )


    # ========================================================
    # LATITUDINE E LONGITUDINE
    #
    # Cercate nel codice sorgente
    # JavaScript della pagina.
    # ========================================================

    latitude, longitude = extract_coordinates(
        html
    )


    # ========================================================
    # IMAGE URL
    #
    # Viene estratta dal:
    #
    # <div class="iframe-main-img"
    #      style="background-image: url(...)">
    #
    # Se non presente viene restituita
    # una stringa vuota.
    # ========================================================

    image_url = extract_image_url(
        soup
    )


    # ========================================================
    # SITO UFFICIALE
    #
    # Viene cercato il link:
    #
    # <a ...>
    #     Sito ufficiale
    # </a>
    #
    # Se non presente viene restituita
    # una stringa vuota.
    # ========================================================

    official_url = extract_official_url(
        soup
    )


    # ========================================================
    # FEATURE TESTUALI
    # ========================================================

    text_features = extract_text_features(
        markdown_content
    )


    # ========================================================
    # RISULTATO
    # ========================================================

    return {

        "url": url,

        "resource_types": resource_types,

        "title": title,

        "markdown": markdown_content,

        # ====================================================
        # Metadata
        # ====================================================

        "visit_duration": visit_duration,

        "period": period,

        "latitude": latitude,

        "longitude": longitude,

        "image_url": image_url,

        "official_url": official_url,

        # ====================================================
        # Feature testuali
        # ====================================================

        **text_features
    }


# ============================================================
# MAIN CRAWLER
# ============================================================

async def crawl_resources():

    resource_urls = set()


    # ========================================================
    # RECUPERO LISTA RISORSE
    # ========================================================

    async with httpx.AsyncClient(
        timeout=30
    ) as client:


        for page in range(1, 2):

            payload = {

                "page": page,

                "pageSize": 9,

                "categoryId": 0,

                "subcategoryId": 0,

                "favorite": True,
            }


            print(
                f"\n[PAGE {page}]"
            )


            response = await client.post(

                BASE_URL + "/ResourcesRead",

                json=payload
            )


            response.raise_for_status()


            json_data = response.json()


            resources = json_data.get(
                "Data",
                []
            )


            print(
                f"Risorse trovate: {len(resources)}"
            )


            for resource in resources:

                resource_id = resource.get(
                    "Id"
                )


                if not resource_id:
                    continue


                resource_url = (
                    f"{RESOURCE_BASE_URL}"
                    f"{resource_id}"
                )


                resource_urls.add(
                    resource_url
                )


    # ========================================================
    # RESOURCE PAGES
    # ========================================================

    documents = []


    async with AsyncWebCrawler() as crawler:

        for idx, resource_url in enumerate(
            resource_urls
        ):

            print(
                f"\n[{idx + 1}/{len(resource_urls)}] "
                f"{resource_url}"
            )


            try:

                result = await crawler.arun(
                    url=resource_url
                )


                data = parse_resource_page(
                    result.html,
                    resource_url
                )


                print(
                    f"Titolo: {data['title']}"
                )

                print(
                    f"Durata: {data['visit_duration']}"
                )

                print(
                    f"Periodo: {data['period']}"
                )

                print(
                    f"Latitudine: {data['latitude']}"
                )

                print(
                    f"Longitudine: {data['longitude']}"
                )

                print(
                    f"Image URL: {data['image_url']}"
                )

                print(
                    f"Sito ufficiale: {data['official_url']}"
                )


                documents.append(
                    data
                )


            except Exception as e:

                print(
                    "Errore:",
                    e
                )


    # ========================================================
    # STATISTICHE
    # ========================================================

    type_counter = Counter()


    for doc in documents:

        type_counter.update(
            doc["resource_types"]
        )


    word_counts = [

        doc["num_words"]

        for doc in documents
    ]


    stats = {

        "type_counter":
            dict(type_counter),

        "word_counts_stats": {

            "mean":
                float(
                    np.mean(
                        word_counts
                    )
                ),

            "variance":
                float(
                    np.var(
                        word_counts
                    )
                ),

            "std_dev":
                float(
                    np.std(
                        word_counts
                    )
                ),

            "min":
                int(
                    np.min(
                        word_counts
                    )
                ),

            "max":
                int(
                    np.max(
                        word_counts
                    )
                ),

            "median":
                float(
                    np.median(
                        word_counts
                    )
                )
        }
    }


    # ========================================================
    # SALVATAGGIO STATISTICHE
    # ========================================================

    with open(
        "stats.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Statistiche salvate in stats.json"
    )


    return documents


# ============================================================
# CHUNKING
# ============================================================

def build_chunks(
    documents
):

    chunks = []


    for doc in documents:

        chunks.append({

            # =================================================
            # Contenuto usato per embedding
            # =================================================

            "chunk":
                doc["markdown"],


            # =================================================
            # Metadata
            # =================================================

            "chunk_id":
                0,

            "url":
                doc["url"],

            "title":
                doc["title"],

            "resource_types":
                doc["resource_types"],

            "visit_duration":
                doc["visit_duration"],

            "period":
                doc["period"],

            "latitude":
                doc["latitude"],

            "longitude":
                doc["longitude"],

            "image_url":
                doc["image_url"],

            "official_url":
                doc["official_url"],
        })


    return chunks


# ============================================================
# METADATA FUNCTION
# ============================================================

def metadata_func(
    record: dict,
    metadata: dict
) -> dict:

    metadata["chunk_id"] = record.get(
        "chunk_id"
    )

    metadata["url"] = record.get(
        "url"
    )

    metadata["title"] = record.get(
        "title"
    )

    metadata["resource_types"] = record.get(
        "resource_types"
    )

    metadata["visit_duration"] = record.get(
        "visit_duration"
    )

    metadata["period"] = record.get(
        "period"
    )

    metadata["latitude"] = record.get(
        "latitude"
    )

    metadata["longitude"] = record.get(
        "longitude"
    )

    metadata["image_url"] = record.get(
        "image_url"
    )

    metadata["official_url"] = record.get(
        "official_url"
    )

    return metadata


# ============================================================
# RUN
# ============================================================

async def main():

    # ========================================================
    # CRAWLING
    # ========================================================

    documents = await crawl_resources()


    # ========================================================
    # CHUNKS
    # ========================================================

    chunks = build_chunks(
        documents
    )


    print(
        f"\nChunks creati: {len(chunks)}"
    )


    # ========================================================
    # SALVATAGGIO JSON
    # ========================================================

    with open(
        "resources_chunks6.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # JSON LOADER
    # ========================================================

    from langchain_community.document_loaders import JSONLoader


    loader = JSONLoader(

        file_path=
            "resources_chunks6.json",

        jq_schema=
            ".[]",

        content_key=
            "chunk",

        text_content=
            False,

        metadata_func=
            metadata_func
    )


    docs = loader.load()


    # ========================================================
    # DOCUMENT IDS
    # ========================================================

    import uuid


    ids = [

        str(
            uuid.uuid4()
        )

        for _ in docs
    ]


    for doc, doc_id in zip(
        docs,
        ids
    ):

        doc.metadata[
            "document_id"
        ] = doc_id


    # ========================================================
    # PGVECTOR
    # ========================================================

    vector_store.add_documents(

        documents=docs,

        ids=ids
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )