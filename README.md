# 🧭 Design and development of an agentic system for building itineraries in Sicily

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain%20%7C%20LangGraph-orange)](https://www.langchain.com/)
[![Chainlit](https://img.shields.io/badge/Frontend-Chainlit-red)](https://docs.chainlit.io/)
[![PGVector](https://img.shields.io/badge/Database-PostgreSQL%2016%20%2B%20PGVector-336791)](https://github.com/pgvector/pgvector)
[![Valhalla](https://img.shields.io/badge/Routing-Valhalla%20(Sicily%20OSM)-green)](https://github.com/valhalla/valhalla)

> **Master's Thesis Project (*Tesi di Laurea Magistrale Sperimentale*)**  
> **Author:** Andrea Perricone  
> **Institution:** Università degli studi di Catania  
> **Territory Focus:** Enna and Sicily only, Italy

---

## 📖 Overview

This is an intelligent conversational recommendation and multi-day itinerary planning system designed for the province of Enna, Sicily. It bridges **Agentic LLM Orchestration**, **Hybrid Retrieval-Augmented Generation (Hybrid RAG)**, and **real-world geospatial routing (Valhalla)** to provide travelers with personalized recommendations and realistic day-by-day travel plans.

Unlike standard chatbots that hallucinate unrealistic travel routes, this system enforces **strict physical travel feasibility constraints**: travel durations, distance limits, pedestrian vs. driving transport modes, visit durations, and round-trip return times to user accommodation.

---

## 🌟 Key Features

- **Dual-Mode Conversational Flow**:
  - **Free Discovery Mode (`planning = false`)**: Natural language discovery of points of interest (POIs), food, culture, and events without strict planning restrictions.
  - **Structured Planning Mode (`planning = true`)**: Step-by-step interactive construction of a multi-day itinerary respecting travel time, distance limits, transport mode, and daily hour budgets.
- **Strict Real-World Travel Constraints**:
  - **Transport Modes**: Driving (`auto`) or walking (`pedestrian`).
  - **Time & Distance Budgets**: Daily hour limits, configurable maximum hop distance between POIs, visit duration estimates, and return travel time to lodging.
- **Hybrid Retrieval Pipeline (Dense + Sparse + Reranker)**:
  - **Dense Retrieval**: PostgreSQL with `pgvector` and NVIDIA Nemotron embeddings (`nvidia/nemotron-3-embed-1b`).
  - **Sparse Retrieval**: BM25 inverted index over PostgreSQL documents.
  - **Rank Fusion**: Reciprocal Rank Fusion (RRF) combining dense and lexical results.
  - **Re-ranking**: Cross-encoder reranking (via TEI, OpenRouter, or NVIDIA NIM APIs).
- **Geospatial Intelligence & Routing**:
  - **Valhalla Routing Engine**: Self-hosted Docker container loaded with OpenStreetMap (OSM) data for Sicily (`isole-latest.osm.pbf`).
  - **Precomputed NxN Route Matrix**: Fast SQL lookups (`route_matrix` table) for pairwise POI travel times and distances.
  - **Nominatim Geocoding**: Resolves arbitrary street addresses or landmarks to coordinates with rate-limit compliance.
- **Dynamic User Profiling (`ProfileMiddleware`)**:
  - A background classifier continuously updates user interest categories (e.g., archaeology, food/wine, nature, museums) across conversation turns and transparently injects them into the agent prompt.
- **Interactive Web UI (Chainlit + React)**:
  - Custom React element (`PlaceCarousel.jsx`) rendered inline in Chainlit, offering visual POI cards with images, travel metrics, and direct action buttons (*"Add to Itinerary"*, *"Get Info"*).

---

## ⚙️ Prerequisites

Before getting started, make sure you have the following installed:

1. **Python**: Version `3.12.2` recommended.
2. **Docker**: To run PostgreSQL (PGVector) and Valhalla routing engine.
3. **Hardware / LLM Service**:
   - **Local Inference (Default)**: An OpenAI-compatible server (such as [llama.cpp server](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [Ollama](https://ollama.ai), or [LM Studio](https://lmstudio.ai)) running `unsloth/Qwen3.5-9B-GGUF` at `http://127.0.0.1:8888/v1`.
   - **Cloud Inference (Alternative)**: An API key for [OpenRouter](https://openrouter.ai/) or [NVIDIA NIM](https://build.nvidia.com/).
4. **API Keys**:
   - `NVIDIA_API_KEY`: Required for generating dense embeddings (`nvidia/nemotron-3-embed-1b`).
   - `OPENROUTER_API_KEY` / `OPENAI_API_KEY`: Required if using cloud LLM providers or cloud rerankers.

---

## 🚀 Quickstart guide

Follow these steps to set up and run the project from scratch.

### Step 1: Clone the repository & set up virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate

# Install required Python dependencies
pip install -r requirements.txt
```

### Step 2: Configure environment variables

Edit `.env` with your settings.

### Step 3: Launch supporting infrastructure with docker

Start PostgreSQL with `pgvector` and the Valhalla routing service:

```bash
docker compose up -d
```

> [!NOTE]  
> On the first start, the `valhalla` container will automatically download the OpenStreetMap extract for the Italian Islands / Sicily (`isole-latest.osm.pbf`) from Geofabrik into `./valhalla_data` and build routing graph tiles. This initial download and build process may take several minutes depending on your internet connection.

Verify that both containers are running healthy:

```bash
docker compose ps
```

Ensure the `vector` extension is enabled in PostgreSQL:

```bash
docker exec -it pgvector-db psql -U postgres -d mydb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 4: Index the Knowledge Base & build the route matrix

If you are setting up a fresh database, populate the POI vector store and precalculate travel distances between places:

1. **Ingest POIs into PGVector**:
   ```bash
   python scripts/index_resources.py --file scripts/resources_chunks6_updated.json --collection koreSiciliae
   ```
> [!NOTE]  
> Due to a major overhaul of the koresiciliae.it platform, POI images may not be visible during the chat.


2. **Precompute the Route Distance/Time Matrix**:
   ```bash
   python create_route_matrix.py
   ```
   *This script queries indexed POIs and calls Valhalla to store pairwise travel times and distances in the `route_matrix` table for both `auto` (driving) and `pedestrian` (walking) modes.*

### Step 5: Launch the Chainlit application

Start the interactive assistant using Chainlit:

```bash
chainlit run agent.py -w
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🤖 LLM configuration

The application is configured in `agent.py` to use a local or remote OpenAI-compatible endpoint. You can toggle between providers:

### Option A: Local LLM Server (Default)

In `agent.py`:

```python
model = ChatOpenAI(
    base_url="http://127.0.0.1:8888/v1",
    api_key=os.getenv("OPENAI_API_KEY", "local"),
    model="unsloth/Qwen3.5-9B-GGUF",
    temperature=0.2,
    top_p=0.4,
    streaming=True,
    reasoning={"effort": "low", "summary": "concise"}
)
```

### Option B: Cloud Provider (OpenRouter or NVIDIA NIM)

Uncomment the relevant lines in `agent.py`:

```python
# OpenRouter configuration
model = ChatOpenRouter(
    model="qwen/qwen-2.5-72b-instruct",
    temperature=0.2,
    top_p=0.4,
    streaming=True,
)
```

---

## 🛠️ Available agent tools

The agent is equipped with custom tools designed for the tourist workflow:

| Tool | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `search_places` | Hybrid retrieval of POIs with metadata filtering and RRF ranking | `queries: list[str]`, `filter: dict` |
| `get_place_info` | Detailed information lookup for a specific POI | `document_id: str`, `title: str` |
| `update_planning_constraints` | Initializes or updates itinerary parameters | `days: int`, `transport: "auto" \| "pedestrian"`, `Daily_hours: float`, `Max_distance_km: float`, `starting_place`, `ending_place` |
| `add_place_to_itinerary` | Validates time/distance feasibility and appends POI to current day | `document_id: str`, `place_name: str`, `visit_duration: float`, `travel_time: float`, `travel_time_to_accomodation: float` |
| `finish_day` | Closes the current day schedule and initializes the next day | `ending_place: str` |
| `finish_itinerary` | Formats the final multi-day itinerary and resets planning state | *None* |

---

## 📊 Evaluation & User Study

As part of the Master's Thesis research, the system underwent an extensive empirical user study evaluating **Usability**, **Agent Capabilities**, **Content Quality**, and **Overall User Satisfaction**.

The evaluation tools in `userStudy/` allow reproducing the analytical figures:

```bash
# Generate Likert boxplots across dimensions
python userStudy/boxplot.py

# Generate radar / spider graph for capability metrics
python userStudy/spidergraph.py
```
---
