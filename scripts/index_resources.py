import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_postgres import PGVector

# Load environment variables
load_dotenv()


def metadata_func(record: dict, metadata: dict) -> dict:
    """Extract metadata attributes from JSON record."""
    metadata["chunk_id"] = record.get("chunk_id")
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    metadata["resource_types"] = record.get("resource_types", ["Altro"])
    metadata["visit_duration"] = record.get("visit_duration")
    metadata["period"] = record.get("period")
    metadata["latitude"] = record.get("latitude")
    metadata["longitude"] = record.get("longitude")
    metadata["image_url"] = record.get("image_url", "")
    metadata["official_url"] = record.get("official_url", "")
    return metadata


def get_default_json_path() -> str:
    """Find the updated json file in old/ or scripts/."""
    candidates = [
        Path("scripts/resources_chunks6_updated.json"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "old/resources_chunks6_updated.json"


def main():
    parser = argparse.ArgumentParser(
        description="Index updated resources into a new PGVector collection."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=get_default_json_path(),
        help="Path to the JSON file with updated resources (default: scripts/resources_chunks6_updated.json)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.getenv("NEW_COLLECTION_NAME", "koreSiciliae_resources_v8"),
        help="Name of the PGVector collection to create/populate (default: koreSiciliae_resources_v8)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nvidia/nemotron-3-embed-1b",
        help="NVIDIA embedding model name (default: nvidia/nemotron-3-embed-1b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of documents to embed and insert per batch (default: 50)",
    )
    args = parser.parse_args()

    # 1. Check prerequisites
    db_url = os.getenv("DATABASE2_URL")
    if not db_url:
        print("❌ ERROR: DATABASE2_URL environment variable is missing in .env.")
        sys.exit(1)

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        print("⚠️ WARNING: NVIDIA_API_KEY not found in environment. You may be prompted for it.")

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"❌ ERROR: Source JSON file not found at '{json_path}'.")
        sys.exit(1)

    print(f"📂 Source file:      {json_path.resolve()}")
    print(f"📦 Collection name:  {args.collection}")
    print(f"🧠 Embedding model:  {args.embedding_model}")

    # 2. Load JSON into LangChain Documents
    print("\n⏳ Loading documents from JSON...")
    loader = JSONLoader(
        file_path=str(json_path),
        jq_schema=".[]",
        content_key="chunk",
        text_content=False,
        metadata_func=metadata_func,
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents.")

    # 3. Assign unique document_id to each chunk
    ids = [str(uuid.uuid4()) for _ in docs]
    for doc, doc_id in zip(docs, ids):
        doc.metadata["document_id"] = doc_id

    # 4. Initialize Embeddings & PGVector store
    print("\n⏳ Initializing NVIDIA Embeddings & PGVector...")
    embeddings = NVIDIAEmbeddings(model=args.embedding_model)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=args.collection,
        connection=db_url,
    )

    # 5. Insert documents in batches
    total_docs = len(docs)
    batch_size = args.batch_size
    print(f"\n🚀 Ingesting {total_docs} documents into collection '{args.collection}' (batch size: {batch_size})...")

    for i in range(0, total_docs, batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"   Indexed [{min(i + batch_size, total_docs)}/{total_docs}] documents...")

    print(f"\n🎉 Successfully ingested all {total_docs} documents into '{args.collection}'!")

    # 6. Retrieve collection UUID from PostgreSQL
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
                {"name": args.collection},
            )
            col_uuid = result.scalar()
            if col_uuid:
                print("\n" + "=" * 60)
                print(f"📌 COLLECTION CREATED SUCCESSFULLY:")
                print(f"   Collection Name: {args.collection}")
                print(f"   Collection UUID: {col_uuid}")
                print("\n👉 To use this new collection in your agent & SparseRetriever,")
                print("   update your .env or configuration with:")
                print(f"   COLLECTION_ID={col_uuid}")
                print("=" * 60)
    except Exception as e:
        print(f"\n⚠️ Note: Could not query collection UUID directly: {e}")

    # 7. Quick verification search
    try:
        print("\n🔍 Running test similarity search query ('Castello di Lombardia')...")
        sample_results = vector_store.similarity_search("Castello di Lombardia", k=1)
        if sample_results:
            sample = sample_results[0]
            print(f"✅ Search OK! Top result: '{sample.metadata.get('title')}'")
            print(f"   Image URL: {sample.metadata.get('image_url')}")
    except Exception as e:
        print(f"⚠️ Search test skipped: {e}")


if __name__ == "__main__":
    main()
