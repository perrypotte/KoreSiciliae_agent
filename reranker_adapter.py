import json
import os

import requests

class TEIReranker:
    def __init__(
        self,
        endpoint="http://localhost:8081",
        top_n=5,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.top_n = top_n

    def rerank(self, query, documents):
        texts = [
            doc.page_content
            for doc in documents
        ]

        response = requests.post(
            f"{self.endpoint}/rerank",
            json={
                "query": query,
                "texts": texts,
                "raw_scores": True,
            },
            timeout=120,
        )

        response.raise_for_status()

        results = response.json()

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True,
        )

        return [
            documents[result["index"]]
            for result in results[:self.top_n]
        ]

class OpenRouterReranker:
    def __init__(
            self,
            endpoint="http://localhost:8081",
            top_n=5,
        ):
            self.endpoint = endpoint.rstrip("/")
            self.top_n = top_n
    
    def rerank(self, query, documents):
        texts = [
            {"text":doc.page_content}
            for doc in documents
        ]

        response = requests.post(
            url="https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                # "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            data=json.dumps({
                "model": "qwen/qwen3-reranker-8b",
                "query": query,
                # Documents can mix images and text. Images are remote URLs or base64 data URIs.
                # "documents": [
                # {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
                # {"text": "A fluffy cat sitting on a windowsill in the sun."},
                # {"text": "A street map of downtown Berlin."}
                # ],
                "documents":texts,
                "top_n": self.top_n
            })
            )

        response.raise_for_status()
        results = response.json()
        # for result in results["results"]:
        #     document = result["document"]
        #     source = document.get("image") or document.get("text")
        #     print(f"Index: {result['index']}, Score: {result['relevance_score']}, Source: {source}")
       # I risultati del reranker sono dentro "results"
        rerank_results = results["results"]

        # print("Reranker results:", response)

        # Ordina gli indici in base al relevance score
        rerank_results = sorted(
            rerank_results,
            key=lambda x: x["relevance_score"],
            reverse=True,
        )

        # Restituisce gli OGGETTI Document originali di LangChain
        return [
            documents[result["index"]]
            for result in rerank_results[:self.top_n]
        ]