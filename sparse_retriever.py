import os
from collections import defaultdict
from typing import List, Optional

from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class SparseRetriever:
    def __init__(
        self, 
        db_url: Optional[str] = None, 
        collection_id: str = "a8d572b6-ac8b-4d21-9132-6d2c808e2d6a", #Collection V5
        rrf_k: int = 20
    ):
        self.db_url = db_url or os.getenv("DATABASE2_URL")
        if not self.db_url:
            raise ValueError("DATABASE2_URL non trovato nell'ambiente o non fornito.")
            
        self.engine = create_engine(self.db_url)
        self.collection_id = collection_id
        self.rrf_k = rrf_k
        
        self.documents: List[Document] = []
        self._bm25: Optional[BM25Retriever] = None

    def load_documents(self) -> List[Document]:
        """Carica tutti i documenti dal database Postgres per la collezione definita."""
        docs = []
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT document, cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                """),
                {"collection_id": self.collection_id},
            )

            for row in rows:
                docs.append(
                    Document(
                        page_content=row.document,
                        metadata=row.cmetadata or {}
                    )
                )

        self.documents = docs
        self._build_bm25(self.documents)
        return self.documents

    def _build_bm25(self, docs: List[Document]):
        """Inizializza o aggiorna l'istanza BM25 con la lista di documenti data."""
        if docs:
            self._bm25 = BM25Retriever.from_documents(docs)
        else:
            self._bm25 = None

    def filter_documents(
        self, 
        documents: List[Document], 
        resource_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Filtra i documenti in base ai resource_types nei metadata."""
        if not resource_types:
            return documents

        filtered_docs = []
        for doc in documents:
            doc_types = doc.metadata.get("resource_types", [])

            if isinstance(doc_types, str):
                doc_types = [doc_types]

            if any(rt in resource_types for rt in doc_types):
                filtered_docs.append(doc)

        return filtered_docs

    def reciprocal_rank_fusion(
        self, 
        rankings: List[List[Document]]
    ) -> List[Document]:
        """Unisce e riordina multiple liste di risultati tramite Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        documents = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                # Utilizziamo document_id se presente, altrimenti fallback sull'hash del contenuto
                doc_id = doc.metadata.get("document_id") or hash(doc.page_content)

                scores[doc_id] += 1 / (self.rrf_k + rank + 1)
                documents[doc_id] = doc

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [documents[doc_id] for doc_id, _ in ranked]

    def invoke(
        self, 
        query: str, 
        k: int = 5, 
        resource_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Esegue la ricerca sparse (BM25) con opzione di filtraggio dinamico.
        
        Se viene specificato resource_types, crea un BM25 temporaneo sul sottoinsieme 
        filtrato per garantire che 'k' restituisca solo risultati validi.
        """
        if not self.documents:
            #TODO: valutare computazionalmente e spazialmente questa cosa
            self.load_documents()

        # Se dobbiamo filtrare, ricreiamo un BM25 temporaneo per non inquinare gli score
        if resource_types:
            filtered_docs = self.filter_documents(self.documents, resource_types)
            if not filtered_docs:
                return []
            temp_bm25 = BM25Retriever.from_documents(filtered_docs)
            temp_bm25.k = k
            return temp_bm25.invoke(query)

        # Altrimenti usiamo l'indice BM25 globale già pronto
        if self._bm25:
            self._bm25.k = k
            return self._bm25.invoke(query)

        return []