from embedder import VectorStore, E5EmbeddingFunction
from embedder.embed import encode_query
from .base import BaseRetriever


class VectorStoreRetriever(BaseRetriever):
    def __init__(self, store: VectorStore, name: str = "vector_store"):
        self._store = store
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = encode_query(query)
        results = self._store.search(query_vec, k=top_k)

        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "rank":         rank,
                "chunk_id":     chunk_id,
                "score":        round(1.0 - distance, 4),
                "text":         text,
                "siman_parent": int(meta.get("siman", 0)),
            }
            for rank, (chunk_id, text, meta, distance) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            )
        ]
