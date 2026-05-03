from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
import numpy as np

from .embed import _get_model, DEFAULT_MODEL, BATCH_SIZE


class E5EmbeddingFunction:
    def __init__(self, model_name=DEFAULT_MODEL, prefix="passage: ", batch_size=BATCH_SIZE):
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size

    def name(self) -> str:
        return "E5EmbeddingFunction"

    def __call__(self, input: list[str]) -> list[list[float]]:
        prefixed = [self.prefix + s for s in input]
        embeddings = _get_model(self.model_name).encode(
            prefixed, batch_size=self.batch_size, normalize_embeddings=True
        )
        return embeddings.tolist()


class VectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(self, query_vector, k: int = 5) -> dict: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def get_existing_ids(self) -> set[str]: ...


class ChromaStore(VectorStore):
    def __init__(
        self,
        chroma_dir: str | Path,
        collection_name: str,
        embedding_function: E5EmbeddingFunction,
    ):
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if embeddings is not None:
            self._collection.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        else:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query_vector, k: int = 5) -> dict:
        q = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        return self._collection.query(query_embeddings=[q], n_results=k)

    def count(self) -> int:
        return self._collection.count()

    def get_existing_ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])


class ManualStore(VectorStore):
    def __init__(self, store_dir: str | Path, embedding_function: E5EmbeddingFunction):
        self._store_dir = Path(store_dir)
        self._ef = embedding_function
        self._embeddings: np.ndarray = np.empty((0, 0))
        self._records: list[dict] = []
        self._loaded: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        emb_path = self._store_dir / "embeddings.npy"
        doc_path = self._store_dir / "documents.json"
        if emb_path.exists() and doc_path.exists():
            self._embeddings = np.load(str(emb_path))
            with open(doc_path, encoding="utf-8") as f:
                self._records = json.load(f)
        else:
            self._embeddings = np.empty((0, 0))
            self._records = []
        self._loaded = True

    def _save(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._store_dir / "embeddings.npy"), self._embeddings)
        with open(self._store_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self._load()
        if embeddings is None:
            new_vecs = np.array(self._ef(documents), dtype=np.float32)
        else:
            new_vecs = np.array(embeddings, dtype=np.float32)

        existing = self.get_existing_ids()
        new_mask = [doc_id not in existing for doc_id in ids]

        new_records = [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": (metadatas[i] if metadatas else {}),
            }
            for i in range(len(ids))
            if new_mask[i]
        ]
        new_vecs_filtered = new_vecs[new_mask]

        self._records.extend(new_records)
        if self._embeddings.shape[1] == 0:
            self._embeddings = new_vecs_filtered
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs_filtered])
        self._save()

    def search(self, query_vector, k: int = 5) -> dict:
        self._load()
        q = np.array(query_vector, dtype=np.float32)
        scores = self._embeddings @ q
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return {
            "ids": [[self._records[i]["id"] for i in top_idx]],
            "documents": [[self._records[i]["document"] for i in top_idx]],
            "metadatas": [[self._records[i]["metadata"] for i in top_idx]],
            "distances": [[float(1.0 - scores[i]) for i in top_idx]],
        }

    def count(self) -> int:
        self._load()
        return len(self._records)

    def get_existing_ids(self) -> set[str]:
        self._load()
        return {r["id"] for r in self._records}
