"""
ChromaRetriever — semantic retrieval from a ChromaDB collection
===============================================================
Queries the ChromaDB collection built by embedder/embed.py.
Supports filtering by one variant, multiple variants, or all variants.
When multiple variants are requested, each variant is queried independently
and returns top_k results — so the total returned is top_k × len(variants).

Usage:
    from retrievers import get_retriever

    # single variant
    r = get_retriever("chroma", type_text="text+hagah")
    results = r.retrieve("מה דין ציצית?", top_k=10)

    # multiple variants — 10 results each, flat list
    r = get_retriever("chroma", type_text=["text+hagah", "text_only"])
    results = r.retrieve("מה דין ציצית?", top_k=10)  # 20 total

    # all variants in the collection
    r = get_retriever("chroma", type_text=None)
    results = r.retrieve("מה דין ציצית?", top_k=10)  # 30 total (3 variants × 10)

    # per-variant results — one correctly-ranked list per variant
    r = get_retriever("chroma", type_text=["text+hagah", "text_only"])
    results = r.retrieve_by_variant("מה דין ציצית?", top_k=10)
    # {"text+hagah": [{rank:1, ...}, ...], "text_only": [{rank:1, ...}, ...]}
"""

from pathlib import Path

import chromadb

from .base import BaseRetriever
from embedder.embed import encode_query, _get_model

DEFAULT_MODEL      = "intfloat/multilingual-e5-large"
DEFAULT_CHROMA_DIR = Path(__file__).parent.parent / "embedder" / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh_seifs"


class ChromaRetriever(BaseRetriever):

    @property
    def name(self) -> str:
        return "chroma"

    def __init__(
        self,
        type_text:       "str | list[str] | None" = "text+hagah",
        chroma_dir:      "str | Path" = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model:           str = DEFAULT_MODEL,
        prefix_query:    str = "query: ",
        **_ignored,
    ):
        """
        Args:
            type_text:       variant(s) to query.
                             str   → single variant
                             list  → multiple variants, top_k results each
                             None  → all variants in the collection, top_k each
            chroma_dir:      path to the ChromaDB directory (embedder/chroma_db)
            collection_name: ChromaDB collection name
            model:           embedding model name (must match the one used at embed time)
            prefix_query:    E5 query prefix (default: "query: ")
        """
        self._chroma_dir      = Path(chroma_dir)
        self._collection_name = collection_name
        self._type_text       = type_text
        self._model_name      = model
        self._prefix_query    = prefix_query

        if not self._chroma_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self._chroma_dir}\n"
                f"Run: python embedder/embed.py --chunks data/chunks_siman.json"
            )

        # Lazily loaded
        self._model      = None
        self._collection = None
        self._variants: list[str] | None = None

    def _load_collection(self) -> None:
        """Lazy load ChromaDB collection and variants (no model needed)."""
        if self._collection is not None:
            return

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection = client.get_collection(self._collection_name)

        if self._type_text is None:
            # Paginate to avoid SQLite "too many SQL variables" on large collections.
            # ChromaDB's get() builds a single SQL query per call; with ~80k+ records
            # we exceed SQLite's default ~32k parameter limit. Batches of 5000 stay
            # well below the cap regardless of how the backend constructs the query.
            all_variants: set[str] = set()
            offset, batch = 0, 5000
            while True:
                chunk = self._collection.get(include=["metadatas"], limit=batch, offset=offset)
                metadatas = chunk["metadatas"]
                if not metadatas:
                    break
                all_variants.update(m["type_text"] for m in metadatas if "type_text" in m)
                offset += batch
            self._variants = sorted(all_variants)
        elif isinstance(self._type_text, str):
            self._variants = [self._type_text]
        else:
            self._variants = list(self._type_text)

    def _load(self) -> None:
        """Lazy load model + collection (needed when encoding queries on the fly)."""
        if self._model is not None:
            return
        self._model = _get_model(self._model_name)
        self._load_collection()

    def _query_variant(self, variant: str, vec, top_k: int) -> list[dict]:
        """Run one ChromaDB query for a single variant and return ranked result dicts."""
        raw = self._collection.query(
            query_embeddings=[vec.tolist()],
            n_results=top_k,
            where={"type_text": variant},
            include=["documents", "metadatas", "distances"],
        )
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        return [
            {
                "rank":         rank,
                "chunk_id":     chunk_id,
                "score":        round(1.0 - dist, 4),
                "text":         doc,
                "siman_parent": int(meta["siman"]),
                "siman":        int(meta["siman"]),
                "seif":         int(meta["seif"]),
                "type_text":    meta["type_text"],
            }
            for rank, (chunk_id, doc, meta, dist) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            )
        ]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top_k chunks per variant.

        Returns a flat list — all variants concatenated, each result labeled
        with its type_text. Total results = top_k × len(variants).

        Each result dict contains:
            rank, chunk_id, score, text, siman_parent, siman, seif, type_text
        """
        self._load()
        vec = encode_query(query, model=self._model, prefix_query=self._prefix_query)
        all_results = []
        for variant in self._variants:
            all_results.extend(self._query_variant(variant, vec, top_k))
        return all_results

    def retrieve_by_variant(self, query: str, top_k: int = 10) -> dict[str, list[dict]]:
        """
        Retrieve top_k chunks per variant, returned as a dict keyed by variant name.

        Each variant's list is independently ranked (rank 1-based within the variant).
        Use this instead of retrieve() when you need per-variant metrics.

        Returns:
            {variant_name: [{rank, chunk_id, score, text, siman_parent, siman, seif, type_text}, ...]}
        """
        self._load()
        vec = encode_query(query, model=self._model, prefix_query=self._prefix_query)
        return {variant: self._query_variant(variant, vec, top_k) for variant in self._variants}

    def retrieve_by_variant_vec(self, vec, top_k: int = 10) -> dict[str, list[dict]]:
        """Like retrieve_by_variant but takes a pre-computed embedding vector."""
        self._load_collection()
        return {variant: self._query_variant(variant, vec, top_k) for variant in self._variants}

    def _query_variant_batch(
        self, variant: str, vecs, top_k: int, chunk_size: int = 50
    ) -> list[list[dict]]:
        """Batched query for a single variant.

        Chunks the query vectors internally so ChromaDB's underlying SQLite
        query stays under the parameter cap (same class of issue paginated
        around in `_load_collection`). Public API is unchanged: caller passes
        one (N, dim) array and gets back a list of N ranked-result-lists.
        """
        results: list[list[dict]] = []
        for start in range(0, len(vecs), chunk_size):
            vec_chunk = vecs[start:start + chunk_size]
            raw = self._collection.query(
                query_embeddings=vec_chunk.tolist(),
                n_results=top_k,
                where={"type_text": variant},
                include=["documents", "metadatas", "distances"],
            )
            for ids, documents, metadatas, distances in zip(
                raw["ids"], raw["documents"], raw["metadatas"], raw["distances"]
            ):
                results.append([
                    {
                        "rank":         rank,
                        "chunk_id":     chunk_id,
                        "score":        round(1.0 - dist, 4),
                        "text":         doc,
                        "siman_parent": int(meta["siman"]),
                        "siman":        int(meta["siman"]),
                        "seif":         int(meta["seif"]),
                        "type_text":    meta["type_text"],
                    }
                    for rank, (chunk_id, doc, meta, dist) in enumerate(
                        zip(ids, documents, metadatas, distances), start=1
                    )
                ])
        return results
