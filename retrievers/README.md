# Retrievers

Provides retrieval classes used by the evaluation pipeline — every retriever wraps a backend and exposes a uniform `retrieve()` interface.

---

## Interface

Every retriever inherits from `BaseRetriever` and implements two members:

```python
@property
@abstractmethod
def name(self) -> str:
    """Unique experiment name (e.g. 'retrieval_npy', 'vector_store_chroma')."""

@abstractmethod
def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
    """Return the top_k most relevant chunks for a Hebrew query string."""
```

Each dict in the returned list contains:

| Key | Type | Description |
|---|---|---|
| `rank` | `int` | Position in results (1 = most relevant) |
| `chunk_id` | `str` or `int` | Unique chunk identifier |
| `score` | `float` | Cosine similarity (higher = better) |
| `text` | `str` | Raw chunk text |
| `siman_parent` | `int` | Siman (chapter) number this chunk belongs to |

---

## Available Retrievers

| Registry key | Class | Backend | Config / injection |
|---|---|---|---|
| `retrieval_npy` | `NpyRetriever` | `.npy` matrix + CSV | `chunks_csv`, `embeddings_npy` kwargs |
| `vector_store` | `VectorStoreRetriever` | Any `VectorStore` | `store` injected at construction |

All classes are importable from the `retrievers` package:

```python
from retrievers import NpyRetriever, VectorStoreRetriever, get_retriever
```

---

## NpyRetriever

Loads a pre-built CSV + `.npy` matrix and retrieves via dot-product similarity.

**Constructor:**
```python
NpyRetriever(
    chunks_csv:     str | Path,   # flat CSV with siman, seif, text columns
    embeddings_npy: str | Path,   # (N, D) float32 matrix, row-aligned with CSV
    model:          str = "intfloat/multilingual-e5-large",
    prefix_query:   str = "query: ",
)
```

**How it works:**
1. Validates file existence at `__init__` (fail fast before any heavy work)
2. Loads CSV and `.npy` lazily on the first `retrieve()` call
3. Encodes the query via `embed.encode_query` (same model + prefix used for passages)
4. Computes `scores = embeddings @ query_vec`, returns top-k via `argpartition`

**When to use:** pre-built artifacts already exist on disk and the corpus is static (no incremental growth needed).

---

## VectorStoreRetriever

Wraps any `VectorStore` backend (ChromaStore or ManualStore) and delegates search to it.

**Constructor:**
```python
VectorStoreRetriever(store: VectorStore, name: str = "vector_store")
```

- `store` — any `VectorStore` instance (ChromaStore or ManualStore from `embedder`)
- `name` — override to distinguish backends in evaluation logs (e.g. `"vector_store_manual"`)

**How it works:**
1. Encodes the query via `embed.encode_query`
2. Calls `store.search(query_vec, k=top_k)` — returns ChromaDB-format dict
3. Converts distance → similarity: `score = round(1.0 - distance, 4)`

**Example:**
```python
from embedder import ChromaStore, ManualStore, E5EmbeddingFunction
from retrievers.vector_store_retriever import VectorStoreRetriever

ef = E5EmbeddingFunction()

# ChromaDB backend
store = ChromaStore("embedder/chroma_db", "shulchan_arukh_seifs", ef)
retriever = VectorStoreRetriever(store, name="vector_store_chroma")

# or NPY+JSON backend
store = ManualStore("embedder/manual_store", ef)
retriever = VectorStoreRetriever(store, name="vector_store_manual")

results = retriever.retrieve("מה דין ציצית?", top_k=5)
```

The retriever is backend-agnostic — swapping ChromaStore for ManualStore requires no other code change.

---

## Adding a New Retriever

1. Create `retrievers/my_retriever.py` inheriting `BaseRetriever`:
   ```python
   from .base import BaseRetriever

   class MyRetriever(BaseRetriever):
       @property
       def name(self) -> str:
           return "my_retriever"

       def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
           ...
   ```

2. Register it in `retrievers/__init__.py`:
   ```python
   from .my_retriever import MyRetriever

   REGISTRY = {
       ...
       "my_retriever": MyRetriever,
   }
   ```

3. Use it via `get_retriever`:
   ```python
   retriever = get_retriever("my_retriever", **kwargs)
   ```
