# Embedder

Reads the chunker output (`chunks_siman.json`) and stores sentence embeddings in a ChromaDB collection.

---

## Input

`data/chunks_siman.json` — output of the chunker: a JSON array of table objects, one per variant.

```json
[
  {
    "metadata": { "type_text": "text+hagah" },
    "data": [
      { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
      ...
    ]
  },
  {
    "metadata": { "type_text": "text_only" },
    "data": [ ... ]
  },
  {
    "metadata": { "type_text": "text+hilchot_group" },
    "data": [ ... ]
  }
]
```

Each table has:
- `metadata.type_text` — variant name (used as a label in ChromaDB)
- `data` — list of chunks with `siman`, `seif`, `siman_seif`, `text`

---

## Process

For each table:

1. Build encoding text per chunk:
   ```
   "passage: שולחן ערוך אורח חיים, סימן N, סעיף M: <text>"
   ```
2. Encode all texts into normalized 1024-dim vectors using `intfloat/multilingual-e5-large`
3. Store in ChromaDB

All tables are unified into a **single ChromaDB collection**.

---

## Output

`embedder/chroma_db/` — persistent ChromaDB collection named `shulchan_arukh_seifs`.

Each record contains:

| Field | Value | Example |
|-------|-------|---------|
| `id` | `{type_text}__siman_{N}_seif_{M}` | `text+hagah__siman_1_seif_1` |
| `document` | raw text | `"יתגבר כארי לעמוד..."` |
| `embedding` | 1024-dim float32 vector | `[0.0503, 0.0000, ...]` |
| `metadata.siman` | int | `1` |
| `metadata.seif` | int | `1` |
| `metadata.type_text` | str | `"text+hagah"` |

Total records: 3 variants × 4,168 chunks = **12,504 records**.

---

## Run (CLI)

```bash
python embedder/embed.py --chunks data/chunks_siman.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunks` | required | Path to `chunks_siman.json` |
| `--model` | `intfloat/multilingual-e5-large` | Embedding model |
| `--chroma-dir` | `embedder/chroma_db` | ChromaDB directory |
| `--collection` | `shulchan_arukh_seifs` | Collection name |
| `--batch-size` | `32` | Encoding batch size |

---

## Use as API

```python
from embedder.embed import encode_query

# encode a query at retrieval time
query_vec = encode_query("מה דין ציצית?")
# returns: normalized 1024-dim numpy array
```

---

## Query ChromaDB

```python
import chromadb

client = chromadb.PersistentClient("embedder/chroma_db")
col    = client.get_collection("shulchan_arukh_seifs")

# query a specific variant
results = col.query(
    query_embeddings=[query_vec.tolist()],
    n_results=10,
    where={"type_text": "text+hagah"},
)
```

---

## Vector Store (Strategy Pattern)

`embedder/vector_store.py` introduces a Strategy pattern for swappable storage backends. The abstract `VectorStore` interface has two concrete implementations: `ChromaStore` (persists to ChromaDB) and `ManualStore` (persists to `.npy` + `.json` files). Both return results in the same ChromaDB-compatible dict format.

---

### `E5EmbeddingFunction`

```python
E5EmbeddingFunction(model_name=DEFAULT_MODEL, prefix="passage: ", batch_size=BATCH_SIZE)
```

Wraps the project's `intfloat/multilingual-e5-large` model. Prepends `prefix` to each input string before encoding and returns normalized 1024-dim vectors as `list[list[float]]`. The `__call__` signature follows the ChromaDB embedding function protocol (`input` is the exact parameter name).

---

### `VectorStore` interface

```python
class VectorStore(ABC):
    def add_documents(self, documents: list[str], ids: list[str],
                      metadatas: list[dict] | None = None,
                      embeddings: list[list[float]] | None = None) -> None: ...
    def search(self, query_vector, k: int = 5) -> dict: ...
    def count(self) -> int: ...
    def get_existing_ids(self) -> set[str]: ...
```

`add_documents`: if `embeddings` is provided the store uses them directly; otherwise it calls the embedding function to encode `documents`. `search` returns `{ids, documents, metadatas, distances}`.

---

### `ChromaStore`

```python
ChromaStore(chroma_dir: str | Path, collection_name: str, embedding_function: E5EmbeddingFunction)
```

Wraps a persistent ChromaDB collection (cosine similarity space). Use when you want ChromaDB's HNSW index and its query filtering features.

---

### `ManualStore`

```python
ManualStore(store_dir: str | Path, embedding_function: E5EmbeddingFunction)
```

Persists embeddings and documents to disk in two files:

```
<store_dir>/
  embeddings.npy   — (N, 1024) float32, row-indexed
  documents.json   — list of {id, document, metadata} in the same row order
```

Files are loaded lazily on first access and saved once at the end of `add_documents`. Already-stored IDs are skipped on each call, so the store grows incrementally without duplicates.

---

### Usage example

```python
from embedder import ChromaStore, ManualStore, E5EmbeddingFunction

ef = E5EmbeddingFunction()

# ChromaDB backend
store = ChromaStore("embedder/chroma_db", "shulchan_arukh_seifs", ef)
store.add_documents(documents, ids, metadatas, embeddings=vectors)
results = store.search(query_vec, k=5)

# File-based backend
store = ManualStore("embedder/manual_store", ef)
store.add_documents(documents, ids, metadatas, embeddings=vectors)
results = store.search(query_vec, k=5)
```

`results` format (both stores):
```python
{
    "ids":       [["text+hagah__siman_1_seif_1", ...]],
    "documents": [["יתגבר כארי...", ...]],
    "metadatas": [[{"siman": 1, "seif": 1, "type_text": "text+hagah"}, ...]],
    "distances": [[0.12, ...]],   # 1 − cosine_similarity
}
```
