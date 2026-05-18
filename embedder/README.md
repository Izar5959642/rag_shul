# Embedder

Reads the chunker output (`chunks_siman.json`) and stores sentence embeddings in a ChromaDB collection.

---

## Input

`data/chunks_siman.json` — output of the chunker: a JSON array of table objects, one per variant. The chunker currently emits **28 variants** (see `chunker.text_variants` in `config/config.yaml`). Representative shape:

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
  }
  // …and 26 more variant tables
]
```

Each table has:
- `metadata.type_text` — variant name (used as a label in ChromaDB)
- `data` — list of chunks with `siman`, `seif`, `siman_seif`, `text`

---

## Process

`run()` (and the CLI) does, per `chunks_siman.json`:

1. **Check what's already embedded** — call `get_existing_type_texts()` and skip any variant already present in the target collection. If every variant is already there, exit early with `All tables already embedded. Nothing to do.`
2. **Build encoding text** per chunk: `"passage: " + chunk["text"]`. The `"passage: "` prefix is required by E5 models; any breadcrumb / heading inside `chunk["text"]` is contributed by the chunker, not the embedder.
3. **Encode** all texts into normalized 1024-dim float32 vectors using `intfloat/multilingual-e5-large` (configurable).
4. **Store** in ChromaDB — `store_in_chroma()` gets-or-creates the collection (cosine space) and `add()`s every `(id, embedding, document, metadata)` row.

All variants are unified into a **single ChromaDB collection**.

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

Total records: **N variants × M chunks** — currently **28 variants** (see `chunker.text_variants` in `config/config.yaml`). The exact total depends on the current chunker output (`collection.count()` reports it after the run).

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

## Config keys

The `embedder/embed.py` CLI is self-contained (flags above). The keys below live under `embeddings:` in `config/config.yaml` and are consumed by `experiments/exp_main.py:_build_embed_stage` when the pipeline drives the embed stage.

| Key | Default | Used by | Description |
|---|---|---|---|
| `model` | `intfloat/multilingual-e5-large` | `_build_embed_stage` | SentenceTransformer model name passed through to `run()` |
| `batch_size` | `32` | `_build_embed_stage` | Encoding batch size passed through to `embed()` |
| `prefix_passage` | `"passage: "` | `build_encoding_texts` | Prefix prepended to each chunk before encoding (E5 requires `"passage: "`) |
| `prefix_query` | `"query: "` | `encode_query` | Prefix prepended to the query at retrieval time (E5 requires `"query: "`) |
| `enrich_fields` | `[]` | (forwarded to chunker) | Extra chunker fields concatenated into the embedded text — backwards-compatible default is empty |
| `enrich_separator` | `" \| "` | (forwarded to chunker) | Separator joining `enrich_fields` into one string |
| `rebuild` | (anchor) | `_build_embed_stage` | If `true`, the experiment deletes the existing `chroma_dir` before embedding (see `experiments/README.md`) |
| `chunk_file` | (anchor) | `_build_embed_stage` | Path to `chunks_siman.json` — usually the YAML anchor `&chunk_file` |
| `embeddings_file` | (anchor) | `_build_embed_stage` | Path to the `.npy` cache used by the legacy `retrieval_npy` retriever |

---

## Use as API

`embedder/__init__.py` exports `encode_query` and `load_tables`. The full programmatic entry point `run()` is available from `embedder.embed`.

```python
from embedder import encode_query

# encode a query at retrieval time
query_vec = encode_query("מה דין ציצית?")
# returns: normalized 1024-dim float32 numpy array
```

`encode_query` also accepts the optional kwargs used by the retriever / evaluator:

```python
from sentence_transformers import SentenceTransformer
from embedder import encode_query

# pass a pre-loaded model to avoid reloading ~500MB on every call
model = SentenceTransformer("intfloat/multilingual-e5-large")
vec = encode_query("מה דין ציצית?", model=model, prefix_query="query: ")
```

Drive the whole embed stage programmatically (mirrors what `experiments/exp_main.py` does):

```python
from pathlib import Path
from embedder.embed import run

run(
    chunks_json=Path("data/chunks_siman.json"),
    chroma_dir=Path("embedder/chroma_db"),
    model="intfloat/multilingual-e5-large",
    collection="shulchan_arukh_seifs",
    batch_size=32,
)
# Skips any variant already present in the collection.
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
