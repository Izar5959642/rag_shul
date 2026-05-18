# Retrievers

Semantic retrieval layer for the Shulchan Arukh RAG pipeline.

---

## Architecture

```
BaseRetriever (base.py)
    ├── ChromaRetriever           — queries ChromaDB (new, recommended)
    ├── NpyRetriever              — queries CSV + NPY matrix (legacy)
    └── SemanticE5SeifV6Combined  — legacy, hardcoded paths
```

All retrievers implement the same interface (`BaseRetriever`) so the evaluator works with any of them without modification.

---

## ChromaRetriever

Queries the ChromaDB collection built by `embedder/embed.py`.

### How it works

```
query (Hebrew text)
    ↓
encode_query()  — same model + prefix as embed time ("query: ")
    ↓
ChromaDB.query()  — cosine similarity search, filtered by type_text
    ↓
list[dict]  — ranked results with siman, seif, score, type_text
```

### Variants (type_text)

The ChromaDB collection stores **28 text variants**, each labeled with `type_text`. The full list is defined in `config/config.yaml` under `chunker.text_variants` and built by the chunker (see `chunker/README.md`). Representative sample:

| `type_text` | Content |
|-------------|---------|
| `text+hagah` | Original text + Rema commentary |
| `text_only` | Original text only |
| `text+hilchot_group` | Original text + halachic category prefix |
| `text+modern_summary` | Original text + GPT-generated modern summary |
| `text+questions` | Original text + GPT-generated study questions |
| …and 23 more | See `chunker.text_variants` in `config/config.yaml` |

### Usage

```python
from retrievers import get_retriever

# single variant — returns top_k results
r = get_retriever("chroma", type_text="text+hagah")
results = r.retrieve("מה דין ציצית?", top_k=10)

# multiple variants — returns top_k per variant (20 total)
r = get_retriever("chroma", type_text=["text+hagah", "text_only"])
results = r.retrieve("מה דין ציצית?", top_k=10)

# all variants in the collection — top_k per variant (280 total = 28 × 10)
r = get_retriever("chroma", type_text=None)
results = r.retrieve("מה דין ציצית?", top_k=10)

# per-variant results — one clean ranked list per variant (recommended for evaluation)
r = get_retriever("chroma", type_text=None)
results = r.retrieve_by_variant("מה דין ציצית?", top_k=10)
# {"text+hagah": [{rank:1, ...}, ...], "text_only": [{rank:1, ...}, ...], ...}
```

### Result structure

Each result dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Position within the variant (1 = best) |
| `chunk_id` | str | `{type_text}__siman_{N}_seif_{M}` |
| `score` | float | Cosine similarity (0–1, higher = better) |
| `text` | str | Raw chunk text |
| `siman_parent` | int | Siman (chapter) number — used by evaluator |
| `siman` | int | Same as `siman_parent` |
| `seif` | int | Seif (sub-chapter) number |
| `type_text` | str | Which variant this result came from |

### retrieve_by_variant() — per-variant output format

`retrieve_by_variant(query, top_k)` returns `dict[str, list[dict]]` — one key per queried variant (up to 28 when `type_text=None`).

Unlike `retrieve()` which concatenates all variants into one flat list, this method
returns each variant's results separately and independently ranked. Use this when
you want per-variant results without flat concatenation.

**Outer dict:**

| Key | Type | Description |
|-----|------|-------------|
| variant name | `str` | e.g. `"text+hagah"`, `"text_only"`, `"text+modern_summary"` — one entry per queried variant |
| value | `list[dict]` | top_k result dicts, sorted by score descending. `rank` resets to 1 for each variant. |

**Each result dict** — same fields as `retrieve()`:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Position within this variant's list (1 = best). Independent per variant. |
| `chunk_id` | str | `{type_text}__siman_{N}_seif_{M}` |
| `score` | float | Cosine similarity (0–1, higher = better) |
| `text` | str | Raw Hebrew chunk text |
| `siman_parent` | int | Siman (chapter) number |
| `siman` | int | Same as `siman_parent` |
| `seif` | int | Seif (sub-section) number within the siman |
| `type_text` | str | Variant name — always equals the outer dict key |

**Example** (`top_k=2`, two variants):

```json
{
  "text+hagah": [
    {
      "rank": 1,
      "chunk_id": "text+hagah__siman_1_seif_1",
      "score": 0.8661,
      "text": "יתגבר כארי לעמוד בבוקר לעבודת בוראו שיהא הוא מעורר השחר",
      "siman_parent": 1,
      "siman": 1,
      "seif": 1,
      "type_text": "text+hagah"
    },
    {
      "rank": 2,
      "chunk_id": "text+hagah__siman_289_seif_1",
      "score": 0.8424,
      "text": "יהיה שולחנו ערוך ומיטה מוצעת יפה...",
      "siman_parent": 289,
      "siman": 289,
      "seif": 1,
      "type_text": "text+hagah"
    }
  ],
  "text_only": [
    {
      "rank": 1,
      "chunk_id": "text_only__siman_1_seif_1",
      "score": 0.8661,
      "text": "יתגבר כארי לעמוד בבוקר לעבודת בוראו שיהא הוא מעורר השחר",
      "siman_parent": 1,
      "siman": 1,
      "seif": 1,
      "type_text": "text_only"
    },
    {
      "rank": 2,
      "chunk_id": "text_only__siman_289_seif_1",
      "score": 0.8424,
      "text": "יהיה שולחנו ערוך ומיטה מוצעת יפה...",
      "siman_parent": 289,
      "siman": 289,
      "seif": 1,
      "type_text": "text_only"
    }
  ]
}
```

### retrieve_by_variant_vec() — pre-computed vector variant

```python
def retrieve_by_variant_vec(self, vec, top_k: int = 10) -> dict[str, list[dict]]:
```

Same return shape as `retrieve_by_variant`, but takes a pre-computed embedding vector instead of a query string — skips the encode step. Used by the per-variant evaluator, which encodes each query once and reuses the vector across all 28 variants.

---

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type_text` | `"text+hagah"` | Variant(s) to query: `str`, `list[str]`, or `None` (all) |
| `chroma_dir` | `embedder/chroma_db` | ChromaDB directory |
| `collection_name` | `shulchan_arukh_seifs` | Collection name |
| `model` | `intfloat/multilingual-e5-large` | Must match the model used at embed time |
| `prefix_query` | `"query: "` | E5 query prefix |

### top_k vs retrieve_k

- **`top_k`** — results returned per variant per call (what the caller asks for).
- **`retrieve_k`** — set in `config.yaml` under `evaluation.retrieve_k` (default `100`). `RetrievalEvaluatorByVariant.__init__` computes the effective value as `max(retrieve_k or max_k, max_k)` where `max_k = max(k_values)`, so it always covers the largest K used in the Recall@K report.

The evaluator does **not** call `retrieve()` per query. It calls the private workhorse `_query_variant_batch(variant, query_vecs, top_k=retrieve_k)` **once per variant** with all query vectors batched together. The method chunks the batch internally to stay under ChromaDB's SQLite parameter cap. See `_query_variant_batch` in `chroma_retriever.py` and the call site in `evaluation/retrieval_evaluator_by_variant.py`.

---

## Adding a new retriever

1. Create `retrievers/my_retriever.py` — inherit from `BaseRetriever`, implement `name` and `retrieve()`
2. Register in `retrievers/__init__.py`:
   ```python
   from .my_retriever import MyRetriever
   REGISTRY["my_retriever"] = MyRetriever
   ```
3. Run: `python experiments/exp_main.py --retriever my_retriever`

### BaseRetriever contract

```python
class BaseRetriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        # Must return dicts with at least: rank, chunk_id, score, text, siman_parent
        ...
```

---

## Registered retrievers

| Name | Class | Status |
|------|-------|--------|
| `chroma` | `ChromaRetriever` | Active — uses ChromaDB |
| `retrieval_npy` | `NpyRetriever` | Legacy — requires CSV + NPY files |
| `semantic_e5_seif_v6_combined` | `SemanticE5SeifV6CombinedRetriever` | Legacy — hardcoded paths |
