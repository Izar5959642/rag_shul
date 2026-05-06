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

The ChromaDB collection stores 3 text variants, each labeled with `type_text`:

| `type_text` | Content |
|-------------|---------|
| `text+hagah` | Original text + Rema commentary |
| `text_only` | Original text only |
| `text+hilchot_group` | Original text + halachic category prefix |

### Usage

```python
from retrievers import get_retriever

# single variant — returns top_k results
r = get_retriever("chroma", type_text="text+hagah")
results = r.retrieve("מה דין ציצית?", top_k=10)

# multiple variants — returns top_k per variant (20 total)
r = get_retriever("chroma", type_text=["text+hagah", "text_only"])
results = r.retrieve("מה דין ציצית?", top_k=10)

# all variants in the collection — top_k per variant (30 total)
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

`retrieve_by_variant(query, top_k)` returns `dict[str, list[dict]]`.

Unlike `retrieve()` which concatenates all variants into one flat list, this method
returns each variant's results separately and independently ranked. Use this for
evaluation — it avoids the ranking bias introduced by flat concatenation.

**Outer dict:**

| Key | Type | Description |
|-----|------|-------------|
| variant name | `str` | e.g. `"text+hagah"`, `"text_only"`, `"text+hilchot_group"` |
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

- **`top_k`** — results returned per variant (what the user sees)
- **`retrieve_k`** — results fetched for evaluation purposes (set in `config.yaml`)

The evaluator calls `retrieve(query, top_k=retrieve_k)` and deduplicates by `siman_parent` internally.

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
