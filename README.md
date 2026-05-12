# rag_shul

RAG pipeline over the Shulchan Arukh (Orach Chaim), with a Hebrew chat UI for comparing retrieval-augmented answers against plain GPT answers.

---

## Pipeline Overview

```
Preprocess Data  →  Chunker  →  Embedder  →  Retriever  →  Chat UI / Evaluation
 data/scripts/      chunker/    embedder/    retrievers/    chat-ui/   evaluation/
```

Each stage is an independent module. Config lives in `config/config.yaml`.

---

## Quick Start — Chat UI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp chat-ui/.env.example chat-ui/.env
# edit chat-ui/.env and set OPENAI_API_KEY=sk-...

# 3. Build the ChromaDB (first time only — takes ~2 hours)
python3 embedder/embed.py --chunks data/chunks_siman.json

# 4. Run the local server
cd chat-ui
python3 server.py
```

Open [http://localhost:3000](http://localhost:3000)

---

## Chat UI — Features

Three tabs:

| Tab | Description |
|-----|-------------|
| **שאלות מאגר** | Browse and search 600 eval questions. Click "השווה" to load a question into the comparison tab. |
| **השוואה** | Side-by-side: GPT-4o-mini without RAG vs GPT-4o-mini with RAG (Shulchan Arukh). Adjustable K (1/3/5/10). Shows retrieved chunks + Recall@K/MRR when question is from the eval set. Vote which answer is better. |
| **סטטיסטיקות** | Aggregated vote counts and retrieval metrics (Recall@K, MRR). Persisted in localStorage. |

### Chat UI Files

| File | Purpose |
|------|---------|
| `chat-ui/server.py` | Local dev server — serves static files and routes API calls |
| `chat-ui/index.html` | Frontend — three-tab comparison UI |
| `chat-ui/api/chat.py` | POST `/api/chat` — calls ChromaRetriever then OpenAI |
| `chat-ui/api/eval.py` | GET `/api/eval` — serves the eval CSV as JSON |
| `chat-ui/.env` | API key file (not committed) |
| `chat-ui/.env.example` | Template |

---

## Stage 1 — Preprocess Data (`data/scripts/`)

Converts the raw source text into a structured JSON file ready for chunking. Run the two scripts in order.

### Script A — `build_shulchan_aruch_rag.py`

Parses the raw Torat Emet TXT file and produces the canonical RAG JSON.

**Input:** `data/source_original/data_fixed.txt`

**Output:** `data/processed/shulchan_aruch_rag.json`

```json
{
  "title": "שולחן ערוך, אורח חיים",
  "simanim": [
    {
      "siman": 1,
      "seifim": [{ "seif": 1, "text": "...", "hagah": "..." }]
    }
  ]
}
```

```bash
python data/scripts/build_shulchan_aruch_rag.py
python data/scripts/build_shulchan_aruch_rag.py --test   # regression tests
```

### Script B — `add_breadcrumb_to_json.py`

Adds `hilchot_group` and `siman_sign` headings to each siman.

**Output:** `data/processed/shulchan_aruch_rag_with_breadcrumb.json`

```bash
python data/scripts/add_breadcrumb_to_json.py
```

---

## Stage 2 — Chunker (`chunker/`)

Reads the RAG JSON and produces a multi-variant chunks file for the embedder.

**Input:** `data/processed/shulchan_aruch_rag.json`

**Output:** `data/chunks_siman.json` — array of variant tables, each with metadata and chunk list:

```json
[
  {
    "metadata": { "type_text": "text+hagah" },
    "data": [{ "siman": 1, "seif": 1, "text": "..." }]
  }
]
```

Three variants are built: `text+hagah`, `text_only`, `text+hilchot_group`.

```bash
python -m chunker.main
```

**Python API:**

```python
from chunker import load_schema, build_tables

schema = load_schema("data/processed/shulchan_aruch_rag.json")
tables = build_tables(schema)
```

---

## Stage 3 — Embedder (`embedder/`)

Encodes all chunks into dense vector embeddings and stores them in ChromaDB.

**Model:** `intfloat/multilingual-e5-large` (1024-dim, L2-normalized)

**E5 prefix convention:** `"passage: "` for corpus, `"query: "` at retrieval time

**Input:** `data/chunks_siman.json` (output of Stage 2)

**Output:** `embedder/chroma_db/` — ChromaDB collection `shulchan_arukh_seifs`

Current corpus: **12,504 records** (3 variants × 4,168 chunks from 688 simanim)

```bash
python3 embedder/embed.py --chunks data/chunks_siman.json

# with explicit model / collection:
python3 embedder/embed.py \
    --chunks data/chunks_siman.json \
    --model intfloat/multilingual-e5-large \
    --collection shulchan_arukh_seifs
```

Skip logic: already-embedded variants are detected and skipped automatically.

**Python API:**

```python
from embedder.embed import encode_query, _get_model

model = _get_model("intfloat/multilingual-e5-large")
vec = encode_query("מה דין ציצית?", model=model, prefix_query="query: ")
```

---

## Stage 4 — Retriever (`retrievers/`)

**Implemented:** `ChromaRetriever` — queries the ChromaDB collection built in Stage 3.

```python
from retrievers import get_retriever

# single variant (default)
r = get_retriever("chroma", type_text="text+hagah")
results = r.retrieve("מה דין ציצית?", top_k=5)

# multiple variants — top_k results each
r = get_retriever("chroma", type_text=["text+hagah", "text_only"])
results = r.retrieve("מה דין ציצית?", top_k=5)  # 10 total

# all variants
r = get_retriever("chroma", type_text=None)
results = r.retrieve("מה דין ציצית?", top_k=5)  # 15 total
```

Each result dict:

```python
{
  "rank": 1,
  "chunk_id": "text+hagah__siman_1_seif_1",
  "score": 0.87,        # cosine similarity (1 - L2 distance)
  "text": "...",
  "siman": 1,
  "seif": 1,
  "type_text": "text+hagah"
}
```

The retriever loads lazily — the embedding model and ChromaDB client are initialized on the first `retrieve()` call.

---

## Stage 5 — Evaluation (`evaluation/`)

**Input:** `data/eval/sa_eval.csv` — 600 questions with correct siman/seif labels

**Metrics:** Recall@K, MRR (computed per-question against the correct siman)

**`max_questions`** (config) — limits the number of questions evaluated. Set to an integer (e.g. `100`) for a quick smoke-test; `null` runs all questions.

```python
from evaluation import get_evaluator
from retrievers import get_retriever

retriever = get_retriever("chroma", type_text="text+hagah")
evaluator = get_evaluator("retrieval", retriever=retriever)
report = evaluator.evaluate("data/eval/sa_eval.csv", top_k=5)
```

Interactive per-question evaluation is also available through the Chat UI's **השוואה** tab.

---

## Config (`config/config.yaml`)

All pipeline settings in one place:

```yaml
paths:
  data_file: "data/processed/shulchan_aruch_rag.json"
  chunks_json: "data/chunks_siman.json"

chunker:
  mode: seif            # seif | siman | sliding_window

embeddings:
  model: intfloat/multilingual-e5-large
  batch_size: 32

retrieval:
  top_k: 5
  type_text: "text+hagah"

evaluation:
  k_values: [1, 3, 5, 10]
  target_recall: 0.8
  max_questions: null   # null = all questions; integer = limit for quick smoke-test
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `openai`, `chromadb`, `sentence-transformers`, `torch`, `pyyaml`
