# rag_shul

RAG pipeline over the **Shulchan Arukh (Orach Chaim)** — Hebrew halachic corpus. Four-stage pipeline (data → chunks → embeddings → eval) plus a Hebrew RTL chat UI that compares GPT answers with and without retrieval, scored on a 600-question benchmark with Recall@K and MRR.

---

## Pipeline

```
Stage 0 Data    →   Stage 1 Chunk   →   Stage 2 Embed   →   Stage 3 Eval
data/scripts/       chunker/            embedder/           evaluation/
                                            │
                                            ▼
                              retrievers/  ←  chat-ui/   (runtime consumers)
```

Single entry point: `python experiments/exp_main.py`. Each stage self-skips when its output already exists, so re-running only does the missing work. See [experiments/README.md](experiments/README.md).

---

## Modules

| Module | What it does | README |
|---|---|---|
| `experiments/` | Pipeline orchestrator; reads `config/config.yaml` and runs all four stages with skip-if-exists gating. | [experiments/README.md](experiments/README.md) |
| `chunker/` | Reads the RAG JSON and produces `chunks_siman.json` — one table per text variant (currently 28). | [chunker/README.md](chunker/README.md) |
| `embedder/` | Encodes each chunk with `intfloat/multilingual-e5-large` and stores 1024-d vectors in ChromaDB. | [embedder/README.md](embedder/README.md) |
| `retrievers/` | Semantic retrieval over the ChromaDB collection — single-variant, multi-variant, and per-variant batched paths. | [retrievers/README.md](retrievers/README.md) |
| `evaluation/` | Runs a retriever over a benchmark CSV and reports Recall@K + MRR (flat or per-variant). | [evaluation/README.md](evaluation/README.md) |
| `chat-ui/` | Hebrew RTL web UI — compares GPT-4o-mini with vs. without RAG; vote, browse eval questions, see live retrieval. | [chat-ui/README.md](chat-ui/README.md) |
| `data/` | Source TXT, processed JSON, eval CSV, modern-summary / questions caches, and the `data/scripts/` preprocessors. | _(no README yet)_ |
| `config/` | Single source of truth — `config/config.yaml` drives every module. | _(no README)_ |

---

## Quickstart

```bash
# 1. install deps
pip install -r requirements.txt

# 2. (chat UI only) set your OpenAI key
cp chat-ui/.env.example chat-ui/.env
# edit chat-ui/.env → OPENAI_API_KEY=sk-...

# 3. run the full pipeline (data → chunks → embed → eval).
#    First run builds ChromaDB (~2 h on CPU; minutes on a GPU runtime).
#    Subsequent runs skip stages whose outputs already exist.
python experiments/exp_main.py

# 4. (optional) launch the chat UI locally
python run_chat.py
# → http://localhost:3000
```

Re-run a single stage by deleting its output and invoking `exp_main.py` again. Full clean rerun: set `rebuild: true` in `config/config.yaml`. The full recipe table (re-chunk, re-embed, add a new variant, point at a different ChromaDB) lives in [experiments/README.md](experiments/README.md).

---

## Config

All settings live in `config/config.yaml`. Top-level keys and which module each one drives:

| Key | Consumer |
|---|---|
| `rebuild` | `experiments/exp_main.py` — wipes the data file, the chunks file, and the ChromaDB directory |
| `log_level` | `experiments/exp_main.py` — Python `logging` level |
| `paths` | every module — single registry of file locations |
| `chunker` | `chunker.chunker.run` — `mode`, `chunk_fields`, `siman_fields`, `text_variants` |
| `embeddings` | `embedder.embed.run` — `model`, `batch_size`, `prefix_passage`, `prefix_query` |
| `retrieval` | `retrievers/` — default `top_k`, default `type_text` |
| `query` | legacy — kept for backward compat with older retriever paths |
| `evaluation` | `evaluation.run` — `type`, `k_values`, `retrieve_k`, `max_questions`, … |

Per-key detail lives in each module's `## Config keys` section.

---

## Repository layout

```
rag_shul/
├── chat-ui/        # Hebrew RTL web UI (local dev + Vercel deploy)
├── chunker/        # RAG JSON → multi-variant chunks
├── config/         # config.yaml — single source of truth
├── data/           # source text + processed JSON + eval CSV
│   ├── eval/       # sa_eval.csv + cached query embeddings + results/
│   ├── processed/  # canonical RAG JSON (+ breadcrumb variant)
│   ├── scripts/    # build_shulchan_aruch_rag.py, enrich_with_modern_summary.py, add_breadcrumb_to_json.py
│   └── source_original/   # raw Torat Emet TXT
├── embedder/       # ChromaDB builder + encode_query() API
├── evaluation/     # Recall@K / MRR evaluator + runner
├── experiments/    # exp_main.py — full-pipeline orchestrator
├── notebooks/      # Colab notebooks (rebuild + eval on a GPU runtime)
├── private/        # Manager / Worker session folders (project-owner scoped)
└── retrievers/     # BaseRetriever + ChromaRetriever (+ legacy retrievers)
```

---

## Development workflow

- **Branches**: feature branches off `main` (current dev branch: `Izar_Dahan__new_ref_evaluation`).
- **`chunker/` is protected**: never edit any file inside `chunker/` without explicit confirmation — see `chunker/CLAUDE.md`.
- **Notebooks**: `notebooks/` holds the Colab cells used to rebuild ChromaDB on a GPU runtime; the persistent ChromaDB lives on Google Drive.
- **Manager / Worker sessions**: multi-session refactor work is tracked under `private/Manager-*/` (each with its own `README.md` + `progress.md`).

---

## Requirements

```bash
pip install -r requirements.txt
```

Key deps: `openai`, `chromadb`, `sentence-transformers`, `torch`, `pyyaml`, `pandas`, `numpy`.
