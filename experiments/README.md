# Experiments

Orchestrator for the full Shulchan Arukh RAG pipeline: data → chunks → embeddings → eval. Single entry point, fully driven by `config/config.yaml`.

---

## The pipeline

`experiments/exp_main.py` runs four stages in order. Each stage is gated by an output-exists check, so re-running only does the work that's missing.

```
Stage 0  Data generation   →  data/processed/shulchan_aruch_rag_with_breadcrumb.json
Stage 1  Chunking          →  data/chunks.json
Stage 2  Embedding         →  embedder/chroma_db_ver_28_tables_clean/
Stage 3  Evaluation        →  data/eval/results/
```

| Stage | Skip rule |
|---|---|
| 0 — Data generation | Skipped if `paths.data_file_with_breadcrumb` already exists. Prints `Data file found at … — skipping data generation.` |
| 1 — Chunking | Skipped if `paths.chunks_json` already exists. Prints `Chunks found at … — skipping chunking.` |
| 2 — Embedding | **Always entered.** Per-variant skipping happens inside `embedder.embed.run` (via `get_existing_type_texts`), which also handles the "added a new variant" case. |
| 3 — Evaluation | Always runs. Output files are timestamped, so reruns don't overwrite. |

Setting `rebuild: true` at the top of `config.yaml` deletes the data file, the chunks file, and the ChromaDB directory before running — i.e. forces a clean rerun of stages 0–2.

---

## Run (CLI)

```bash
python experiments/exp_main.py
```

The script takes **no command-line flags** — every parameter is read from `config/config.yaml`. To point at a different ChromaDB without editing config, set the `CHROMA_DIR` environment variable:

```bash
CHROMA_DIR=/path/to/other_chroma python experiments/exp_main.py
```

---

## What each stage does

| Stage | Function in `exp_main.py` | Underlying entry point | Module README |
|---|---|---|---|
| 0 — Data generation | `_build_data_stage()` | `data/scripts/build_shulchan_aruch_rag.py` → `enrich_with_modern_summary.py` → `add_breadcrumb_to_json.py` | — |
| 1 — Chunking | `_build_chunks_stage()` | `chunker.chunker.run` | [`../chunker/README.md`](../chunker/README.md) |
| 2 — Embedding | `_build_embed_stage()` | `embedder.embed.run` | [`../embedder/README.md`](../embedder/README.md) |
| 3 — Evaluation | `_run_eval()` | builds a `chroma` retriever via `retrievers.get_retriever`, then calls `evaluation.run` | [`../evaluation/README.md`](../evaluation/README.md) |

Refer to each module's README for the details of inputs, outputs, and config keys consumed at that stage. This page only documents the orchestration layer.

---

## Config keys

Keys read directly by `exp_main.py` (everything else under per-module sections is forwarded as-is to the corresponding module).

### Top-level

| Key | Default | Used for |
|---|---|---|
| `log_level` | `INFO` | Python logging level (`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`) |
| `rebuild` | `false` | When `true`, deletes `data_file_with_breadcrumb`, `chunks_json`, and `chroma_dir` before running |

### `paths.*`

| Key | Stage | Role |
|---|---|---|
| `source_txt` | 0 | Raw Torat Emet text (input) |
| `headings_txt` | 0 | Table of contents (input) |
| `summaries_cache` | 0 | Pre-generated modern-summary cache (input) |
| `questions_cache` | 0 | Pre-generated questions cache (input) |
| `data_file_without_additional_breadcrumb` | 0 | Intermediate JSON from `build_rag` |
| `data_file_with_breadcrumb` | 0 | Final stage-0 artifact; presence skips stage 0 |
| `chunks_json` | 1 | Chunker output; presence skips stage 1 |
| `chroma_dir` | 2 | ChromaDB directory (overridable via `CHROMA_DIR` env var) |
| `csv_path` | 3 | Benchmark questions CSV |
| `eval_results_dir` | 3 | Where eval reports (`.txt` + `.json`) are written |
| `query_embeddings_npy` | 3 | Cached query-embedding matrix (read/written by the evaluator) |
| `query_texts_json` | 3 | Companion JSON listing the cached query texts |

### Forwarded sections

`exp_main.py` does not interpret these keys — it just passes them down.

| Key | Forwarded to | Notes |
|---|---|---|
| `chunker.text_variants` | `chunker.chunker.run(..., variants=...)` | Defines the variants emitted into `chunks.json` |
| `embeddings.model` | `embedder.embed.run(..., model=...)` | Embedding model (default `intfloat/multilingual-e5-large`) |
| `embeddings.batch_size` | `embedder.embed.run(..., batch_size=...)` | Default `32` |
| `evaluation` (whole dict) | `evaluation.run(..., eval_params=...)` | `query_embeddings_npy` and `query_texts_json` are injected from `paths.*` before forwarding |

---

## Logging

Configured once at module load via `logging.basicConfig`:

| Setting | Value |
|---|---|
| Level | `cfg["log_level"]` (default `INFO`) |
| Format | `%(asctime)s [%(levelname)s] %(name)s: %(message)s` |
| Date format | `%H:%M:%S` |
| Handler | stdout (no file handler) |
| Logger name | `exp_main` |

To get verbose output, set `log_level: DEBUG` in `config/config.yaml`. Stages 0–2 also use plain `print(...)` for high-level progress (`=== Stage N: … ===`), so those lines appear regardless of log level.

---

## Typical usage

| Goal | How |
|---|---|
| Full clean rerun from scratch | Set `rebuild: true` in `config/config.yaml`, then `python experiments/exp_main.py` |
| Re-run only chunking + downstream | Delete `data/chunks.json`; the data file is reused |
| Re-embed everything | Delete the directory at `paths.chroma_dir` (or use `rebuild: true`) |
| Add a new variant to `chunker.text_variants` | Just re-run — stage 2's per-variant skip logic encodes only the new variant |
| Point at a different ChromaDB without editing config | `CHROMA_DIR=/path/to/other_chroma python experiments/exp_main.py` |
| Re-run only the evaluation | Just re-run — stages 0–2 self-skip when their outputs exist; stage 3 always executes |
