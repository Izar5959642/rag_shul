# Experiments

Pipeline scripts for running end-to-end RAG evaluation: chunk → embed → retrieve → evaluate.

---

## Experiment Files

| File | Description | Run |
|---|---|---|
| `exp_main.py` | Full pipeline, NPY backend, CLI args | `python experiments/exp_main.py` |
| `exp_vector_store.py` | VectorStore pipeline, CLI args | `python experiments/exp_vector_store.py --store-type manual` |
| `exp_pipeline.py` | Config-only VectorStore pipeline, no CLI args | `python experiments/exp_pipeline.py` |

### `exp_main.py` CLI args
| Arg | Default | Description |
|---|---|---|
| `--mode {full,mini}` | from config | `full` = real data, `mini` = smoke test |
| `--retriever NAME` | `retrieval_npy` | Retriever key from the registry |
| `--eval-type TYPE` | from config | Override `evaluation.type` |
| `--max-questions N` | from config | Limit eval questions; `all` = use every row |
| `--force-rebuild` | false | Delete existing artifacts and rebuild |
| `--dump-first-query` | false | Write raw retriever output for question 1 to JSON |

### `exp_vector_store.py` CLI args
| Arg | Default | Description |
|---|---|---|
| `--store-type {chroma,manual}` | `chroma` | Which VectorStore backend to use |
| `--max-questions N` | none | Limit eval questions |
| `--force-rebuild` | false | Wipe store and rebuild from scratch |

### `exp_pipeline.py`
No CLI args — all settings come from `config/config.yaml`.

---

## Pipeline Stages

All three scripts share the same logical stages:

```
1. Chunker  → data/chunks_siman.json      (skip if file already exists)
2. Embedder → VectorStore                 (skip already-stored tables)
3. Retriever + Evaluator → report
```

Stages are idempotent: re-running without `--force-rebuild` resumes from where the previous run left off.

---

## Configuration (`config/config.yaml`)

### `paths` — data file locations (relative to project root)
```yaml
paths:
  schema_json: "data/processed/shulchan_aruch_rag_with_breadcrumb.json"
  chunks_json: "data/chunks_siman.json"
  csv_path:    "../data/eval/sa_eval.csv"
```

### `embeddings` — model and encoding settings
```yaml
embeddings:
  model: "intfloat/multilingual-e5-large"
  batch_size: 32
  prefix_passage: "passage: "
  prefix_query:   "query: "
```

### `vector_store` — backend selection (`exp_pipeline.py` and `exp_vector_store.py`)
```yaml
vector_store:
  type: chroma                        # chroma | manual
  chroma_dir: "embedder/chroma_db"
  manual_dir: "embedder/manual_store"
  collection: "shulchan_arukh_seifs"
```

Switch backend: change `vector_store.type` to `manual` — no code change needed.

### `evaluation` — metrics and scope
```yaml
evaluation:
  type: "retrieval"
  k_values: [1, 3, 5, 10, 18, 30, 50]
  target_k: 50
  target_recall: 0.85
  retrieve_k: 100
  max_questions: null   # null = all; set an integer for a quick test
```

Quick test: set `max_questions: 20` to evaluate on 20 questions only.

---

## Output

Reports are saved in the `experiments/` directory alongside the script:

```
experiments/
  exp_results_pipeline_chroma_20250503_120000.txt
  exp_results_pipeline_chroma_20250503_120000.json
```

**Metrics in the report:**
- **Recall@K** — fraction of questions where the gold seif appears in the top K results, for each K in `k_values`
- **MRR** — Mean Reciprocal Rank across all questions
- **Target** — pass/fail against `target_recall` at `target_k`

---

## Quick Start

```bash
# 1. (optional) set backend and question limit in config/config.yaml:
#    vector_store.type: manual
#    evaluation.max_questions: 20

# 2. run
python experiments/exp_pipeline.py
```
