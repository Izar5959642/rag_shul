# Evaluation

Runs a retriever against a benchmark CSV and reports retrieval quality (Recall@K, MRR) — flat or per-variant.

---

## Architecture

```
BaseEvaluator (base.py)
    ├── RetrievalEvaluator           — flat eval, unique-siman granularity
    ├── RetrievalEvaluatorByVariant  — per-variant eval, (siman, seif) granularity
    └── LLMEvaluator                 — answer-quality eval (stub, not implemented)
```

All evaluators share one entry point — `evaluation.run(...)` in `runner.py` — which loads the CSV, instantiates the evaluator from `REGISTRY`, runs it, prints the report, and saves both a `.txt` and a `.json` to disk.

---

## Evaluators

### RetrievalEvaluator (`type: retrieval`)

Per-query loop. For each row, calls `retriever.retrieve(query, top_k=retrieve_k)` and finds the ground-truth rank at **unique-siman** granularity (duplicate simanim in the result list are collapsed). Reports Recall@K, MRR, and a single pass/fail flag against `target_recall` at `target_k`.

### RetrievalEvaluatorByVariant (`type: retrieval_by_variant`)

Per-variant batched evaluation. **Hit granularity: (siman, seif)** — both must match the ground truth.

Single code path:
1. **Query embeddings** — if `query_embeddings_npy` + `query_texts_json` both exist and the cached texts match the current questions, load the `.npy`. Otherwise encode in-memory with the retriever's E5 model and write the cache for next time.
2. **Variant-first batched retrieval** — `retriever._load_collection()`, then for each variant in `retriever._variants` call `retriever._query_variant_batch(variant, query_vecs, top_k=retrieve_k)` once. The per-variant result block is released before the next variant loads, so memory stays bounded.

### LLMEvaluator (`type: llm_qa`)

**Stub.** `evaluate()` raises `NotImplementedError`. The constructor accepts `llm_model`, `top_k_context`, and `sleep_between_calls` so YAML keys can stay valid; calling it just fails fast with a message describing the planned BLEU/ROUGE/F1 flow.

---

## Input

Benchmark CSV at `paths.csv_path` (default: `data/eval/sa_eval.csv`).

Required columns after rename: `question`, `siman`, `seif`.

The runner applies a default Hebrew → English rename map before validation:

```python
DEFAULT_COLUMN_MAP = {"שאלה": "question", "סימן": "siman", "סעיף": "seif"}
```

Pass `column_map={}` to `run()` to disable the rename, or pass a custom dict.

---

## Output

`BaseEvaluator.save()` writes two files into `paths.eval_results_dir` (default: `data/eval/results/`):

```
eval_{evaluator_name}_{retriever_name}_{YYYYMMDD_HHMMSS}.txt   # formatted report
eval_{evaluator_name}_{retriever_name}_{YYYYMMDD_HHMMSS}.json  # full result dict
```

**`RetrievalEvaluator.evaluate()` returns:**

```python
{
  "evaluator":     "retrieval",
  "granularity":   "unique-siman",
  "metrics":       {"recall_at": {"1": ...}, "recall_rate": {"1": ...}, "mrr": ..., "n_total": ...},
  "n_questions":   100,
  "elapsed_sec":   12.345,
  "retrieve_k":    50,
  "k_values":      [1, 3, 5, 10, 18, 30, 50],
  "target_k":      50,
  "target_recall": 0.85,
  "target_passed": True,
}
```

**`RetrievalEvaluatorByVariant.evaluate()` returns:**

```python
{
  "evaluator":   "retrieval_by_variant",
  "granularity": "(siman, seif)",
  "metrics":     {"<variant_name>": {"recall_at": {...}, "recall_rate": {...}, "mrr": ..., "n_total": ...}, ...},
  "n_questions": 100,
  "elapsed_sec": 87.421,
  "extra": {
      "retrieve_k": 100,
      "k_values":   [1, 3, 5, 10, 18, 30, 50],
      "variants":   ["text+hagah", "text_only", ...],
  },
}
```

Recall dicts use **string** keys (e.g. `"10"`) so the result is JSON-serializable as-is.

---

## Run (CLI)

The module has no standalone CLI — it is invoked as the last stage of `experiments/exp_main.py`:

```bash
python experiments/exp_main.py
```

That script builds a `chroma` retriever covering all variants, injects the query-embedding cache paths into `evaluation_params`, and calls `evaluation.run(...)`. See [`../experiments/README.md`](../experiments/README.md).

---

## Use as API

```python
from retrievers import get_retriever
from evaluation import run

retriever = get_retriever("chroma", type_text=None)  # all 28 variants

result = run(
    retriever=retriever,
    csv_path="data/eval/sa_eval.csv",
    output_dir="data/eval/results",
    eval_params={
        "type":                 "retrieval_by_variant",
        "k_values":             [1, 3, 5, 10, 18, 30, 50],
        "retrieve_k":           100,
        "max_questions":        None,
        "query_embeddings_npy": "data/eval/query_embeddings.npy",
        "query_texts_json":     "data/eval/query_texts.json",
    },
)
print(result["metrics"]["text+hagah"]["mrr"])
```

---

## Config keys

All keys live under `evaluation:` in `config/config.yaml`. Unknown keys are forwarded to every evaluator's `__init__` (sibling-evaluator keys are silently ignored).

| Key | Default | Used by | Description |
|---|---|---|---|
| `type` | `retrieval_by_variant` | runner | Registry key — picks the evaluator class |
| `k_values` | `[1, 3, 5, 10, 18, 30, 50]` | both retrieval evaluators | K values for Recall@K |
| `retrieve_k` | `max(k_values)` | both retrieval evaluators | Results fetched per query/variant. Forced to `≥ max(k_values)` |
| `max_questions` | `null` | runner | Trim CSV to first N rows (debug); `null` = all |
| `target_k` | `50` | `RetrievalEvaluator` | K at which the pass/fail target is checked |
| `target_recall` | `0.85` | `RetrievalEvaluator` | Recall threshold for pass/fail |
| `llm_model` | `gpt-4o` | `LLMEvaluator` (stub) | Future: which LLM to call |
| `top_k_context` | `3` | `LLMEvaluator` (stub) | Future: chunks per prompt |
| `sleep_between_calls` | `0.3` | `LLMEvaluator` (stub) | Future: throttle between calls |

Paths the eval stage reads (under `paths:`):

| Path key | Default | Purpose |
|---|---|---|
| `csv_path` | `data/eval/sa_eval.csv` | Benchmark CSV |
| `eval_results_dir` | `data/eval/results` | Where `.txt` + `.json` reports are written |
| `query_embeddings_npy` | `data/eval/query_embeddings.npy` | Cached query vectors (per-variant evaluator) |
| `query_texts_json` | `data/eval/query_texts.json` | Cache-validity check — must match the current questions list |

---

## Adding a new evaluator

1. Create `evaluation/my_evaluator.py` — inherit from `BaseEvaluator`, implement `name`, `evaluate(retriever, queries_df, **kwargs)`, and `format_report(result, **meta)`.
2. Register in `evaluation/__init__.py`:
   ```python
   from .my_evaluator import MyEvaluator
   REGISTRY["my_evaluator"] = MyEvaluator
   ```
3. Set `evaluation.type: my_evaluator` in `config/config.yaml`.

### BaseEvaluator contract

```python
class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, retriever, queries_df, **kwargs) -> dict:
        # Return at least: metrics, n_questions, elapsed_sec
        ...

    @abstractmethod
    def format_report(self, result: dict, **meta) -> str: ...

    # Concrete — writes <stem>.txt and <stem>.json into output_dir
    def save(self, result, report_text, output_dir, filename_stem) -> dict: ...
```

Accept `**_unused` in `__init__` and forward unknown YAML keys silently — the runner passes the whole `evaluation:` block as kwargs, including keys meant for sibling evaluators.
