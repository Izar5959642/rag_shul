"""
RetrievalEvaluatorByVariant — per-variant retrieval evaluation.

Computes Recall@K and MRR independently per variant returned by
ChromaRetriever.retrieve_by_variant(). A "hit" is a result whose
(siman, seif) tuple matches the ground truth.

Standalone — does not import anything from evaluation.retrieval_evaluator.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc)
        return iterable

from .base import BaseEvaluator


DEFAULT_K_VALUES = [1, 3, 5, 10, 18, 30, 50]


def _compute_recall_mrr(ranks: list, k_values: list[int]) -> dict:
    """
    Compute Recall@K and MRR for a single list of ranks.

    Returns string-keyed `recall_at` / `recall_rate` so the result is
    JSON-serializable as-is.
    """
    n = len(ranks)
    recall_at_int = {k: 0 for k in k_values}
    reciprocal    = 0.0
    for r in ranks:
        if r is None:
            continue
        reciprocal += 1.0 / r
        for k in k_values:
            if r <= k:
                recall_at_int[k] += 1
    recall_rate_int = {k: (recall_at_int[k] / n if n else 0.0) for k in k_values}
    mrr = reciprocal / n if n else 0.0
    return {
        "recall_at":   {str(k): v for k, v in recall_at_int.items()},
        "recall_rate": {str(k): v for k, v in recall_rate_int.items()},
        "mrr":         mrr,
        "n_total":     n,
    }


class RetrievalEvaluatorByVariant(BaseEvaluator):
    """
    Per-variant retrieval evaluator.

    For each query, calls retriever.retrieve_by_variant(query, top_k=retrieve_k)
    and computes Recall@K and MRR independently for each variant. A hit is
    matched at the (siman, seif) tuple level — seif numbers reset per siman,
    so both must match.
    """

    @property
    def name(self) -> str:
        return "retrieval_by_variant"

    def __init__(self, k_values=None, retrieve_k=None,
                 query_embeddings_npy=None, query_texts_json=None, **_unused):
        """
        Args from YAML (evaluation section):
            k_values:   list of K values to evaluate (or None = default)
            retrieve_k: how many results to retrieve per variant per call
                        (must be >= max(k_values))
            _unused:    additional YAML fields not relevant to this evaluator
                        (silently accepted — sibling evaluators may set them)
        """
        self.k_values   = sorted(set(k_values)) if k_values else list(DEFAULT_K_VALUES)
        max_k           = max(self.k_values)
        self.retrieve_k = max(retrieve_k or max_k, max_k)
        self._emb_cache  = Path(query_embeddings_npy) if query_embeddings_npy else None
        self._text_cache = Path(query_texts_json)     if query_texts_json     else None

    def evaluate(self, retriever, queries_df, **kwargs) -> dict:
        questions = [str(getattr(row, "question")) for row in queries_df.itertuples(index=False)]

        # Ground truth pre-extracted once
        gt_rows = [
            (int(getattr(row, "siman")), int(getattr(row, "seif")))
            for row in queries_df.itertuples(index=False)
        ]

        # Step 1 — get query embeddings. Load from disk if the cache exists and matches;
        # otherwise encode in-memory (and save the cache if paths are configured).
        query_vecs = None
        if (
            self._emb_cache and self._text_cache
            and self._emb_cache.exists() and self._text_cache.exists()
        ):
            cached_qs = json.loads(self._text_cache.read_text(encoding="utf-8"))
            if cached_qs == questions:
                query_vecs = np.load(str(self._emb_cache))
                print(f"Loaded {len(query_vecs)} query embeddings from cache ({self._emb_cache.name})")

        if query_vecs is None:
            retriever._load()
            texts = [retriever._prefix_query + q for q in questions]
            print(f"Encoding {len(texts)} queries...")
            query_vecs = retriever._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
            )
            if self._emb_cache and self._text_cache:
                np.save(str(self._emb_cache), query_vecs)
                self._text_cache.write_text(
                    json.dumps(questions, ensure_ascii=False), encoding="utf-8"
                )
                print(f"Saved query embedding cache -> {self._emb_cache}")

        # Step 2 — variant-first batched eval, one variant at a time (keeps memory bounded).
        ranks_by_variant: dict[str, list] = {}
        t_start = time.perf_counter()

        retriever._load_collection()
        print(
            f"Running batched evaluation "
            f"({len(retriever._variants)} variants × {len(questions)} queries)..."
        )
        for variant in tqdm(retriever._variants, desc="Variants", unit="variant"):
            variant_results = retriever._query_variant_batch(
                variant, query_vecs, top_k=self.retrieve_k
            )
            ranks: list = []
            for i, (gt_siman, gt_seif) in enumerate(gt_rows):
                rank = next(
                    (r["rank"] for r in variant_results[i]
                     if r["siman"] == gt_siman and r["seif"] == gt_seif),
                    None,
                )
                ranks.append(rank)
            ranks_by_variant[variant] = ranks
            # variant_results goes out of scope here -> GC reclaims before the next variant.

        elapsed_sec = time.perf_counter() - t_start

        metrics_by_variant = {
            v: _compute_recall_mrr(ranks, self.k_values)
            for v, ranks in ranks_by_variant.items()
        }

        return {
            "evaluator":   self.name,
            "granularity": "(siman, seif)",
            "metrics":     metrics_by_variant,
            "n_questions": len(queries_df),
            "elapsed_sec": round(elapsed_sec, 3),
            "extra": {
                "retrieve_k": self.retrieve_k,
                "k_values":   self.k_values,
                "variants":   sorted(metrics_by_variant.keys()),
            },
        }

    def format_report(self, result: dict, retriever_name: str = "",
                      ts_readable: str | None = None, **_meta) -> str:
        metrics  = result["metrics"]
        k_values = result["extra"]["k_values"]
        n_total  = result["n_questions"]
        elapsed  = result["elapsed_sec"]

        if ts_readable is None:
            ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"Run: {ts_readable}",
            f"Evaluator: {self.name} (granularity: (siman, seif))",
            f"Retriever: {retriever_name}",
            f"Questions: {n_total}",
            f"Elapsed:   {elapsed:.2f} sec",
        ]

        for variant in sorted(metrics.keys()):
            v_metrics = metrics[variant]
            v_n       = v_metrics["n_total"]
            lines.append("")
            lines.append(f"-- Variant: {variant} --")
            lines.append("Recall@K:")
            for k in k_values:
                rate  = v_metrics["recall_rate"].get(str(k), 0.0)
                count = v_metrics["recall_at"].get(str(k), 0)
                lines.append(f"  K={k:<3} -> {rate:.4f}  ({count}/{v_n})")
            lines.append("")
            lines.append(f"MRR: {v_metrics['mrr']:.4f}")

        return "\n".join(lines)
