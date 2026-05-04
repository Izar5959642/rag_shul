"""
exp_pipeline.py — Full RAG pipeline, all text-variant models.
All settings from config/config.yaml — no CLI arguments.

Runs ChromaRetriever once per text variant (text+hagah, text_only,
text+hilchot_group) and prints a comparison table at the end.

Run:
    python experiments/exp_pipeline.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunker.chunker import load_schema, build_tables
from embedder.embed import (
    load_tables,
    build_encoding_texts,
    embed,
    store_in_chroma,
    get_existing_type_texts,
    _get_model,
)
from retrievers.chroma_retriever import ChromaRetriever
from evaluation import get_evaluator

HERE        = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

embed_cfg   = cfg["embeddings"]
eval_cfg    = cfg["evaluation"]
paths_cfg   = cfg["paths"]
chunker_cfg = cfg["chunker"]

SCHEMA_JSON   = (ROOT / paths_cfg["data_file"]).resolve()
CHUNKS_JSON   = (ROOT / paths_cfg["chunks_json"]).resolve()
EVAL_CSV      = (ROOT / paths_cfg["csv_path"]).resolve()
CHROMA_DIR    = ROOT / "embedder" / "chroma_db"
COLLECTION    = "shulchan_arukh_seifs"
TEXT_VARIANTS = [v["type_text"] for v in chunker_cfg["text_variants"]]


# ── helpers ───────────────────────────────────────────────────────────────────

def get_chunks() -> list:
    """Load chunks.json, building it first if missing."""
    if not CHUNKS_JSON.exists():
        print(f"[1/3 chunker]  BUILD — {CHUNKS_JSON.name}")
        schema = load_schema(SCHEMA_JSON)
        tables = build_tables(schema)
        CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_JSON.write_text(
            json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"[1/3 chunker]  SKIP  — {CHUNKS_JSON.name} exists")
    return load_tables(CHUNKS_JSON)


def build_chroma_store(tables: list) -> None:
    """Embed and store each variant in ChromaDB, skipping already-embedded ones."""
    already = get_existing_type_texts(CHROMA_DIR, COLLECTION)
    model   = _get_model(embed_cfg["model"])
    prefix  = embed_cfg.get("prefix_passage", "passage: ")
    batch   = embed_cfg.get("batch_size", 32)

    new_tables = []
    for type_text, chunks in tables:
        if type_text in already:
            print(f"[2/3 embed]    SKIP  — [{type_text}]")
            continue
        print(f"[2/3 embed]    BUILD — [{type_text}]  {len(chunks)} chunks")
        texts   = build_encoding_texts(chunks, prefix_passage=prefix)
        vectors = embed(model, texts, batch_size=batch)
        new_tables.append((type_text, chunks, vectors.tolist()))

    if new_tables:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        store_in_chroma(new_tables, CHROMA_DIR, COLLECTION)
    else:
        print(f"[2/3 embed]    DONE  — all variants already in store")


def load_queries() -> pd.DataFrame:
    """Load and normalize the eval CSV columns."""
    df = pd.read_csv(EVAL_CSV)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("שאלה", "question", "query"): col_map[col] = "question"
        elif lc in ("סימן", "siman"):           col_map[col] = "siman"
        elif lc in ("סעיף", "seif"):            col_map[col] = "seif"
    return df.rename(columns=col_map)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Pipeline:  ChromaRetriever × {len(TEXT_VARIANTS)} variants")
    print(f"Variants:  {', '.join(TEXT_VARIANTS)}")
    print("=" * 60)

    # 1. chunks
    tables = get_chunks()

    # 2. chroma store
    build_chroma_store(tables)

    # 3. queries
    queries_df = load_queries()
    max_q = eval_cfg.get("max_questions")
    if max_q:
        queries_df = queries_df.head(max_q).reset_index(drop=True)
    print(f"[3/3 eval]     LOAD  — {len(queries_df)} questions")

    # 4. evaluator (shared across all variants)
    eval_type   = eval_cfg.get("type", "retrieval")
    eval_kwargs = {k: v for k, v in eval_cfg.items() if k not in ("type", "max_questions")}
    evaluator   = get_evaluator(eval_type, **eval_kwargs)

    # 5. run each variant
    run_ts      = datetime.now()
    ts_filename = run_ts.strftime("%Y%m%d_%H%M%S")
    ts_readable = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    all_results = {}
    for variant in TEXT_VARIANTS:
        retriever_name = f"chroma_{variant}"
        print(f"\n{'─' * 60}")
        print(f"Variant: {variant}")
        print("─" * 60)

        retriever = ChromaRetriever(
            type_text=variant,
            chroma_dir=CHROMA_DIR,
            collection_name=COLLECTION,
            model=embed_cfg["model"],
            prefix_query=embed_cfg.get("prefix_query", "query: "),
        )

        result = evaluator.evaluate(retriever, queries_df)
        result["variant"]        = variant
        result["retriever"]      = retriever_name
        result["timestamp"]      = ts_readable
        all_results[variant]     = result

        report = evaluator.format_report(
            result,
            retriever_name=retriever_name,
            ts_readable=ts_readable,
        )
        print("\n" + report)

        stem  = f"exp_results_pipeline_{variant.replace('+', '_')}_{ts_filename}"
        saved = evaluator.save(result, report, output_dir=HERE, filename_stem=stem)
        print(f"Report -> {saved['txt'].name}")
        print(f"JSON   -> {saved['json'].name}")

    # 6. comparison table
    target_k = eval_cfg.get("target_k", 50)
    print(f"\n{'=' * 60}")
    print(f"COMPARISON — Recall@{target_k} + MRR")
    print("=" * 60)
    print(f"{'Variant':<30}  {'Recall@'+str(target_k):>12}  {'MRR':>8}")
    print("-" * 56)
    for variant, result in all_results.items():
        recall = result["metrics"]["recall_rate"].get(str(target_k), 0.0)
        mrr    = result["metrics"]["mrr"]
        print(f"{variant:<30}  {recall:>12.4f}  {mrr:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
