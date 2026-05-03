"""
exp_vector_store.py — Full RAG pipeline using the VectorStore Strategy Pattern
===============================================================================
Runs the complete pipeline with the new storage abstraction:
  1. Chunker  : source JSON       → data/chunks_siman.json
  2. Embedder : chunks_siman.json → VectorStore (ChromaDB or NPY+JSON)
  3. Retriever: VectorStoreRetriever wrapping the store
  4. Evaluator: RetrievalEvaluator (Recall@K, MRR)

Usage:
    python experiments/exp_vector_store.py
    python experiments/exp_vector_store.py --store-type manual
    python experiments/exp_vector_store.py --store-type chroma
    python experiments/exp_vector_store.py --force-rebuild
    python experiments/exp_vector_store.py --max-questions 50
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedder import E5EmbeddingFunction, ChromaStore, ManualStore
from embedder.embed import load_tables, build_encoding_texts, embed, _get_model
from retrievers.vector_store_retriever import VectorStoreRetriever
from evaluation import get_evaluator

HERE        = Path(__file__).parent
CONFIG_PATH = HERE / "../config/config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SCHEMA_JSON     = ROOT / "data" / "processed" / "shulchan_aruch_rag_with_breadcrumb.json"
CHUNKS_JSON     = ROOT / "data" / "chunks_siman.json"
EVAL_CSV        = ROOT / "data" / "eval" / "sa_eval.csv"
DEFAULT_CHROMA  = ROOT / "embedder" / "chroma_db"
DEFAULT_MANUAL  = ROOT / "embedder" / "manual_store"
COLLECTION_NAME = "shulchan_arukh_seifs"


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def ensure_chunks_json() -> None:
    if CHUNKS_JSON.exists():
        print(f"[1/3 chunker]  SKIP   — {CHUNKS_JSON.name} already exists")
        return
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Source JSON not found: {SCHEMA_JSON}")
    print(f"[1/3 chunker]  BUILD  — {SCHEMA_JSON.name} → {CHUNKS_JSON.name}")
    from chunker import build_tables, load_schema
    schema = load_schema(SCHEMA_JSON)
    tables = build_tables(schema)
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    print(f"[1/3 chunker]  DONE   — {len(tables)} tables written")


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def ensure_vector_store(store, embed_cfg: dict) -> None:
    print(f"[2/3 embed]    LOAD   — {CHUNKS_JSON.name}")
    tables = load_tables(CHUNKS_JSON)

    already_done = {id_.split("__")[0] for id_ in store.get_existing_ids()}
    tables_to_embed = [(t, c) for t, c in tables if t not in already_done]

    for t, _ in tables:
        status = "SKIP" if t in already_done else "EMBED"
        print(f"               [{t}]  {status}")

    if not tables_to_embed:
        print(f"[2/3 embed]    SKIP   — all tables stored ({store.count()} records total)")
        return

    model   = _get_model(embed_cfg["model"])
    batch   = embed_cfg.get("batch_size", 32)
    prefix  = embed_cfg.get("prefix_passage", "passage: ")

    for type_text, chunks in tables_to_embed:
        print(f"[2/3 embed]    BUILD  — [{type_text}]  {len(chunks)} chunks")
        texts     = build_encoding_texts(chunks, prefix_passage=prefix)
        vectors   = embed(model, texts, batch_size=batch)
        ids       = [f"{type_text}__siman_{r['siman']}_seif_{r['seif']}" for r in chunks]
        documents = [r["text"] for r in chunks]
        metadatas = [
            {"siman": int(r["siman"]), "seif": int(r["seif"]), "type_text": type_text}
            for r in chunks
        ]
        store.add_documents(documents, ids, metadatas, embeddings=vectors.tolist())
        print(f"               stored {len(chunks)} records")

    print(f"[2/3 embed]    DONE   — {store.count()} total records in store")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_queries(max_questions: int | None) -> pd.DataFrame:
    df = pd.read_csv(EVAL_CSV)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("שאלה", "question", "query"):  col_map[col] = "question"
        elif lc in ("סימן", "siman"):             col_map[col] = "siman"
        elif lc in ("סעיף", "seif"):              col_map[col] = "seif"
    df = df.rename(columns=col_map)
    total = len(df)
    if max_questions and max_questions < total:
        df = df.head(max_questions).reset_index(drop=True)
    print(f"               {len(df)} / {total} questions")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline — VectorStore strategy")
    parser.add_argument("--store-type", choices=["chroma", "manual"], default="chroma")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Wipe existing store and rebuild from scratch")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    embed_cfg      = cfg["embeddings"]
    eval_params    = cfg["evaluation"]
    retriever_name = f"vector_store_{args.store_type}"

    print("=" * 72)
    print(f"Store type:  {args.store_type.upper()}")
    print(f"Retriever:   {retriever_name}")
    print("=" * 72)

    # ── Stage 1: chunks_siman.json ────────────────────────────────────────────
    ensure_chunks_json()

    # ── Stage 2: VectorStore ──────────────────────────────────────────────────
    ef = E5EmbeddingFunction(
        model_name=embed_cfg["model"],
        prefix=embed_cfg.get("prefix_passage", "passage: "),
        batch_size=embed_cfg.get("batch_size", 32),
    )

    if args.store_type == "chroma":
        if args.force_rebuild and DEFAULT_CHROMA.exists():
            shutil.rmtree(DEFAULT_CHROMA)
            print(f"[force]  removed {DEFAULT_CHROMA.name}")
        store = ChromaStore(DEFAULT_CHROMA, COLLECTION_NAME, ef)
    else:
        if args.force_rebuild and DEFAULT_MANUAL.exists():
            shutil.rmtree(DEFAULT_MANUAL)
            print(f"[force]  removed {DEFAULT_MANUAL.name}")
        store = ManualStore(DEFAULT_MANUAL, ef)

    ensure_vector_store(store, embed_cfg)

    # ── Stage 3: retriever ────────────────────────────────────────────────────
    retriever = VectorStoreRetriever(store, name=retriever_name)
    print(f"[3/3 eval]     LOAD   — retriever: {retriever_name}")

    # ── Stage 4: queries ──────────────────────────────────────────────────────
    queries_df = load_queries(args.max_questions)

    # ── Stage 5: evaluate ─────────────────────────────────────────────────────
    eval_type   = eval_params.get("type", "retrieval")
    eval_kwargs = {k: v for k, v in eval_params.items() if k not in ("type", "max_questions")}
    evaluator   = get_evaluator(eval_type, **eval_kwargs)

    print("-" * 72)
    result = evaluator.evaluate(retriever, queries_df)

    # ── Stage 6: report + save ────────────────────────────────────────────────
    run_ts      = datetime.now()
    ts_filename = run_ts.strftime("%Y%m%d_%H%M%S")
    ts_readable = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    report_text = evaluator.format_report(
        result, retriever_name=retriever_name, ts_readable=ts_readable
    )
    print("\n" + report_text)

    result["timestamp"]  = ts_readable
    result["retriever"]  = retriever_name
    result["store_type"] = args.store_type

    stem  = f"exp_results_vector_store_{args.store_type}_{ts_filename}"
    saved = evaluator.save(result, report_text, output_dir=HERE, filename_stem=stem)
    print(f"\nReport saved -> {saved['txt'].name}")
    print(f"JSON saved   -> {saved['json'].name}")


if __name__ == "__main__":
    main()
