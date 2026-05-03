"""
exp_pipeline.py — Simple full RAG pipeline using the VectorStore strategy.
All settings are read from config/config.yaml — no CLI arguments.

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

from embedder import E5EmbeddingFunction, ChromaStore, ManualStore
from embedder.embed import load_tables, build_encoding_texts, embed, _get_model
from retrievers.vector_store_retriever import VectorStoreRetriever
from evaluation import get_evaluator

HERE        = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

embed_cfg = cfg["embeddings"]
eval_cfg  = cfg["evaluation"]
vs_cfg    = cfg["vector_store"]
paths_cfg = cfg["paths"]

SCHEMA_JSON = ROOT / paths_cfg["schema_json"]
CHUNKS_JSON = ROOT / paths_cfg["chunks_json"]
EVAL_CSV    = (HERE / paths_cfg["csv_path"]).resolve()


# ── helpers ───────────────────────────────────────────────────────────────────

def get_chunks() -> list:
    """Load chunks_siman.json, building it first if missing."""
    if not CHUNKS_JSON.exists():
        print(f"[1/3 chunker]  BUILD — {CHUNKS_JSON.name}")
        from chunker import build_tables, load_schema
        tables = build_tables(load_schema(SCHEMA_JSON))
        CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_JSON.write_text(
            json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"[1/3 chunker]  SKIP  — {CHUNKS_JSON.name} exists")
    return load_tables(CHUNKS_JSON)


def get_store():
    """Create the VectorStore from config (ChromaStore or ManualStore)."""
    ef = E5EmbeddingFunction(
        model_name=embed_cfg["model"],
        prefix=embed_cfg.get("prefix_passage", "passage: "),
        batch_size=embed_cfg.get("batch_size", 32),
    )
    if vs_cfg["type"] == "chroma":
        return ChromaStore(ROOT / vs_cfg["chroma_dir"], vs_cfg["collection"], ef)
    return ManualStore(ROOT / vs_cfg["manual_dir"], ef)


def fill_store(store, tables: list) -> None:
    """Embed and store each table, skipping tables already in the store."""
    already = {id_.split("__")[0] for id_ in store.get_existing_ids()}
    model  = _get_model(embed_cfg["model"])
    prefix = embed_cfg.get("prefix_passage", "passage: ")
    batch  = embed_cfg.get("batch_size", 32)

    for type_text, chunks in tables:
        if type_text in already:
            print(f"[2/3 embed]    SKIP  — [{type_text}]")
            continue
        print(f"[2/3 embed]    BUILD — [{type_text}]  {len(chunks)} chunks")
        texts     = build_encoding_texts(chunks, prefix_passage=prefix)
        vectors   = embed(model, texts, batch_size=batch)
        ids       = [f"{type_text}__siman_{r['siman']}_seif_{r['seif']}" for r in chunks]
        documents = [r["text"] for r in chunks]
        metadatas = [{"siman": int(r["siman"]), "seif": int(r["seif"]), "type_text": type_text}
                     for r in chunks]
        store.add_documents(documents, ids, metadatas, embeddings=vectors.tolist())

    print(f"[2/3 embed]    DONE  — {store.count()} records in store")


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
    retriever_name = f"vector_store_{vs_cfg['type']}"

    print("=" * 60)
    print(f"Store:     {vs_cfg['type'].upper()}")
    print(f"Retriever: {retriever_name}")
    print("=" * 60)

    # 1. chunks
    tables = get_chunks()

    # 2. store
    store = get_store()
    fill_store(store, tables)

    # 3. retriever
    retriever = VectorStoreRetriever(store, name=retriever_name)
    print(f"[3/3 eval]     LOAD  — {retriever_name}")

    # 4. queries
    queries_df = load_queries()
    max_q = eval_cfg.get("max_questions")
    if max_q:
        queries_df = queries_df.head(max_q).reset_index(drop=True)
    print(f"               {len(queries_df)} questions")

    # 5. evaluate
    eval_type   = eval_cfg.get("type", "retrieval")
    eval_kwargs = {k: v for k, v in eval_cfg.items() if k not in ("type", "max_questions")}
    evaluator   = get_evaluator(eval_type, **eval_kwargs)

    print("-" * 60)
    result = evaluator.evaluate(retriever, queries_df)

    # 6. report + save
    run_ts      = datetime.now()
    ts_filename = run_ts.strftime("%Y%m%d_%H%M%S")
    ts_readable = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    report = evaluator.format_report(result, retriever_name=retriever_name, ts_readable=ts_readable)
    print("\n" + report)

    result["timestamp"]  = ts_readable
    result["retriever"]  = retriever_name
    result["store_type"] = vs_cfg["type"]

    stem  = f"exp_results_pipeline_{vs_cfg['type']}_{ts_filename}"
    saved = evaluator.save(result, report, output_dir=HERE, filename_stem=stem)
    print(f"\nReport -> {saved['txt'].name}")
    print(f"JSON   -> {saved['json'].name}")


if __name__ == "__main__":
    main()
