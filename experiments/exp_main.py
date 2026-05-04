"""
main.py — RAG app template (YAML config)
========================================
Loads all pipeline settings from config/config.yaml and runs an
interactive query loop over the Shulchan Arukh corpus.

All settings come from the YAML config (model, chunks file, top_k, etc.).
To change anything, edit config/config.yaml or point CONFIG_PATH below
at a different config file.
"""
import pandas as pd  
import logging
import sys
from pathlib import Path
import yaml
import json


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker.chunker import load_schema, build_tables
from embedder import embed as embed_main
from retrievers import get_retriever
from evaluation import get_evaluator

# ─── Load config ──────────────────────────────────────────────────────────────

#HERE        = Path(__file__).parent
#CONFIG_PATH = HERE / "config" / "config.yaml"
ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print("CONFIG_PATH:", CONFIG_PATH)
print("retrieval from yaml:", cfg["retrieval"])

# Logging
logging.basicConfig(
    level=getattr(logging, cfg.get("log_level", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# Paths (resolved against this file's directory)
DATA_FILE      = (HERE / cfg["paths"]["data_file"]).resolve()
CSV_PATH       = (HERE / cfg["paths"]["csv_path"]).resolve()
CHUNKS_JSON    = (HERE / cfg["paths"]["chunks_json"]).resolve()
EMBEDDINGS_NPY = (HERE / cfg["paths"]["embeddings_file"]).resolve()

# Per-stage param dicts
chunker_params    = cfg["chunker"]
embed_params      = cfg["embeddings"]
retrieval_params  = cfg["retrieval"]
evaluation_params = cfg["evaluation"]



# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    
    
    log.info(f"Config:     {CONFIG_PATH}")
    log.info(f"Data file:  {DATA_FILE}")
    log.info(f"Chunks:     {CHUNKS_JSON}")
    log.info(f"Embeddings: {EMBEDDINGS_NPY}")
    log.info(f"Eval CSV:   {CSV_PATH}")
    log.info(f"Model:      {embed_params['model']}")
    
    
    # 1. Load JSON
    schema = load_schema(DATA_FILE)

    # 2. Build chunks
    tables = build_tables(schema)

    # 3. Save to file (chunks.json)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    print("Chunks built and saved")
    
    sys.argv = [
    "embed.py",
    "--chunks", str(CHUNKS_JSON),
    "--model", embed_params["model"],
]

    embed_main.main()
                                                                                
                                                            
    # Build retriever (all 3 variants)
    retriever = get_retriever(
        "chroma",
        type_text=None,          # None = all variants in the collection
        model=embed_params["model"],
        prefix_query=embed_params["prefix_query"],
    )

    # Take first real question from eval CSV
    queries_df = pd.read_csv(CSV_PATH)
    row = queries_df.iloc[0]
    test_query = row["question"]
    gt_siman   = int(row["siman"])
    gt_seif    = int(row["seif"])

    print(f"\nQuery:        {test_query}")
    print(f"Ground truth: siman={gt_siman}, seif={gt_seif}")

    results = retriever.retrieve(test_query, top_k=5)

    # Raw list
    print("\n--- raw list (first 6 items) ---")
    for r in results[:6]:
        print(r)

    # DataFrame — mark the correct answer
    print("\n--- as DataFrame ---")
    df = pd.DataFrame(results)
    df["is_correct"] = (df["siman_parent"] == gt_siman) & (df["seif"] == gt_seif)
    print(df[["rank", "score", "siman_parent", "seif", "type_text", "is_correct"]].head(10))    
            
    print("Ready.\n")

    


if __name__ == "__main__":
    main()
