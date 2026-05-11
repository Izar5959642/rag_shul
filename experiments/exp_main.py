"""
main.py — RAG app template (YAML config)
========================================
Loads all pipeline settings from config/config.yaml and runs an
interactive query loop over the Shulchan Arukh corpus.

All settings come from the YAML config (model, chunks file, top_k, etc.).
To change anything, edit config/config.yaml or point CONFIG_PATH below
at a different config file.
"""

import logging
import sys
from pathlib import Path
import yaml
import json
import pandas as pd


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

    # 4. Build evaluation queries from questions embedded in the JSON
    log.info("Building evaluation queries from JSON questions...")
    rows = []
    for siman_obj in schema["simanim"]:
        siman_num = siman_obj["siman"]
        for seif_obj in siman_obj["seifim"]:
            for q in seif_obj.get("questions", []):
                rows.append({"question": q, "siman": siman_num})
    queries_df = pd.DataFrame(rows)
    log.info(f"Loaded {len(queries_df)} evaluation questions from JSON")

    # 5. Evaluate modern_summary retriever
    retriever  = get_retriever("chroma", type_text="modern_summary")
    evaluator  = get_evaluator(**evaluation_params)
    result     = evaluator.evaluate(retriever, queries_df)
    report     = evaluator.format_report(result, retriever_name="chroma/modern_summary")
    print("\n" + report)

    print("Ready.\n")

    


if __name__ == "__main__":
    main()
