"""
exp_main.py — RAG pipeline with incremental skip logic
=======================================================
Runs only the stages whose output artifacts are missing.

Stage 0 — Data generation  → data/processed/shulchan_aruch_rag_with_breadcrumb.json
Stage 1 — Chunking         → data/chunks.json
Stage 2 — Embedding        → embedder/chroma_db_ver_2/
Stage 3 — Eval + Retrieval → data/eval/results/  (always runs)

Set  rebuild: true  in config/config.yaml to delete all artifacts and rerun from scratch.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from build_shulchan_aruch_rag import process_file as _build_rag
from enrich_with_modern_summary import process as _enrich_rag
from add_breadcrumb_to_json import process as _add_breadcrumb


from chunker.chunker import run as _run_chunker
from embedder.embed import run as _run_embed
from retrievers import get_retriever
from evaluation import run as _run_evaluation

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, cfg.get("log_level", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exp_main")

# ─── Paths (all from config.yaml) ─────────────────────────────────────────────

paths = cfg["paths"]

def _p(key: str) -> Path:
    return (ROOT / paths[key]).resolve()

# Stage 0 — data generation
SOURCE_TXT     = _p("source_txt")
HEADINGS_TXT   = _p("headings_txt")
SUMMARIES_JSON = _p("summaries_cache")
QUESTIONS_JSON = _p("questions_cache")
DATA_RAW       = _p("data_file_without_additional_breadcrumb")
DATA_FILE      = _p("data_file_with_breadcrumb")

# Stage 1 — chunking
CHUNKS_JSON = _p("chunks_json")

# Stage 2 — embedding (env CHROMA_DIR overrides config)
_chroma_env = os.environ.get("CHROMA_DIR")
CHROMA_DIR  = Path(_chroma_env) if _chroma_env else _p("chroma_dir")

# Stage 3 — evaluation
CSV_PATH         = _p("csv_path")
EVAL_RESULTS_DIR = _p("eval_results_dir")
QUERY_EMB_NPY    = _p("query_embeddings_npy")
QUERY_TEXTS_JSON = _p("query_texts_json")

# Per-stage param dicts
chunker_params    = cfg["chunker"]
embed_params      = cfg["embeddings"]
evaluation_params = cfg["evaluation"]

# ─── Stage functions ──────────────────────────────────────────────────────────

def _build_data_stage() -> None:
    print("=== Stage 0: Data generation ===")
    # 1. raw build:   SOURCE_TXT  → DATA_RAW         (DATA_RAW stays untouched after this)
    _build_rag(SOURCE_TXT, DATA_RAW)
    # 2. enrichment:  DATA_RAW + caches → DATA_FILE  (writes a new file, NOT in-place)
    _enrich_rag(DATA_RAW, SUMMARIES_JSON, QUESTIONS_JSON, DATA_FILE)
    # 3. breadcrumb:  DATA_FILE + headings → DATA_FILE  (overwrites with breadcrumb fields)
    _add_breadcrumb(DATA_FILE, HEADINGS_TXT, DATA_FILE)


def _build_chunks_stage() -> None:
    print("=== Stage 1: Chunking ===")
    _run_chunker(
        data_file=DATA_FILE,
        chunks_json=CHUNKS_JSON,
        variants=chunker_params.get("text_variants"),
    )


def _build_embed_stage() -> None:
    print("=== Stage 2: Embedding ===")
    _run_embed(
        chunks_json=CHUNKS_JSON,
        chroma_dir=CHROMA_DIR,
        model=embed_params["model"],
        batch_size=embed_params.get("batch_size", 32),
    )


def _run_eval() -> None:
    print("=== Stage 3: Evaluation ===")
    retriever = get_retriever("chroma", type_text=None, chroma_dir=CHROMA_DIR)
    evaluation_params["query_embeddings_npy"] = str(QUERY_EMB_NPY)
    evaluation_params["query_texts_json"]     = str(QUERY_TEXTS_JSON)
    _run_evaluation(
        retriever=retriever,
        csv_path=CSV_PATH,
        output_dir=EVAL_RESULTS_DIR,
        eval_params=evaluation_params,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    log.info(f"Config:    {CONFIG_PATH}")
    log.info(f"Data file: {DATA_FILE}")
    log.info(f"Chunks:    {CHUNKS_JSON}")
    log.info(f"ChromaDB:  {CHROMA_DIR}")
    log.info(f"Eval CSV:  {CSV_PATH}")

    rebuild = cfg.get("rebuild", False)
    if rebuild:
        print("rebuild=true — removing intermediate files...")
        for p in [DATA_FILE, CHUNKS_JSON]:
            if p.exists():
                p.unlink()
                print(f"  Deleted {p}")
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            print(f"  Deleted {CHROMA_DIR}")

    if not DATA_FILE.exists():
        _build_data_stage()
    else:
        print(f"Data file found at {DATA_FILE} — skipping data generation.")

    if not CHUNKS_JSON.exists():
        _build_chunks_stage()
    else:
        print(f"Chunks found at {CHUNKS_JSON} — skipping chunking.")

    # Always enter the embed stage — run() does fine-grained per-variant skipping
    # via get_existing_type_texts(), which also covers the "added a new variant" case.
    _build_embed_stage()

    _run_eval()


if __name__ == "__main__":
    main()
