#!/usr/bin/env python3
"""
enrich_with_modern_summary.py
==============================
Enriches shulchan_aruch_rag.json with two GPT-generated fields per seif:
  • modern_summary — a plain-Hebrew summary of the seif
  • questions      — a list of questions whose answer is in that seif

Expected project layout (script lives in data/scripts/):
    <project_root>/
      └── data/
          ├── scripts/
          │   └── enrich_with_modern_summary.py   ← this file
          ├── seif_modern_summary_cache.json       ← input  (key: "siman_seif")
          ├── seif_questions_gpt_cache.json        ← input  (key: "siman_seif")
          └── processed/
              └── shulchan_aruch_rag.json          ← input & output (enriched in-place)

Cache key format:  "<siman>_<seif>"  e.g. "1_1", "12_3"

Usage:
    python data/scripts/enrich_with_modern_summary.py
    python data/scripts/enrich_with_modern_summary.py --rag processed/shulchan_aruch_rag.json \
        --summaries seif_modern_summary_cache.json \
        --questions seif_questions_gpt_cache.json \
        --output processed/shulchan_aruch_rag.json
"""

import argparse
import json
from pathlib import Path

# ── Default paths (relative to data/) ────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # data/scripts/
_DATA = _HERE.parent                             # data/

DEFAULT_RAG       = _DATA / "processed" / "shulchan_aruch_rag.json"
DEFAULT_SUMMARIES = _DATA / "seif_modern_summary_cache.json"
DEFAULT_QUESTIONS = _DATA / "seif_questions_gpt_cache.json"
DEFAULT_OUTPUT    = DEFAULT_RAG                  # overwrite in-place by default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich RAG JSON with modern summaries and questions")
    p.add_argument("--rag",       default=DEFAULT_RAG,       help="Path to shulchan_aruch_rag.json")
    p.add_argument("--summaries", default=DEFAULT_SUMMARIES, help="Path to seif_modern_summary_cache.json")
    p.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Path to seif_questions_gpt_cache.json")
    p.add_argument("--output",    default=DEFAULT_OUTPUT,    help="Output path (default: overwrite --rag)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rag_path       = Path(args.rag)
    summaries_path = Path(args.summaries)
    questions_path = Path(args.questions)
    output_path    = Path(args.output)

    print(f"Loading RAG JSON:   {rag_path}")
    with open(rag_path, encoding="utf-8") as f:
        rag = json.load(f)

    print(f"Loading summaries:  {summaries_path}")
    with open(summaries_path, encoding="utf-8") as f:
        summaries = json.load(f)

    print(f"Loading questions:  {questions_path}")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    added   = 0
    missing = 0

    for siman_obj in rag["simanim"]:
        siman_num = siman_obj["siman"]
        for seif_obj in siman_obj["seifim"]:
            seif_num = seif_obj["seif"]
            key = f"{siman_num}_{seif_num}"

            seif_obj["modern_summary"] = summaries.get(key, "")
            seif_obj["questions"]      = questions.get(key, [])

            if key in summaries:
                added += 1
            else:
                missing += 1

    print(f"Enriched: {added} seifim  |  Missing in cache: {missing}")

    print(f"Writing output:    {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rag, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
