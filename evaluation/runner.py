"""
runner.py — end-to-end evaluation orchestrator
================================================
Loads the benchmark CSV, picks an evaluator via the registry, runs it against
a caller-supplied retriever, prints the report and saves results to disk.

Use from Python:
    from evaluation import run
    run(retriever=..., csv_path=..., output_dir=..., eval_params=cfg["evaluation"])
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import get_evaluator

# Default Hebrew → English mapping for the Shulchan Arukh benchmark CSV.
# Callers can override (or disable) via the `column_map` argument of run().
DEFAULT_COLUMN_MAP = {"שאלה": "question", "סימן": "siman", "סעיף": "seif"}
REQUIRED_COLUMNS   = {"question", "siman", "seif"}


def run(
    retriever,
    csv_path: str | Path,
    output_dir: str | Path,
    eval_params: dict,
    column_map: dict[str, str] | None = None,
) -> dict:
    """End-to-end evaluation: load CSV → evaluate retriever → format report → save.

    Args:
        retriever:    a pre-built retriever instance (caller chooses Chroma, npy, etc.)
        csv_path:     benchmark CSV with columns mapped to {question, siman, seif}
        output_dir:   directory where .txt and .json reports are written
        eval_params:  dict from cfg["evaluation"] (must include "type"; may include
                      "max_questions" and any evaluator-specific kwargs)
        column_map:   optional rename map applied to the CSV. None → use the default
                      Hebrew → English mapping; {} → no rename.

    Returns:
        The result dict produced by evaluator.evaluate(...).
    """
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(csv_path)
    rename_map = DEFAULT_COLUMN_MAP if column_map is None else column_map
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Eval CSV missing required columns: {missing}")

    max_q = eval_params.get("max_questions")
    if max_q is not None:
        df = df.head(int(max_q))

    evaluator   = get_evaluator(eval_params["type"], **eval_params)
    result      = evaluator.evaluate(retriever, df)
    report_text = evaluator.format_report(result, retriever_name=retriever.name)
    print(report_text)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem  = f"eval_{evaluator.name}_{retriever.name}_{ts}"
    saved = evaluator.save(result, report_text, output_dir, stem)
    print(f"Saved: {saved['txt']}\n       {saved['json']}")
    return result
