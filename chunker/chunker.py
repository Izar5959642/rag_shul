"""
chunker.py — Chunker for Shulchan Arukh RAG pipeline
=====================================================
Reads the RAG JSON and builds a flat DataFrame dispatched by mode:
  - seif          : one row per seif (default, backward compatible)
  - siman         : one row per siman (all seifim concatenated)
  - sliding_window: word-count windows across the full corpus

Output DataFrame columns:
    siman      (int)      — chapter number
    seif       (int|None) — sub-chapter number (None for siman/sliding_window)
    siman_seif (str)      — "סימן N, סעיף M" or "סימן N"
    text       (str)      — joined chunk text sent to the embedder
"""

import json
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _build_seif_chunks(schema: dict, chunk_fields: list[str], siman_fields: list[str]) -> list[dict]:
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        siman_parts = [siman_data.get(f) for f in siman_fields if siman_data.get(f)]
        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]
            seif_parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            rows.append({
                "siman":      siman_num,
                "seif":       seif_num,
                "siman_seif": f"סימן {siman_num}, סעיף {seif_num}",
                "text":       " ".join(siman_parts + seif_parts),
            })
    return rows


def _build_siman_chunks(schema: dict, chunk_fields: list[str], siman_fields: list[str]) -> list[dict]:
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        siman_parts = [siman_data.get(f) for f in siman_fields if siman_data.get(f)]
        seif_parts = []
        for seif_data in siman_data["seifim"]:
            seif_parts += [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
        rows.append({
            "siman":      siman_num,
            "seif":       None,
            "siman_seif": f"סימן {siman_num}",
            "text":       " ".join(siman_parts + seif_parts),
        })
    return rows


def _build_sliding_window_chunks(schema: dict, chunk_fields: list[str], siman_fields: list[str]) -> list[dict]:
    cfg = load_config()["chunker"]
    chunk_size = cfg["chunk_size"]
    overlap = cfg["overlap"]
    step = chunk_size - overlap

    all_words: list[str] = []
    word_siman: list[int] = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        siman_parts = [siman_data.get(f) for f in siman_fields if siman_data.get(f)]
        siman_prefix_words = " ".join(siman_parts).split() if siman_parts else []
        all_words.extend(siman_prefix_words)
        word_siman.extend([siman_num] * len(siman_prefix_words))
        for seif_data in siman_data["seifim"]:
            parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            words = " ".join(parts).split()
            all_words.extend(words)
            word_siman.extend([siman_num] * len(words))

    rows = []
    for start in range(0, len(all_words), step):
        window = all_words[start:start + chunk_size]
        if not window:
            break
        siman_parent = word_siman[start]
        rows.append({
            "siman":      siman_parent,
            "seif":       None,
            "siman_seif": f"סימן {siman_parent}",
            "text":       " ".join(window),
        })
    return rows


def build_tables(schema: dict, variants: list[dict] | None = None) -> list[dict]:
    """
    Build one chunk table per variant defined in text_variants.

    Args:
        schema:   parsed JSON dict
        variants: list of variant dicts (type_text, chunk_fields, siman_fields, mode).
                  Defaults to chunker.text_variants from config.

    Returns:
        List of {metadata: {type_text}, data: [chunk rows]} objects.
    """
    if variants is None:
        variants = load_config()["chunker"]["text_variants"]
    tables = []
    for variant in variants:
        df = build_dataframe(
            schema,
            chunk_fields=variant.get("chunk_fields"),
            siman_fields=variant.get("siman_fields", []),
            mode=variant.get("mode"),
        )
        records = [{"id": i, **row} for i, row in enumerate(df.to_dict(orient="records"))]
        tables.append({
            "metadata": {"type_text": variant["type_text"]},
            "data": records,
        })
    return tables


def run(
    data_file: str | Path,
    chunks_json: str | Path,
    variants: list[dict] | None = None,
) -> None:
    """End-to-end chunker entry point: load schema → build tables → save chunks_json."""
    data_file   = Path(data_file)
    chunks_json = Path(chunks_json)
    print(f"  Loading schema from {data_file.name}...")
    schema = load_schema(data_file)
    print("  Building tables...")
    tables = build_tables(schema, variants=variants)
    chunks_json.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    print(f"  Chunks saved → {chunks_json}  ({len(tables)} tables)")


_DISPATCH = {
    "seif":           _build_seif_chunks,
    "siman":          _build_siman_chunks,
    "sliding_window": _build_sliding_window_chunks,
}


def build_dataframe(schema: dict, chunk_fields: list[str] | None = None, siman_fields: list[str] | None = None, mode: str | None = None) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame.

    Args:
        schema:       parsed JSON dict
        chunk_fields: seif fields to join into the text column (defaults to config)
        siman_fields: siman-level fields to prepend to each chunk (defaults to config)
        mode:         "seif" | "siman" | "sliding_window" (defaults to config)

    Returns:
        DataFrame with columns: siman, seif, siman_seif, text
        Sorted by siman, with a clean integer index.
    """
    cfg = load_config()["chunker"]
    if chunk_fields is None:
        chunk_fields = cfg["chunk_fields"]
    if siman_fields is None:
        siman_fields = cfg.get("siman_fields", [])
    if mode is None:
        mode = cfg["mode"]

    if mode not in _DISPATCH:
        raise ValueError(f"Unknown chunker mode: {mode!r}. Choose from {list(_DISPATCH)}")

    rows = _DISPATCH[mode](schema, chunk_fields, siman_fields)
    df = pd.DataFrame(rows)
    return df.sort_values(["siman"]).reset_index(drop=True)
