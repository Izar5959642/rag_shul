"""
embed.py — Embedding layer for Shulchan Arukh RAG
===================================================
Reads chunks_siman.json (output of the chunker) — a list of tables,
each with a type_text label and a list of chunk records — and stores
sentence embeddings in a single ChromaDB collection.

Input JSON structure:
    [
      {
        "metadata": {"type_text": "text+hagah"},
        "data": [
          {"id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..."},
          ...
        ]
      },
      ...
    ]

ChromaDB record:
    id:       "{type_text}__siman_{siman}_seif_{seif}"
    document: raw text
    embedding: 1024-dim normalized float32 vector
    metadata: {siman (int), seif (int), type_text (str)}

Usage:
    python embed.py --chunks data/chunks_siman.json
    python embed.py --chunks data/chunks_siman.json --model intfloat/multilingual-e5-large
    python embed.py --chunks data/chunks_siman.json --chroma-dir ./my_chroma
    python embed.py --chunks data/chunks_siman.json --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "intfloat/multilingual-e5-large"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh_seifs"
BATCH_SIZE         = 32

# ─── Model cache ───────────────────────────────────────────────────────────────
# Keeps one loaded model per model name — avoids reloading 500MB on every call.
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ─── Load ───────────────────────────────────────────────────────────────────────

def load_tables(json_path: Path) -> list[tuple[str, list[dict]]]:
    """
    Load tables from chunks_siman.json.

    Returns:
        list of (type_text, chunks) where chunks is a list of dicts
        with keys: id, siman, seif, siman_seif, text
    """
    with open(json_path, encoding="utf-8") as f:
        tables = json.load(f)
    result = []
    for table in tables:
        type_text = table["metadata"]["type_text"]
        chunks    = table["data"]
        result.append((type_text, chunks))
        print(f"  [{type_text}]  {len(chunks)} chunks")
    return result


# ─── Encode ─────────────────────────────────────────────────────────────────────

def build_encoding_texts(chunks: list[dict], prefix_passage: str = "passage: ") -> list[str]:
    """
    Build the text string that gets embedded for each chunk.
    E5 models require "passage: " prefix for corpus texts.
    """
    return [
        prefix_passage + "שולחן ערוך אורח חיים, " + row["siman_seif"] + ": " + row["text"]
        for row in chunks
    ]


def embed(model: SentenceTransformer, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode all texts into normalized float32 vectors. Returns (N, dim) float32 array."""
    print(f"  Encoding {len(texts)} chunks (batch_size={batch_size})...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    return vectors  # float32 ndarray — no .tolist() conversion


def encode_query(
    query: str,
    model: "str | SentenceTransformer" = DEFAULT_MODEL,
    prefix_query: str = "query: ",
) -> np.ndarray:
    """
    Encode a single query string into a normalized vector.
    Used by the retriever at query time.

    Args:
        query:        the question string
        model:        a loaded SentenceTransformer instance, or a model name string
        prefix_query: prefix to prepend (default: "query: " for E5 models)

    Returns:
        1-D normalized float32 numpy array
    """
    if isinstance(model, str):
        model = _get_model(model)
    return model.encode(
        prefix_query + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ─── Store ──────────────────────────────────────────────────────────────────────

def get_existing_type_texts(chroma_dir: Path, collection_name: str) -> set[str]:
    """
    Return the set of type_text values already stored in the collection.
    Returns an empty set if the collection does not exist.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return set()

    col = client.get_collection(collection_name)
    result = col.get(include=["metadatas"])
    return {m["type_text"] for m in result["metadatas"] if "type_text" in m}


def store_in_chroma(
    all_tables: "list[tuple[str, list[dict], list[list[float]]]]",
    chroma_dir: Path,
    collection_name: str,
) -> None:
    """
    Store embeddings from all tables into a single ChromaDB collection.
    Gets or creates the collection — does NOT delete existing data.

    Args:
        all_tables:      list of (type_text, chunks, vectors)
        chroma_dir:      path for the persistent ChromaDB storage
        collection_name: name of the ChromaDB collection
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    for type_text, chunks, vectors in all_tables:
        ids       = [f"{type_text}__siman_{row['siman']}_seif_{row['seif']}" for row in chunks]
        documents = [row["text"] for row in chunks]
        metadatas = [
            {
                "siman":     int(row["siman"]),
                "seif":      int(row["seif"]),
                "type_text": type_text,
            }
            for row in chunks
        ]
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"  [{type_text}]  added {len(chunks)} records")

    print(f"  Total in collection '{collection_name}': {collection.count()}")
    print(f"  ChromaDB path: {chroma_dir}")


# ─── Programmatic entry point ───────────────────────────────────────────────────

def run(
    chunks_json: Path,
    chroma_dir: Path,
    model: str = DEFAULT_MODEL,
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Embed all tables from chunks_json into ChromaDB. Skips already-embedded variants."""
    print(f"\n1. Loading tables from {chunks_json.name}...")
    tables = load_tables(chunks_json)

    print("\n2. Checking existing embeddings in ChromaDB...")
    already_done = get_existing_type_texts(chroma_dir, collection)
    tables_to_embed = [(t, c) for t, c in tables if t not in already_done]
    for t, _ in tables:
        status = "SKIP (already exists)" if t in already_done else "will embed"
        print(f"  [{t}]  {status}")

    if not tables_to_embed:
        print("\nAll tables already embedded. Nothing to do.")
        return

    print(f"\n3. Loading model: {model}")
    model_obj = _get_model(model)
    print(f"   Vector dim: {model_obj.get_sentence_embedding_dimension()}")

    all_tables = []
    for type_text, chunks in tables_to_embed:
        print(f"\nEmbedding [{type_text}]...")
        texts   = build_encoding_texts(chunks)
        vectors = embed(model_obj, texts, batch_size=batch_size)
        all_tables.append((type_text, chunks, vectors))

    print("\nStoring in ChromaDB...")
    store_in_chroma(all_tables, chroma_dir, collection)


# ─── CLI entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed Shulchan Arukh chunks into ChromaDB")
    parser.add_argument("--chunks",      required=True,                   help="Path to chunks_siman.json (chunker output)")
    parser.add_argument("--model",       default=DEFAULT_MODEL,           help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chroma-dir",  default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--collection",  default=DEFAULT_COLLECTION,      help="ChromaDB collection name")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE,    help=f"Encoding batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    json_path  = Path(args.chunks)
    chroma_dir = Path(args.chroma_dir)
    step = 1

    print(f"\n{step}. Loading tables from {json_path.name}..."); step += 1
    tables = load_tables(json_path)

    print(f"\n{step}. Checking existing embeddings in ChromaDB..."); step += 1
    already_done = get_existing_type_texts(chroma_dir, args.collection)
    tables_to_embed = [(t, c) for t, c in tables if t not in already_done]
    for t, _ in tables:
        status = "SKIP (already exists)" if t in already_done else "will embed"
        print(f"  [{t}]  {status}")

    if not tables_to_embed:
        print("\nAll tables already embedded. Nothing to do.")
        return

    print(f"\n{step}. Loading model: {args.model}"); step += 1
    model = _get_model(args.model)
    print(f"   Vector dim: {model.get_sentence_embedding_dimension()}")

    all_tables = []
    for type_text, chunks in tables_to_embed:
        print(f"\n{step}. Embedding [{type_text}]..."); step += 1
        texts   = build_encoding_texts(chunks)
        vectors = embed(model, texts, batch_size=args.batch_size)
        all_tables.append((type_text, chunks, vectors))

    print(f"\n{step}. Storing in ChromaDB...")
    store_in_chroma(all_tables, chroma_dir, args.collection)

    print(f"\nDone. To query:\n"
          f"  import chromadb\n"
          f"  client = chromadb.PersistentClient('{chroma_dir}')\n"
          f"  col = client.get_collection('{args.collection}')\n"
          f"  col.query(query_embeddings=[...], n_results=10, where={{'type_text': 'text+hagah'}})")


if __name__ == "__main__":
    main()
