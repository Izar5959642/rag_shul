"""
Retriever Registry
==================
Maps retriever names (strings) to their classes.

To add a new retriever:
    1. Create a new file in retrievers/  (e.g. my_retriever.py)
    2. Add an import and a REGISTRY entry below
    3. Run:  python experiments/exp_main.py --retriever my_retriever
"""

from .base import BaseRetriever
from .semantic_e5_seif_v6_combined import SemanticE5SeifV6CombinedRetriever
from .npy_retriever import NpyRetriever
from .vector_store_retriever import VectorStoreRetriever

# Map of name → class for every available retriever
REGISTRY: dict[str, type[BaseRetriever]] = {
    "semantic_e5_seif_v6_combined": SemanticE5SeifV6CombinedRetriever,
    "retrieval_npy":                NpyRetriever,
    "vector_store":                 VectorStoreRetriever,
}


def get_retriever(name: str, **kwargs) -> BaseRetriever:
    """
    Return an instance of the named retriever.

    Any kwargs are forwarded to the retriever's __init__ (e.g. chunks_csv,
    embeddings_npy, model, prefix_query, top_k, score_threshold).
    Retrievers are expected to ignore kwargs they don't use.

    Raises ValueError if the name is unknown.
    """
    if name not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(f"Retriever '{name}' not found. Available: {available}")
    return REGISTRY[name](**kwargs)


def list_retrievers() -> list[str]:
    """Return the names of all registered retrievers."""
    return list(REGISTRY.keys())
