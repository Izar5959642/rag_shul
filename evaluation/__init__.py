"""
Evaluator Registry
==================
Maps evaluator types (strings) to their classes.

To add a new evaluator:
    1. Create a new file in evaluation/  (e.g. my_evaluator.py)
    2. Add an import and a REGISTRY entry below
    3. Set `evaluation.type: my_evaluator` in exp_config.yaml

End-to-end orchestration lives in evaluation/runner.py and is re-exported as
`run` so callers can do `from evaluation import run`.
"""

from .base import BaseEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .retrieval_evaluator_by_variant import RetrievalEvaluatorByVariant
from .llm_evaluator import LLMEvaluator

# Map of type → class
REGISTRY: dict[str, type[BaseEvaluator]] = {
    "retrieval":            RetrievalEvaluator,
    "retrieval_by_variant": RetrievalEvaluatorByVariant,
    "llm_qa":               LLMEvaluator,
}


def get_evaluator(eval_type: str, **kwargs) -> BaseEvaluator:
    """
    Returns an evaluator instance by type. kwargs are forwarded to the constructor —
    so YAML parameters flow directly into every evaluator (it picks what it cares about).
    """
    if eval_type not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(
            f"Evaluator '{eval_type}' not found. Available: {available}"
        )
    return REGISTRY[eval_type](**kwargs)


def list_evaluators() -> list[str]:
    return list(REGISTRY.keys())


# Re-export end-to-end runner so callers can `from evaluation import run`
from .runner import run  # noqa: E402  (registry must be defined before runner imports it)
