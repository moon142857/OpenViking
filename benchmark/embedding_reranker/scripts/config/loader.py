"""Load model and suite configurations from YAML files."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CONFIG_DIR = Path(__file__).parent.resolve()
MODELS_DIR = CONFIG_DIR / "models"
SUITES_DIR = CONFIG_DIR / "suites"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(name: str) -> Dict[str, Any]:
    """Load a single model config by file stem (e.g. 'qwen3_embedding_8b_mxfp8')."""
    path = MODELS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    return _load_yaml(path)


def load_suite_config(name: str = "default") -> Dict[str, Any]:
    """Load a suite config by file stem (e.g. 'default')."""
    path = SUITES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Suite config not found: {path}")
    return _load_yaml(path)


def resolve_model_config(identifier: str) -> Dict[str, Any]:
    """Resolve a model identifier to a config dict.

    If `identifier` is a path to an existing YAML file, load it directly.
    If it matches a config file stem under config/models/, load that.
    Otherwise, treat it as a raw model_id/path and return a minimal config
    with auto loader detection.
    """
    maybe_path = Path(identifier)
    if maybe_path.exists() and maybe_path.suffix in (".yaml", ".yml"):
        return _load_yaml(maybe_path)

    config_path = MODELS_DIR / f"{identifier}.yaml"
    if config_path.exists():
        return _load_yaml(config_path)

    # Fallback: treat identifier as a raw model_id / local path.
    return {
        "model_id": identifier,
        "display_name": identifier,
        "loader": _auto_detect_loader(identifier),
        "instruction": "Given a web search query, retrieve relevant passages that answer the query",
        "query_prefix": "Instruct: {instruction}\\nQuery:{text}",
        "document_prefix": "{text}",
        "embedding": {"pooling": "last", "normalize": True},
        "reranker": {
            "prompt_template": "qwen3",
            "yes_token": "yes",
            "no_token": "no",
            "fallback_on_missing_lm_head": True,
        },
        "notes": "Auto-detected fallback config.",
    }


def _auto_detect_loader(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen3-embedding" in lowered:
        # mlx-embeddings implements the correct Qwen3 embedding pooling path;
        # mlx_lm produces near-random embeddings on standard benchmarks.
        return "mlx_embeddings"
    if "qwen3-reranker" in lowered and lowered.endswith("-mxfp8"):
        return "mlx_embeddings"
    return "mlx_lm"


def format_query(text: str, instruction: str, query_prefix: Optional[str]) -> str:
    if query_prefix is None:
        return text
    return query_prefix.format(instruction=instruction, text=text)


def format_document(text: str, document_prefix: Optional[str]) -> str:
    if document_prefix is None:
        return text
    return document_prefix.format(text=text)
