#!/usr/bin/env python3
"""Reranker model functional benchmark (config-driven).

Evaluates reranker models on:
- Pair-wise ranking accuracy (relevant vs irrelevant)
- NDCG-style ranking with graded relevance labels
- Spearman correlation with human relevance labels

Model configs live in config/models/*.yaml and specify:
  - model_id / local path
  - loader: mlx_lm | mlx_embeddings
  - instruction
  - reranker mode: cross_encoder | embedding_similarity

Two official usage patterns are supported:

1. cross_encoder (e.g. Qwen3-Reranker-0.6B-4bit)
   Build the official Qwen3-Reranker prompt and score with
   softmax([logit("no"), logit("yes")])[1] at the last token.

2. embedding_similarity (e.g. Qwen3-Reranker-*-mxfp8 from mlx-embeddings)
   Embed query and document independently and use cosine/dot similarity
   as the relevance score, matching the official mlx-embeddings example.

Usage:
    # Cross-encoder reranker
    python scripts/benchmark_reranker.py --config qwen3_reranker_0.6b_4bit

    # Bi-encoder / embedding-similarity reranker
    python scripts/benchmark_reranker.py --config qwen3_reranker_8b_mxfp8

    # Raw model_id (defaults to cross_encoder)
    python scripts/benchmark_reranker.py --model mlx-community/Qwen3-Reranker-0.6B-4bit
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
from config.loader import (
    format_document,
    format_query,
    resolve_model_config,
)
from mlx_lm import load as mlx_lm_load
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

try:
    from mlx_embeddings import generate as mlx_embeddings_generate
    from mlx_embeddings import load as mlx_embeddings_load

    HAS_MLX_EMBEDDINGS = True
except ImportError:
    HAS_MLX_EMBEDDINGS = False

QWEN3_RANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_RANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_qwen3_reranker_prompt(instruction: str, query: str, doc: str) -> str:
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    return f"{QWEN3_RANKER_PREFIX}{body}{QWEN3_RANKER_SUFFIX}"


def load_model(cfg: Dict) -> Tuple[object, object, float, bool]:
    """Load a reranker model.

    Returns:
        (model, tokenizer, load_time_s, use_mlx_embeddings)
    """
    model_id = cfg["model_id"]
    loader = cfg["loader"]
    mode = cfg.get("reranker", {}).get("mode", "cross_encoder")
    fallback = cfg.get("reranker", {}).get("fallback_on_missing_lm_head", False)

    print(f"Loading reranker model: {model_id}")
    print(f"  mode={mode}, loader={loader}")
    t0 = time.time()

    if loader == "mlx_embeddings":
        if not HAS_MLX_EMBEDDINGS:
            raise RuntimeError("mlx-embeddings is required. Install: pip install mlx-embeddings")
        model, tokenizer = mlx_embeddings_load(model_id)
        print(f"  Loaded with mlx-embeddings in {time.time() - t0:.2f}s")
        return model, tokenizer, time.time() - t0, True

    # loader == mlx_lm
    try:
        model, tokenizer = mlx_lm_load(model_id)
        print(f"  Loaded with mlx_lm in {time.time() - t0:.2f}s")
        return model, tokenizer, time.time() - t0, False
    except ValueError as e:
        if "lm_head.weight" not in str(e) or not fallback:
            raise

    print("  mlx_lm failed: missing lm_head.weight. Falling back to mlx-embeddings "
          "(uses embed_tokens.as_linear as a tied lm_head approximation).")
    if not HAS_MLX_EMBEDDINGS:
        raise RuntimeError("mlx-embeddings is required for this fallback. Install: pip install mlx-embeddings")

    model, tokenizer = mlx_embeddings_load(model_id)
    print(f"  Loaded with mlx-embeddings in {time.time() - t0:.2f}s")
    return model, tokenizer, time.time() - t0, True


def tokenize_inputs(tokenizer, texts, max_length: int):
    inner = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
    return inner(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=max_length,
    )


def get_token_id(tokenizer, token: str) -> int:
    inner = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
    return inner.convert_tokens_to_ids(token)


def embed_texts_mlx_embeddings(
    texts: List[str],
    model,
    processor,
    max_length: int,
) -> np.ndarray:
    """Embed a list of texts using the mlx-embeddings generate API."""
    if not texts:
        return np.zeros((0, 0))
    output = mlx_embeddings_generate(
        model,
        processor,
        texts=texts,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    return np.array(output.text_embeds.astype(mx.float32))


def rerank_scores_cross_encoder(
    queries: List[str],
    docs: List[str],
    model,
    tokenizer,
    cfg: Dict,
    use_mlx_embeddings: bool,
) -> np.ndarray:
    instruction = cfg["instruction"]
    yes_token = cfg.get("reranker", {}).get("yes_token", "yes")
    no_token = cfg.get("reranker", {}).get("no_token", "no")
    max_length = cfg["max_length"]

    inner = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer

    # Build inputs at the token level (matching the official Qwen3-Reranker):
    # tokenize the fixed prefix/suffix separately and truncate ONLY the
    # instruct+query+document body. This guarantees the assistant decision
    # suffix (where "yes"/"no" is predicted) always survives -- truncating the
    # full prompt string instead chops off the suffix on long documents, so the
    # scored position lands mid-document and the yes/no logits become garbage.
    prefix_ids = inner(QWEN3_RANKER_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_ids = inner(QWEN3_RANKER_SUFFIX, add_special_tokens=False)["input_ids"]
    body_budget = max(1, max_length - len(prefix_ids) - len(suffix_ids))

    bodies = [
        f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}"
        for q, d in zip(queries, docs)
    ]
    body_ids = inner(
        bodies, add_special_tokens=False, truncation=True, max_length=body_budget
    )["input_ids"]

    seqs = [prefix_ids + b + suffix_ids for b in body_ids]
    batch_len = max(len(s) for s in seqs)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else 0

    input_ids_np = np.full((len(seqs), batch_len), pad_id, dtype=np.int32)
    attention_mask_np = np.zeros((len(seqs), batch_len), dtype=np.int32)
    for i, s in enumerate(seqs):  # right padding
        input_ids_np[i, : len(s)] = s
        attention_mask_np[i, : len(s)] = 1
    input_ids = mx.array(input_ids_np)

    if use_mlx_embeddings:
        attention_mask = mx.array(attention_mask_np)
        last_hidden_state = model.model(input_ids, attention_mask=attention_mask)
        logits = model.model.embed_tokens.as_linear(last_hidden_state)
    else:
        # mlx_lm builds its own causal mask. With right-padding, each sequence's
        # final real token only attends to earlier real tokens, so its logits are
        # uncorrupted by padding -- we just have to read them at the true last
        # token index rather than [:, -1, :] (which is a PAD row for the shorter
        # sequences in a mixed-length batch).
        outputs = model(input_ids)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

    yes_id = get_token_id(tokenizer, yes_token)
    no_id = get_token_id(tokenizer, no_token)

    # Gather logits at each row's last non-padding token (attention_mask.sum - 1),
    # which is now always the final suffix token (the yes/no decision point).
    last_idx = attention_mask_np.sum(axis=1).astype(np.int32) - 1
    gather_idx = mx.broadcast_to(
        mx.array(last_idx).reshape(-1, 1, 1),
        (logits.shape[0], 1, logits.shape[2]),
    )
    last_logits = mx.take_along_axis(logits, gather_idx, axis=1)[:, 0, :]

    score_matrix = np.stack(
        [
            np.array(last_logits[:, no_id].astype(mx.float32)),
            np.array(last_logits[:, yes_id].astype(mx.float32)),
        ],
        axis=1,
    )
    scores = softmax(score_matrix, axis=1)[:, 1].astype(np.float32)
    return scores


def rerank_scores_embedding_similarity(
    queries: List[str],
    docs: List[str],
    model,
    tokenizer,
    cfg: Dict,
) -> np.ndarray:
    """Bi-encoder reranker: embed query/doc separately, score by cosine similarity."""
    instruction = cfg["instruction"]
    query_prefix = cfg.get("query_prefix")
    document_prefix = cfg.get("document_prefix")
    max_length = cfg["max_length"]

    query_texts = [format_query(q, instruction, query_prefix) for q in queries]
    doc_texts = [format_document(d, document_prefix) for d in docs]

    query_embs = embed_texts_mlx_embeddings(query_texts, model, tokenizer, max_length)
    doc_embs = embed_texts_mlx_embeddings(doc_texts, model, tokenizer, max_length)

    scores = cosine_similarity(query_embs, doc_embs).diagonal()
    return scores.astype(np.float32)


def rerank_scores(
    queries: List[str],
    docs: List[str],
    model,
    tokenizer,
    cfg: Dict,
    use_mlx_embeddings: bool,
) -> np.ndarray:
    mode = cfg.get("reranker", {}).get("mode", "cross_encoder")
    if mode == "embedding_similarity":
        return rerank_scores_embedding_similarity(queries, docs, model, tokenizer, cfg)
    return rerank_scores_cross_encoder(queries, docs, model, tokenizer, cfg, use_mlx_embeddings)


def benchmark_pairwise_accuracy(
    data: List[Dict], model, tokenizer, cfg: Dict, use_mlx_embeddings: bool
) -> Dict:
    by_query = defaultdict(list)
    for item in data:
        by_query[item["query"]].append(item)

    correct = 0
    total = 0
    for query, items in by_query.items():
        relevant = [i for i in items if i["relevance"] >= 2]
        irrelevant = [i for i in items if i["relevance"] == 0]
        if not relevant or not irrelevant:
            continue

        q_list = [query] * (len(relevant) + len(irrelevant))
        d_list = [i["doc"] for i in relevant + irrelevant]
        scores = rerank_scores(q_list, d_list, model, tokenizer, cfg, use_mlx_embeddings)

        rel_scores = scores[: len(relevant)]
        irrel_scores = scores[len(relevant) :]
        total += len(relevant) * len(irrelevant)
        correct += int(np.sum(rel_scores[:, None] > irrel_scores[None, :]))

    accuracy = correct / total if total else 0.0
    return {
        "metric": "pairwise_accuracy",
        "value": round(float(accuracy), 4),
        "samples": total,
    }


def benchmark_ndcg(
    data: List[Dict], model, tokenizer, cfg: Dict, use_mlx_embeddings: bool, k: int = 10
) -> Dict:
    by_query = defaultdict(list)
    for item in data:
        by_query[item["query"]].append(item)

    ndcg_scores = []
    for query, items in by_query.items():
        q_list = [query] * len(items)
        d_list = [i["doc"] for i in items]
        scores = rerank_scores(q_list, d_list, model, tokenizer, cfg, use_mlx_embeddings)

        order = np.argsort(scores)[::-1][:k]
        rels = np.array([items[i]["relevance"] for i in order])

        dcg = float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))
        ideal_rels = np.sort([i["relevance"] for i in items])[::-1][:k]
        idcg = float(np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2))))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return {
        f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 4),
        "samples": len(by_query),
    }


def benchmark_correlation(
    data: List[Dict], model, tokenizer, cfg: Dict, use_mlx_embeddings: bool
) -> Dict:
    queries = [item["query"] for item in data]
    docs = [item["doc"] for item in data]
    labels = np.array([item["relevance"] for item in data])
    scores = rerank_scores(queries, docs, model, tokenizer, cfg, use_mlx_embeddings)

    corr, _ = spearmanr(labels, scores)
    return {
        "metric": "spearman_correlation",
        "value": round(float(corr), 4),
        "samples": len(data),
    }


def build_config(args) -> Dict:
    if args.config:
        raw = resolve_model_config(args.config)
    else:
        raw = resolve_model_config(args.model)

    cfg = {
        "model_id": args.model if args.model else raw["model_id"],
        "display_name": raw.get("display_name", raw["model_id"]),
        "loader": raw.get("loader", "mlx_lm"),
        "instruction": raw.get(
            "instruction",
            "Given a web search query, retrieve relevant passages that answer the query",
        ),
        "query_prefix": raw.get("query_prefix"),
        "document_prefix": raw.get("document_prefix"),
        "max_length": args.max_length,
        "reranker": raw.get("reranker", {
            "mode": "cross_encoder",
            "prompt_template": "qwen3",
            "yes_token": "yes",
            "no_token": "no",
            "fallback_on_missing_lm_head": False,
        }),
        "notes": raw.get("notes", ""),
    }

    if args.instruction is not None:
        cfg["instruction"] = args.instruction

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Model config name (e.g. qwen3_reranker_0.6b_4bit) or path to a YAML file",
    )
    parser.add_argument(
        "--model",
        help="Override model_id or local path from the config",
    )
    parser.add_argument(
        "--dataset",
        default="../datasets/rerank_pairs.jsonl",
        help="Rerank pairs dataset",
    )
    parser.add_argument("--output", default="../results/reranker_results.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--instruction", help="Override task instruction")
    args = parser.parse_args()

    if not args.config and not args.model:
        parser.error("One of --config or --model is required.")

    cfg = build_config(args)
    print(f"Loading reranker model: {cfg['model_id']}")
    print(f"  config={args.config or 'auto'}, mode={cfg['reranker']['mode']}, "
          f"loader={cfg['loader']}, instruction={cfg['instruction']!r}")
    if cfg["notes"]:
        print(f"  note: {cfg['notes']}")

    model, tokenizer, load_time, use_mlx_embeddings = load_model(cfg)
    print(f"  Active loader: {'mlx-embeddings' if use_mlx_embeddings else 'mlx_lm'}")

    data = load_jsonl(Path(args.dataset))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": cfg["model_id"],
        "display_name": cfg["display_name"],
        "load_time_s": round(load_time, 2),
        "max_length": cfg["max_length"],
        "instruction": cfg["instruction"],
        "reranker_mode": cfg["reranker"]["mode"],
        "use_mlx_embeddings": use_mlx_embeddings,
        "tasks": {},
    }

    print("\n[Pairwise Accuracy] Relevant vs irrelevant ranking...")
    results["tasks"]["pairwise_accuracy"] = benchmark_pairwise_accuracy(
        data, model, tokenizer, cfg, use_mlx_embeddings
    )
    print(f"  Accuracy: {results['tasks']['pairwise_accuracy']['value']}")

    print("\n[NDCG] Graded relevance ranking...")
    results["tasks"]["ndcg"] = benchmark_ndcg(
        data, model, tokenizer, cfg, use_mlx_embeddings, k=10
    )
    print(f"  NDCG@10: {results['tasks']['ndcg']['ndcg@10']}")

    print("\n[Correlation] Score-label correlation...")
    results["tasks"]["correlation"] = benchmark_correlation(
        data, model, tokenizer, cfg, use_mlx_embeddings
    )
    print(f"  Spearman correlation: {results['tasks']['correlation']['value']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
