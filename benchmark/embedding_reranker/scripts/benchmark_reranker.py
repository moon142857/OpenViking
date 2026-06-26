#!/usr/bin/env python3
"""Reranker model functional benchmark.

Evaluates reranker models on:
- Pair-wise ranking accuracy (relevant vs irrelevant)
- NDCG-style ranking with graded relevance labels
- Spearman correlation with human relevance labels

Usage:
    python scripts/benchmark_reranker.py \
        --model mlx-community/Qwen3-Reranker-0.6B-4bit \
        --dataset ../datasets/rerank_pairs.jsonl \
        --output ../results/reranker_results.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.stats import spearmanr


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rerank_scores(queries: List[str], docs: List[str], model, tokenizer) -> np.ndarray:
    pairs = [[q, d] for q, d in zip(queries, docs)]
    inputs = tokenizer._tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=512,
    )
    input_ids = mx.array(inputs["input_ids"].astype(np.int32))
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    yes_id = tokenizer._tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer._tokenizer.convert_tokens_to_ids("no")
    last_logits = logits[:, -1, :]
    # Use logit difference (yes - no) as relevance score.
    scores = last_logits[:, yes_id] - last_logits[:, no_id]
    return np.array(scores.astype(mx.float32))


def benchmark_pairwise_accuracy(data: List[Dict], model, tokenizer) -> Dict:
    """For each query, check if relevant docs score higher than irrelevant docs."""
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
        scores = rerank_scores(q_list, d_list, model, tokenizer)

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


def benchmark_ndcg(data: List[Dict], model, tokenizer, k: int = 10) -> Dict:
    by_query = defaultdict(list)
    for item in data:
        by_query[item["query"]].append(item)

    ndcg_scores = []
    for query, items in by_query.items():
        q_list = [query] * len(items)
        d_list = [i["doc"] for i in items]
        scores = rerank_scores(q_list, d_list, model, tokenizer)

        # Sort by score descending
        order = np.argsort(scores)[::-1][:k]
        rels = np.array([items[i]["relevance"] for i in order])

        # DCG
        dcg = float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))
        # Ideal DCG
        ideal_rels = np.sort([i["relevance"] for i in items])[::-1][:k]
        idcg = float(np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2))))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return {
        f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 4),
        "samples": len(by_query),
    }


def benchmark_correlation(data: List[Dict], model, tokenizer) -> Dict:
    queries = [item["query"] for item in data]
    docs = [item["doc"] for item in data]
    labels = np.array([item["relevance"] for item in data])
    scores = rerank_scores(queries, docs, model, tokenizer)

    corr, _ = spearmanr(labels, scores)
    return {
        "metric": "spearman_correlation",
        "value": round(float(corr), 4),
        "samples": len(data),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Reranker-0.6B-4bit",
        help="Reranker model ID or local path",
    )
    parser.add_argument(
        "--dataset",
        default="../datasets/rerank_pairs.jsonl",
        help="Rerank pairs dataset",
    )
    parser.add_argument("--output", default="../results/reranker_results.json")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading reranker model: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    data = load_jsonl(Path(args.dataset))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "load_time_s": round(load_time, 2),
        "max_length": args.max_length,
        "tasks": {},
    }

    print("\n[Pairwise Accuracy] Relevant vs irrelevant ranking...")
    results["tasks"]["pairwise_accuracy"] = benchmark_pairwise_accuracy(data, model, tokenizer)
    print(f"  Accuracy: {results['tasks']['pairwise_accuracy']['value']}")

    print("\n[NDCG] Graded relevance ranking...")
    results["tasks"]["ndcg"] = benchmark_ndcg(data, model, tokenizer, k=10)
    print(f"  NDCG@10: {results['tasks']['ndcg']['ndcg@10']}")

    print("\n[Correlation] Score-label correlation...")
    results["tasks"]["correlation"] = benchmark_correlation(data, model, tokenizer)
    print(f"  Spearman correlation: {results['tasks']['correlation']['value']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
