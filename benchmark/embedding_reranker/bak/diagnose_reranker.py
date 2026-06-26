#!/usr/bin/env python3
"""诊断两个 reranker 模型在同一数据集上的打分差异。"""

import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_pairs(queries, docs, model, tokenizer, use_logit_diff=True):
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
    if use_logit_diff:
        scores = last_logits[:, yes_id] - last_logits[:, no_id]
    else:
        scores = last_logits[:, yes_id]
    return np.array(scores.astype(mx.float32))


def main():
    data = load_jsonl(Path(__file__).parent.parent / "datasets" / "rerank_pairs.jsonl")

    models = {
        "0.6B": "mlx-community/Qwen3-Reranker-0.6B-4bit",
        "8B": "/Users/zhengxiaoxi/.cache/huggingface/hub/Qwen3-Reranker-8B-MLX-4bit",
    }

    for name, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        model, tokenizer = load(model_path)

        for use_diff in [True, False]:
            label = "logit_diff" if use_diff else "yes_logit"
            print(f"\n--- Scoring method: {label} ---")
            queries = [d["query"] for d in data]
            docs = [d["doc"] for d in data]
            labels = np.array([d["relevance"] for d in data])
            scores = score_pairs(queries, docs, model, tokenizer, use_diff)

            print(f"{'Q':<25} {'Rel':>4} {'Score':>10}")
            print("-" * 45)
            for i, item in enumerate(data):
                q_short = item["query"][:22] + "..." if len(item["query"]) > 25 else item["query"]
                print(f"{q_short:<25} {item['relevance']:>4} {scores[i]:>10.4f}")

            # Pairwise accuracy
            by_query = defaultdict(list)
            for item, score in zip(data, scores):
                by_query[item["query"]].append({**item, "score": float(score)})

            correct = 0
            total = 0
            for query, items in by_query.items():
                rel = [i for i in items if i["relevance"] >= 2]
                irrel = [i for i in items if i["relevance"] == 0]
                if not rel or not irrel:
                    continue
                total += len(rel) * len(irrel)
                correct += int(np.sum(
                    np.array([r["score"] for r in rel])[:, None] >
                    np.array([i["score"] for i in irrel])[None, :]
                ))
            print(f"\nPairwise accuracy: {correct}/{total} = {correct/total:.4f}" if total else "No pairs")

            # Spearman
            from scipy.stats import spearmanr
            corr, _ = spearmanr(labels, scores)
            print(f"Spearman correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
