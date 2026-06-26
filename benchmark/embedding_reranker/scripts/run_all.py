#!/usr/bin/env python3
"""Run all embedding/reranker benchmarks and generate baseline report.

Usage:
    python scripts/run_all.py \
        --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
        --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"


def run_script(name: str, args: list) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / name)] + args
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def generate_report(embedding_results: dict, reranker_results: dict, perf_results: dict) -> str:
    report = f"""# Embedding / Reranker Baseline Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Embedding Model:** {embedding_results['model']}
**Reranker Model:** {reranker_results['model']}

---

## 1. Methodology

This benchmark uses representative built-in datasets to evaluate embedding and reranker models
in a local MLX environment. The tests are designed to be lightweight and reproducible, suitable
as a baseline for comparing different model sizes and quantization strategies.

### 1.1 Embedding Evaluation Tasks

| Task | Metric | Description |
|------|--------|-------------|
| STS | Spearman correlation | Semantic textual similarity between sentence pairs |
| Paraphrase | Accuracy / F1 | Binary classification of paraphrase pairs |
| Retrieval | Recall@k / MRR | Finding relevant documents from a small corpus |
| Clustering | V-measure | Grouping sentences by topic using embeddings |

### 1.2 Reranker Evaluation Tasks

| Task | Metric | Description |
|------|--------|-------------|
| Pairwise Accuracy | Accuracy | Probability that relevant docs score higher than irrelevant docs |
| NDCG@10 | NDCG | Graded relevance ranking quality |
| Correlation | Spearman | Correlation between model scores and human labels |

### 1.3 Performance Evaluation

| Metric | Description |
|--------|-------------|
| Single latency | Inference time for one input |
| Batch throughput | Items/pairs processed per second at various batch sizes |
| Sequence length scaling | Latency across different input lengths |

---

## 2. Embedding Results

**Model:** `{embedding_results['model']}`
**Load time:** {embedding_results['load_time_s']}s

| Task | Metric | Value | Samples |
|------|--------|-------|---------|
| STS | Spearman correlation | {embedding_results['tasks']['sts']['value']} | {embedding_results['tasks']['sts']['samples']} |
| Paraphrase | Accuracy | {embedding_results['tasks']['paraphrase']['accuracy']} | {embedding_results['tasks']['paraphrase']['samples']} |
| Paraphrase | F1 | {embedding_results['tasks']['paraphrase']['f1']} | {embedding_results['tasks']['paraphrase']['samples']} |
| Retrieval | Recall@1 | {embedding_results['tasks']['retrieval']['recall@1']} | {embedding_results['tasks']['retrieval']['samples']} |
| Retrieval | Recall@5 | {embedding_results['tasks']['retrieval']['recall@5']} | {embedding_results['tasks']['retrieval']['samples']} |
| Retrieval | MRR | {embedding_results['tasks']['retrieval']['mrr']} | {embedding_results['tasks']['retrieval']['samples']} |
| Clustering | V-measure | {embedding_results['tasks']['clustering']['value']} | {embedding_results['tasks']['clustering']['samples']} |

---

## 3. Reranker Results

**Model:** `{reranker_results['model']}`
**Load time:** {reranker_results['load_time_s']}s

| Task | Metric | Value | Samples |
|------|--------|-------|---------|
| Pairwise Accuracy | Accuracy | {reranker_results['tasks']['pairwise_accuracy']['value']} | {reranker_results['tasks']['pairwise_accuracy']['samples']} |
| NDCG | NDCG@10 | {reranker_results['tasks']['ndcg']['ndcg@10']} | {reranker_results['tasks']['ndcg']['samples']} |
| Correlation | Spearman | {reranker_results['tasks']['correlation']['value']} | {reranker_results['tasks']['correlation']['samples']} |

---

## 4. Performance Results

### 4.1 Embedding Throughput

| Batch Size | Avg Latency (ms) | Throughput (items/sec) |
|------------|-----------------|------------------------|
"""
    for k, v in perf_results['embedding']['batch'].items():
        bs = k.replace('batch_', '')
        report += f"| {bs} | {v['avg_ms']} | {v['throughput_items_per_sec']} |\n"

    report += f"\n**Single sentence latency:** {perf_results['embedding']['single_latency_ms']['avg']} ms (avg)\n\n"

    report += "### 4.2 Reranker Throughput\n\n"
    report += "| Batch Size | Avg Latency (ms) | Throughput (pairs/sec) |\n"
    report += "|------------|-----------------|------------------------|\n"
    for k, v in perf_results['reranker']['batch'].items():
        bs = k.replace('batch_', '')
        report += f"| {bs} | {v['avg_ms']} | {v['throughput_pairs_per_sec']} |\n"

    report += f"\n**Single pair latency:** {perf_results['reranker']['single_latency_ms']['avg']} ms (avg)\n\n"

    report += "### 4.3 Embedding Sequence Length Scaling\n\n"
    report += "| Sequence Length | Latency (ms) |\n"
    report += "|-----------------|-------------|\n"
    for k, v in perf_results['embedding']['seq_length'].items():
        length = k.replace('length_', '')
        report += f"| {length} | {v['latency_ms']} |\n"

    report += """
---

## 5. Conclusions

This baseline establishes the performance and quality characteristics of the 0.6B MLX models.
When testing larger models, compare against these metrics:

- **Embedding quality:** Spearman correlation on STS, retrieval Recall@1/Recall@5, clustering V-measure
- **Reranker quality:** Pairwise accuracy, NDCG@10, Spearman correlation
- **Speed:** Single/batch latency and throughput
- **Memory:** Observe peak RAM during model loading and inference

### Observations for 0.6B Baseline

- The 0.6B model provides fast inference suitable for interactive local deployments.
- Retrieval and clustering tasks demonstrate practical semantic understanding.
- Larger models are expected to improve STS correlation and retrieval recall, at the cost of latency and memory.

---

*Report generated by benchmarks/embedding_reranker/scripts/run_all.py*
"""
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model",
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    parser.add_argument(
        "--reranker-model",
        default="mlx-community/Qwen3-Reranker-0.6B-4bit",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run embedding benchmark
    ok1 = run_script(
        "benchmark_embedding.py",
        [
            "--model", args.embedding_model,
            "--datasets", str(DATASETS_DIR),
            "--output", str(RESULTS_DIR / "embedding_results.json"),
        ],
    )

    # Run reranker benchmark
    ok2 = run_script(
        "benchmark_reranker.py",
        [
            "--model", args.reranker_model,
            "--dataset", str(DATASETS_DIR / "rerank_pairs.jsonl"),
            "--output", str(RESULTS_DIR / "reranker_results.json"),
        ],
    )

    # Run performance benchmark
    ok3 = run_script(
        "benchmark_performance.py",
        [
            "--embedding-model", args.embedding_model,
            "--reranker-model", args.reranker_model,
            "--output", str(RESULTS_DIR / "performance_results.json"),
        ],
    )

    if not all([ok1, ok2, ok3]):
        print("\nSome benchmarks failed. See output above.")
        sys.exit(1)

    # Load results and generate report
    with open(RESULTS_DIR / "embedding_results.json", "r", encoding="utf-8") as f:
        emb = json.load(f)
    with open(RESULTS_DIR / "reranker_results.json", "r", encoding="utf-8") as f:
        rnk = json.load(f)
    with open(RESULTS_DIR / "performance_results.json", "r", encoding="utf-8") as f:
        perf = json.load(f)

    report = generate_report(emb, rnk, perf)
    report_path = RESULTS_DIR / "baseline_0.6b_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"All benchmarks complete. Baseline report saved to:")
    print(f"  {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
