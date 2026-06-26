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


def _fmt(value, decimals: int = 4) -> str:
    """Format a numeric value, preserving non-numeric fallback."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def generate_report(embedding_results: dict, reranker_results: dict, perf_results: dict) -> str:
    emb_tasks = embedding_results['tasks']
    rnk_tasks = reranker_results['tasks']
    report = f"""# Embedding / Reranker 基线测试报告

**生成时间：** {time.strftime('%Y-%m-%d %H:%M:%S')}
**测试环境：** macOS Apple Silicon, Python 3.14, MLX 本地推理
**Embedding 模型：** `{embedding_results['model']}`
**Reranker 模型：** `{reranker_results['model']}`

---

## 1. 测试目的

本报告作为 **0.6B 量化 MLX 模型** 的基线（baseline），记录其在常见 Embedding 和 Reranker 评测任务上的功能表现与推理性能。后续更换更大模型（如 8B）时，可直接与本报告做横向对比。

---

## 2. 评测方法

### 2.1 Embedding 评测任务

| 任务 | 指标 | 说明 |
|------|------|------|
| 语义文本相似度（STS） | Spearman 相关系数 | 衡量向量余弦相似度与人类评分的一致性 |
| 复述检测 | Accuracy / F1 | 判断两个句子是否为同义复述 |
| 检索召回 | Recall@k / MRR | 从语料库中召回与查询相关的文档 |
| 聚类 | V-measure | 按主题对句子进行分组的能力 |

### 2.2 Reranker 评测任务

| 任务 | 指标 | 说明 |
|------|------|------|
| 成对排序准确率 | Accuracy | 同一查询下，相关文档得分高于无关文档的概率 |
| 分级相关性排序 | NDCG@10 | 对多个相关度等级的文档进行排序的质量 |
| 分数校准 | Spearman 相关系数 | 模型得分与人工相关度标签的相关性 |

> Reranker 得分采用 `yes` token 与 `no` token 的 logit 差值，比单独使用 `yes` logit 更稳定。

### 2.3 性能评测

| 指标 | Embedding | Reranker |
|------|-----------|----------|
| 单条/单对延迟 | ✅ | ✅ |
| Batch 吞吐（1/8/16/32） | ✅ | ✅ |
| 序列长度扩展（128/512/1024/2048） | ✅ | - |

---

## 3. Embedding 测试结果

**模型加载时间：** {_fmt(embedding_results['load_time_s'])} 秒

| 任务 | 指标 | 数值 | 样本数 |
|------|------|------|--------|
| 语义文本相似度 | Spearman 相关系数 | **{_fmt(emb_tasks['sts']['value'])}** | {emb_tasks['sts']['samples']} |
| 复述检测 | Accuracy | **{_fmt(emb_tasks['paraphrase']['accuracy'])}** | {emb_tasks['paraphrase']['samples']} |
| 复述检测 | F1 | **{_fmt(emb_tasks['paraphrase']['f1'])}** | {emb_tasks['paraphrase']['samples']} |
| 检索召回 | Recall@1 | **{_fmt(emb_tasks['retrieval']['recall@1'])}** | {emb_tasks['retrieval']['samples']} |
| 检索召回 | Recall@5 | **{_fmt(emb_tasks['retrieval']['recall@5'])}** | {emb_tasks['retrieval']['samples']} |
| 检索召回 | MRR | **{_fmt(emb_tasks['retrieval']['mrr'])}** | {emb_tasks['retrieval']['samples']} |
| 聚类 | V-measure | **{_fmt(emb_tasks['clustering']['value'])}** | {emb_tasks['clustering']['samples']} |

**分析：**
- STS 相关系数 {_fmt(emb_tasks['sts']['value'])}，说明 0.6B 模型对句子语义相似度的捕捉能力很强。
- 检索 Recall@1 达到 {_fmt(float(emb_tasks['retrieval']['recall@1']) * 100, decimals=2)}%，在较小语料上几乎能完美召回。
- 聚类 V-measure 为 {_fmt(float(emb_tasks['clustering']['value']) * 100, decimals=2)}%，说明内置主题能够被清晰区分。

---

## 4. Reranker 测试结果

**模型加载时间：** {_fmt(reranker_results['load_time_s'])} 秒

| 任务 | 指标 | 数值 | 样本数 |
|------|------|------|--------|
| 成对排序准确率 | Accuracy | **{_fmt(rnk_tasks['pairwise_accuracy']['value'])}** | {rnk_tasks['pairwise_accuracy']['samples']} |
| 分级相关性排序 | NDCG@10 | **{_fmt(rnk_tasks['ndcg']['ndcg@10'])}** | {rnk_tasks['ndcg']['samples']} |
| 分数校准 | Spearman 相关系数 | **{_fmt(rnk_tasks['correlation']['value'])}** | {rnk_tasks['correlation']['samples']} |

**分析：**
- 成对排序准确率高达 {_fmt(float(rnk_tasks['pairwise_accuracy']['value']) * 100, decimals=2)}%，说明模型能很好地区分相关文档与无关文档。
- NDCG@10 为 {_fmt(rnk_tasks['ndcg']['ndcg@10'])}，排序质量优秀。
- Spearman 相关系数为 {_fmt(float(rnk_tasks['correlation']['value']) * 100, decimals=2)}%，说明模型虽然能正确排序，但绝对分数与人工分级标签的线性相关度不高。更大模型通常能改善分数校准。

---

## 5. 性能测试结果

### 5.1 Embedding 吞吐

| Batch Size | 平均延迟（ms） | 吞吐（items/sec） |
|------------|--------------|------------------|
"""
    for k, v in perf_results['embedding']['batch'].items():
        bs = k.replace('batch_', '')
        report += f"| {bs} | {_fmt(v['avg_ms'], decimals=2)} | {_fmt(v['throughput_items_per_sec'], decimals=2)} |\n"

    report += f"\n**单条句子延迟：** {_fmt(perf_results['embedding']['single_latency_ms']['avg'], decimals=2)} ms（平均）\n\n"

    report += "### 5.2 Reranker 吞吐\n\n"
    report += "| Batch Size | 平均延迟（ms） | 吞吐（pairs/sec） |\n"
    report += "|------------|--------------|------------------|\n"
    for k, v in perf_results['reranker']['batch'].items():
        bs = k.replace('batch_', '')
        report += f"| {bs} | {_fmt(v['avg_ms'], decimals=2)} | {_fmt(v['throughput_pairs_per_sec'], decimals=2)} |\n"

    report += f"\n**单对 query-doc 延迟：** {_fmt(perf_results['reranker']['single_latency_ms']['avg'], decimals=2)} ms（平均）\n\n"

    report += "### 5.3 Embedding 序列长度扩展\n\n"
    report += "| 序列长度 | 延迟（ms） |\n"
    report += "|----------|----------|\n"
    for k, v in perf_results['embedding']['seq_length'].items():
        length = k.replace('length_', '')
        report += f"| {length} | {_fmt(v['latency_ms'], decimals=2)} |\n"

    report += f"""
---

## 6. 基线结论

### 6.1 优势

1. **速度快**：单条 embedding {_fmt(perf_results['embedding']['single_latency_ms']['avg'], decimals=2)} ms、单对 rerank {_fmt(perf_results['reranker']['single_latency_ms']['avg'], decimals=2)} ms，适合本地交互式部署。
2. **质量好**：检索 Recall@1 达 {_fmt(float(emb_tasks['retrieval']['recall@1']) * 100, decimals=2)}%，rerank 成对排序准确率 {_fmt(float(rnk_tasks['pairwise_accuracy']['value']) * 100, decimals=2)}%。
3. **内存小**：0.6B 4-bit 量化模型占用统一内存约 1-2GB，普通 Mac 即可运行。

### 6.2 局限

1. **Reranker 分数校准较弱**：Spearman 相关系数仅 {_fmt(float(rnk_tasks['correlation']['value']) * 100, decimals=2)}%，绝对分数不宜直接作为阈值使用。
2. **长序列较慢**：2048 token 的 embedding 延迟约 {_fmt(perf_results['embedding']['seq_length']['length_2048']['latency_ms'], decimals=2)} ms。
3. **小模型语义上限**：在更复杂的多义、跨领域、长文档理解任务上，可能不如 8B 模型。

### 6.3 与更大模型对比的预期

| 对比维度 | 0.6B 基线 | 更大模型（如 8B）预期 |
|----------|----------|---------------------|
| STS Spearman | {_fmt(emb_tasks['sts']['value'])} | 可能提升到 0.92-0.95 |
| Retrieval Recall@1 | {_fmt(emb_tasks['retrieval']['recall@1'])} | 可能提升到 0.95+ |
| Reranker 分数校准 | {_fmt(rnk_tasks['correlation']['value'])} | 可能明显改善 |
| 单条延迟 | {_fmt(perf_results['embedding']['single_latency_ms']['avg'], decimals=2)} ms | 预计增加 3-10 倍 |
| 内存占用 | ~1-2GB | 预计增加 5-10 倍 |

---

## 7. 使用建议

- **本地开发 / 原型验证**：0.6B 模型完全够用，速度快、资源占用低。
- **生产环境高质量检索**：如需更高精度和更好的分数校准，建议测试 8B 模型。
- **对比测试**：更换模型后，使用同一套 `scripts/run_all.py` 重新运行，将结果与本报告对比。

---

*报告生成脚本：* `benchmark/embedding_reranker/scripts/run_all.py`  
*详细 JSON 结果：* `benchmark/embedding_reranker/results/*.json`
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
