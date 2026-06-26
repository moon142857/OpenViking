#!/usr/bin/env python3
"""Run a full suite of embedding+reranker combinations and generate a comparison report.

Usage:
    python scripts/run_suite.py
    python scripts/run_suite.py --suite default --output-dir ../results
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config.loader import load_model_config, load_suite_config


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
CONFIG_DIR = SCRIPT_DIR / "config"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"


def run_script(name: str, args: list) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / name)] + args
    print(f"\n{'='*60}")
    print(f"Running: {name} {' '.join(args)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def run_embedding(config_name: str, output_path: Path) -> bool:
    return run_script(
        "benchmark_embedding.py",
        [
            "--config", config_name,
            "--datasets", str(DATASETS_DIR),
            "--output", str(output_path),
        ],
    )


def run_reranker(config_name: str, output_path: Path) -> bool:
    return run_script(
        "benchmark_reranker.py",
        [
            "--config", config_name,
            "--dataset", str(DATASETS_DIR / "rerank_pairs.jsonl"),
            "--output", str(output_path),
        ],
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, decimals: int = 4) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def generate_report(suite_name: str, results: dict) -> str:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        f"# Embedding / Reranker 多模型对比报告：{suite_name}",
        "",
        f"**生成时间：** {timestamp}",
        "**测试环境：** macOS Apple Silicon, MLX 本地推理",
        "",
        "---",
        "",
        "## 1. 评测方法",
        "",
        "- Embedding 脚本：`scripts/benchmark_embedding.py`（配置驱动）",
        "- Reranker 脚本：`scripts/benchmark_reranker.py`（配置驱动）",
        "- 数据集：`datasets/` 内置小样本数据集",
        "- Reranker 统一使用官方 Qwen3-Reranker prompt 和 yes/no softmax 概率打分",
        "",
        "---",
        "",
        "## 2. Embedding 结果对比",
        "",
        "| 组合 | 模型 | Loader | STS Spearman | Paraphrase Acc | Paraphrase F1 | Recall@1 | Recall@5 | MRR | V-measure |",
        "|------|------|--------|--------------|----------------|---------------|----------|----------|-----|-----------|",
    ]

    for suite_key, data in results.items():
        emb = data.get("embedding")
        if emb is None:
            continue
        t = emb["tasks"]
        loader = emb.get("loader", "auto")
        lines.append(
            f"| {suite_key} | {emb.get('display_name', emb['model'])} | {loader} | "
            f"{_fmt(t['sts']['value'])} | {_fmt(t['paraphrase']['accuracy'])} | "
            f"{_fmt(t['paraphrase']['f1'])} | {_fmt(t['retrieval']['recall@1'])} | "
            f"{_fmt(t['retrieval']['recall@5'])} | {_fmt(t['retrieval']['mrr'])} | "
            f"{_fmt(t['clustering']['value'])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Reranker 结果对比",
        "",
        "| 组合 | 模型 | Loader | Mode | Pairwise Acc | NDCG@10 | Spearman |",
        "|------|------|--------|------|--------------|---------|----------|",
    ])

    for suite_key, data in results.items():
        rnk = data.get("reranker")
        if rnk is None:
            continue
        t = rnk["tasks"]
        loader = "mlx-embeddings" if rnk.get("use_mlx_embeddings") else "mlx_lm"
        mode = rnk.get("reranker_mode", "cross_encoder")
        lines.append(
            f"| {suite_key} | {rnk.get('display_name', rnk['model'])} | {loader} | {mode} | "
            f"{_fmt(t['pairwise_accuracy']['value'])} | {_fmt(t['ndcg']['ndcg@10'])} | "
            f"{_fmt(t['correlation']['value'])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. 模型配置说明",
        "",
        "| 组合 | Embedding 配置 | Reranker 配置 | 说明 |",
        "|------|----------------|---------------|------|",
    ])

    for suite_key, data in results.items():
        meta = data.get("meta", {})
        lines.append(
            f"| {suite_key} | `{meta.get('embedding', '-')}` | `{meta.get('reranker', '-')}` | "
            f"{meta.get('name', '-')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. 结论与注意事项",
        "",
        "- 所有 `mlx-community/Qwen3-Embedding-*` 模型都应使用 `mlx-embeddings` 加载器；",
        "  使用 `mlx_lm` 会导致 MTEB 等标准评测上的 embedding 输出接近随机（~0.09）。",
        "- `mlx-community/Qwen3-Reranker-*-4bit` 是官方 cross-encoder，使用 `mlx_lm` 加载并以 yes/no logit 打分；",
        "  `mlx-community/Qwen3-Reranker-*-mxfp8` 按官方 mlx-embeddings 用法为 bi-encoder，",
        "  使用 query/document embedding 的 cosine similarity 打分。",
        "- 实测 `Qwen3-Reranker-0.6B-mxfp8` 在 bi-encoder 模式下表现较差；",
        "  `Qwen3-Reranker-8B-mxfp8` 可用但通常仍弱于 0.6B-4bit cross-encoder。",
        "- 内置数据集样本量小，结论仅用于快速横向对比，生产决策请用真实业务数据或 MTEB 官方数据集。", 
        "",
        "---",
        "",
        "*报告生成脚本：* `benchmark/embedding_reranker/scripts/run_suite.py`",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="default", help="Suite config name (default: default)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite_config(args.suite)
    results = {}

    for suite_key, spec in suite["suites"].items():
        print(f"\n\n{'#'*60}")
        print(f"# Suite: {suite_key} - {spec['name']}")
        print(f"{'#'*60}")

        emb_config = spec["embedding"]
        rnk_config = spec["reranker"]

        emb_output = output_dir / f"embedding_{suite_key}.json"
        rnk_output = output_dir / f"reranker_{suite_key}.json"

        ok_emb = run_embedding(emb_config, emb_output)
        ok_rnk = run_reranker(rnk_config, rnk_output)

        results[suite_key] = {
            "meta": {
                "name": spec["name"],
                "embedding": emb_config,
                "reranker": rnk_config,
            },
            "embedding": load_json(emb_output) if ok_emb else None,
            "reranker": load_json(rnk_output) if ok_rnk else None,
        }

    report = generate_report(args.suite, results)
    report_path = output_dir / f"comparison_report_{args.suite}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Suite benchmark complete. Report saved to:")
    print(f"  {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
