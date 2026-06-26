#!/usr/bin/env python3
"""Aggregate MTEB result JSON files into a Markdown report."""

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(data: dict) -> dict:
    """Extract task -> main_score from an MTEB result JSON."""
    scores = {}
    for task_name, task_result in data.get("results", {}).items():
        scores[task_name] = task_result.get("main_score")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--output", default="../results/mteb_report.md")
    parser.add_argument("--pattern", default="mteb_*.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob(args.pattern))
    if not files:
        print(f"No MTEB result files found in {results_dir} matching {args.pattern}")
        return

    # Merge scores by model, summing elapsed time for files from the same model.
    model_data = {}
    all_tasks = set()
    for path in files:
        data = load_json(path)
        model = data.get("display_name") or data.get("model", path.stem)
        elapsed = data.get("elapsed_s", 0) or 0
        scores = extract_scores(data)
        all_tasks.update(scores.keys())
        if model not in model_data:
            model_data[model] = {"scores": {}, "elapsed_s": 0}
        model_data[model]["scores"].update(scores)
        model_data[model]["elapsed_s"] += elapsed

    all_tasks = sorted(all_tasks)

    lines = [
        "# MTEB 轻量评测报告",
        "",
        f"**结果目录:** `{results_dir}`",
        "",
        "## 模型任务得分",
        "",
        "| Model | " + " | ".join(all_tasks) + " | Total elapsed (s) |",
        "|" + "---|" * (len(all_tasks) + 2),
    ]

    for model, info in sorted(model_data.items()):
        cells = [model]
        for task in all_tasks:
            score = info["scores"].get(task)
            cells.append(f"{score:.4f}" if score is not None else "-")
        cells.append(f"{info['elapsed_s']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 说明",
        "",
        "- 使用 `scripts/mteb_benchmark.py` 在本地 MLX (macOS Apple Silicon) 上运行。",
        "- 分数为对应任务的 `main_score`（通常为 cosine_spearman）。",
        "- 受限于本地算力，当前仅完成轻量 STS 任务；Retrieval / Reranking 任务耗时较长，建议后续在 GPU 环境或分批运行。",
    ])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MTEB report saved to {output_path}")


if __name__ == "__main__":
    main()
