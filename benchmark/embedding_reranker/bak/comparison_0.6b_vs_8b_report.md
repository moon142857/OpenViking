# Embedding / Reranker 0.6B vs 8B 对比测试报告

**生成时间：** 2026-06-26（reranker 部分于 2026-06-26 使用官方 Qwen3-Reranker prompt 重新测试并更新）  
**测试环境：** macOS Apple Silicon, Python 3.14, MLX 本地推理  
**Baseline 模型：** `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` / `mlx-community/Qwen3-Reranker-0.6B-4bit`  
**当前测试模型：** `Qwen3-Embedding-8B-4bit-DWQ` / `galaxycore/Qwen3-Reranker-8B-MLX-4bit`

---

## 1. 测试目的

将当前部署的 8B 4-bit 量化模型与 benchmark 目录内置的 0.6B baseline 进行横向对比，验证：

1. 8B 模型在功能评测上是否有明显提升；
2. 8B 模型在推理延迟和吞吐上牺牲了多少；
3. 为生产选型提供数据依据。

---

## 2. 评测方法

使用 `benchmark/embedding_reranker/scripts/` 下的同一套脚本：

- `benchmark_embedding.py`：STS、复述检测、检索召回、聚类
- `benchmark_reranker.py`：成对排序准确率、NDCG@10、Spearman 相关
- `benchmark_performance.py`：单条延迟、batch 吞吐、序列长度扩展

数据集相同，指标计算逻辑相同，仅替换模型路径。

---

## 3. Embedding 功能对比

| 任务 | 指标 | 0.6B Baseline | 8B 当前 | 变化 |
|------|------|---------------|---------|------|
| 语义文本相似度 | Spearman | **0.8927** | **0.8980** | +0.0053 ✅ |
| 复述检测 | Accuracy | 0.85 | **0.90** | +0.05 ✅ |
| 复述检测 | F1 | 0.88 | **0.9167** | +0.0367 ✅ |
| 检索召回 | Recall@1 | 0.9333 | 0.9333 | 持平 |
| 检索召回 | Recall@5 | 0.9333 | **1.0000** | +0.0667 ✅ |
| 检索召回 | Recall@10 | 1.0000 | 1.0000 | 持平 |
| 检索召回 | MRR | 0.9444 | **0.9667** | +0.0223 ✅ |
| 聚类 | V-measure | 1.0000 | 1.0000 | 持平 |

**模型加载时间：** 0.6B 1.08s → 8B 2.06s

### 3.1 分析

- **STS**：8B 略高（0.898 vs 0.893），差异较小，说明 0.6B 在简单语义相似度任务上已经很强。
- **复述检测**：8B 提升明显（Accuracy +5pp，F1 +3.7pp），在判断同义复述上更稳健。
- **检索召回**：Recall@1 持平，但 Recall@5 从 93.33% 提升到 100%，MRR 也有所提升。说明 8B 在前 5 名内召回更稳。
- **聚类**：两者都达到 1.0，小样本 3 类聚类任务太简单，无法区分差距。

---

## 4. Reranker 功能对比

> **重要更新**：首次测试时 8B 指标异常偏低，排查后发现是 `benchmark_reranker.py` 未使用官方 Qwen3-Reranker prompt。修正 prompt 后重新测试，结果如下。

### 4.1 首次测试结果（旧版 legacy prompt）

| 任务 | 指标 | 0.6B Baseline | 8B 当前 | 变化 |
|------|------|---------------|---------|------|
| 成对排序准确率 | Accuracy | **0.9091** | 0.7273 | -0.1818 ❌ |
| 分级相关性排序 | NDCG@10 | **0.9124** | 0.8875 | -0.0249 ❌ |
| 分数校准 | Spearman | 0.1900 | **0.1274** | -0.0626 ❌ |

### 4.2 修正 prompt 后结果（官方 Qwen3-Reranker prompt）

| 任务 | 指标 | 0.6B Baseline | 0.6B（qwen3 prompt） | 8B（qwen3 prompt） | 结论 |
|------|------|---------------|----------------------|---------------------|------|
| 成对排序准确率 | Accuracy | 0.9091 | **1.0000** | 0.9091 | 两者相当，0.6B 在此小样本上满分 |
| 分级相关性排序 | NDCG@10 | 0.9124 | 0.9864 | **0.9871** | 8B 略优，两者均大幅提升 |
| 分数校准 | Spearman | 0.1900 | 0.2502 | **0.4274** | **8B 显著更优** |

**模型加载时间：** 0.6B ~1.0s → 8B ~2.1s

### 4.3 分析

- **8B 并非更差，而是 prompt 不匹配**：使用官方 Qwen3-Reranker prompt 后，8B 的 NDCG@10 从 0.8875 提升到 **0.9871**，Spearman 从 0.1274 提升到 **0.4274**。
- **0.6B 也受益于官方 prompt**：Pairwise Accuracy 从 0.9091 提升到 **1.0000**，NDCG@10 从 0.9124 提升到 **0.9864**。
- **8B 的核心优势是分数校准**：Spearman 0.4274 明显高于 0.6B 的 0.2502，说明 8B 的绝对分数与人工相关度标签更一致，更适合需要阈值判断的场景。
- **测试样本较小**：rerank_pairs 只有 24 条、9 个 query，结论应作为参考，生产决策需结合真实业务数据。

---

## 5. 性能对比

### 5.1 Embedding 性能

| Batch Size | 0.6B 延迟 (ms) | 8B 延迟 (ms) | 0.6B 吞吐 (items/s) | 8B 吞吐 (items/s) | 吞吐下降 |
|------------|----------------|--------------|---------------------|-------------------|----------|
| 1 | 23.02 | 454.07 | 43.44 | 2.20 | **-94.9%** |
| 8 | 54.02 | 1236.00 | 148.09 | 6.47 | **-95.6%** |
| 16 | 98.79 | 2473.93 | 161.96 | 6.47 | **-96.0%** |
| 32 | 186.68 | 5042.09 | 171.42 | 6.35 | **-96.3%** |

**单条句子延迟：** 0.6B 23.07ms → 8B 460.95ms（**约 20 倍**）

### 5.2 Reranker 性能

| Batch Size | 0.6B 延迟 (ms) | 8B 延迟 (ms) | 0.6B 吞吐 (pairs/s) | 8B 吞吐 (pairs/s) | 吞吐下降 |
|------------|----------------|--------------|---------------------|-------------------|----------|
| 1 | 29.43 | 481.94 | 33.98 | 2.07 | **-93.9%** |
| 8 | 94.58 | 1990.57 | 84.58 | 4.02 | **-95.2%** |
| 16 | 168.83 | 3842.80 | 94.77 | 4.16 | **-95.6%** |
| 32 | 329.49 | 7314.63 | 97.12 | 4.37 | **-95.5%** |

**单对 query-doc 延迟：** 0.6B 29.76ms → 8B 497.70ms（**约 16.7 倍**）

### 5.3 Embedding 序列长度扩展

| 序列长度 | 0.6B 延迟 (ms) | 8B 延迟 (ms) | 倍数 |
|----------|----------------|--------------|------|
| 128 | 70.77 | 1723.78 | **24.4×** |
| 512 | 260.28 | 7443.22 | **28.6×** |
| 1024 | 532.22 | 13726.29 | **25.8×** |
| 2048 | 1143.83 | 28032.55 | **24.5×** |

### 5.4 性能分析

- 8B 模型的推理延迟和吞吐均比 0.6B 差约 **20 倍**，这与模型参数量增加（0.6B → 8B，约 13 倍）基本成正比。
- 在当前 Apple Silicon 本地部署场景下，8B 单条 embedding ~461ms、rerank ~498ms，**不适合高并发实时服务**。
- 如果必须使用 8B，建议：
  - 通过 batching 提升吞吐（batch=32 时 embedding 6.35 items/s，rerank 4.37 pairs/s）；
  - 或迁移到云端 GPU/Apple Silicon Max/Ultra 设备。

---

## 6. 综合结论

### 6.1 功能层面

| 维度 | 结论 |
|------|------|
| Embedding 语义能力 | 8B 略优，但提升有限（小样本任务上优势不明显） |
| Reranker 排序能力 | 修正 prompt 后 8B 与 0.6B 相当，NDCG@10 接近 |
| 分数校准 | **8B 更优**（Spearman 0.4274 vs 0.2502），但仍建议业务阈值调优 |

### 6.2 性能层面

| 维度 | 结论 |
|------|------|
| 单条延迟 | 8B 慢约 16–20 倍 |
| 吞吐 | 8B 下降约 94–96% |
| 内存占用 | 8B 占用显著更高（统一内存压力更大） |

### 6.3 是否推荐使用 8B？

| 场景 | 建议 |
|------|------|
| 本地开发 / 原型验证 / 低延迟场景 | **继续用 0.6B**，速度更快、资源占用更小 |
| 追求更高 Embedding 精度，且能容忍延迟 | 可尝试 8B embedding，但收益边际递减 |
| 当前 8B reranker（galaxycore）| 修正 prompt 后排序能力不输 0.6B，分数校准更好；代价是速度显著下降 |
| 生产高并发 | 建议上云端 GPU 或更大显存的 Apple Silicon 设备 |

---

## 7. 后续建议

1. **统一 Reranker 评测口径**：使用 `benchmark_reranker.py --prompt-mode qwen3` 对 0.6B / 4B / 8B 均采用官方 Qwen3-Reranker prompt，避免 prompt 不一致导致指标失真。
2. **使用真实业务数据评测**：内置小样本数据集只能做粗略参考，生产决策应以真实 query-doc 对为准。
3. **分数校准**：即使 8B Spearman 达到 0.4274，仍建议通过 Platt scaling 或业务阈值调优来映射分数。
4. **性能优化**：若坚持使用本地 8B，考虑 batching、并发队列、或升级硬件。
5. **本地模型路径**：Hub 下载失败时，可直接传入已下载的本地目录，例如 `/Users/zhengxiaoxi/.cache/huggingface/hub/Qwen3-Reranker-8B-MLX-4bit`。

---

*报告生成脚本：* `benchmark/embedding_reranker/scripts/run_all.py`  
*详细 JSON 结果：*
- `benchmark/embedding_reranker/results/embedding_results.json`（0.6B）
- `benchmark/embedding_reranker/results/reranker_results.json`（0.6B）
- `benchmark/embedding_reranker/results/performance_results.json`（0.6B）
- `benchmark/embedding_reranker/results/embedding_results_8b.json`（8B）
- `benchmark/embedding_reranker/results/reranker_results_8b.json`（8B）
- `benchmark/embedding_reranker/results/performance_results_8b.json`（8B）
- `benchmark/embedding_reranker/results/reranker_results_0.6b_qwen3_prompt.json`（0.6B 使用官方 prompt 的对照）
