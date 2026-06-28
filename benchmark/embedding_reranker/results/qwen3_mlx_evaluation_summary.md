# Qwen3 MLX Embedding/Reranker 评测现状、问题修复与测试结果汇总

> 综合调研文档 `benchmark/qwen3_mlx_evaluation_report/qwen3_mlx_evaluation_report.md` 与当前实现 `benchmark/embedding_reranker/` 的对比分析，以及实际跑通的关键 MTEB 测试结果。  
> 生成时间：2026-06-27

---

## 1. 当前资源与模型现状

### 1.1 已下载模型（~18.4 GB）

| 模型 | 大小 | 对应配置 | 状态 |
|------|------|----------|------|
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 335M | low_memory_4bit / balanced | ✅ |
| `mlx-community/Qwen3-Embedding-0.6B-mxfp8` | 600M | low_memory_mxfp8 | ✅ |
| `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 2.1G | balanced | ✅ |
| `mlx-community/Qwen3-Embedding-8B-mxfp8` | 7.3G | high_quality | ✅ |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | 331M | low_memory_4bit / balanced | ✅ |
| `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | 600M | low_memory_mxfp8 | ✅ |
| `mlx-community/Qwen3-Reranker-8B-mxfp8` | 7.3G | high_quality | ✅ |

### 1.2 资源约束

- **磁盘**：460G 总计，已用 371G，**可用 55G**。足够下载所有推荐数据集。
- **内存**：32G 总容量（系统报告可用约 28.2G），当前空闲紧张但 Inactive 可释放。8B+8B 组合模型约 16G，加上索引可能接近 21G，需关闭其他应用。
- **数据集缓存**：已占用 3.6G，其中 SciFact、NFCorpus、AskUbuntu、STSBenchmark 等已可用。

---

## 2. 已修复的关键脚本问题

| 文件 | 问题 | 修复内容 |
|------|------|----------|
| `scripts/benchmark_embedding.py` | Retrieval 文档和 Clustering 文本被重复/错误地加上 query instruction | 让 `embed_texts()` 统一负责格式化；`benchmark_retrieval()` 传 `document_prefix` 给文档、`query_prefix` 给查询；`benchmark_clustering()` 传 `query_prefix` 一次 |
| `scripts/mteb_benchmark.py` | MTEB 2.16 下未根据 `prompt_type` 应用 query instruction，检索任务分数偏低 | 根据 `prompt_type`（query/document）自动选择 `query_prefix` 或 `document_prefix` |
| `scripts/mteb_benchmark.py` | 运行 MTEB 时反复尝试连接 HuggingFace Hub，导致超时卡住 | 已通过环境变量 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 跑通 |
| `requirements.txt` | 缺少 `mteb`、`mlx-embeddings`、`datasets`、`pyyaml` | 已补齐 |
| `scripts/summarize_mteb.py` | 结果中的 `NaN` 未处理 | 新增 `_fmt_score()`，将 `NaN`/`None` 转为 `"-"` |

### 2.1 验证：instruction 修复对 SciFact 的显著提升

| 配置 | SciFact nDCG@10（max_length=256） | 说明 |
|------|-----------------------------------|------|
| 0.6B 4bit（修复前，无 instruction） | 0.6590 | 文档/查询均不加前缀 |
| 0.6B 4bit（修复后，查询加 instruction） | **0.6867** | 提升 **+2.77** 个百分点 |
| 0.6B mxfp8（修复后） | 0.6849 | 与 4bit 基本持平 |

---

## 3. 已跑通的关键 MTEB 测试结果

### 3.1 Embedding 模型

| 模型 | STSBenchmark Spearman | SciFact nDCG@10 | NFCorpus nDCG@10 | 备注 |
|------|----------------------|-----------------|------------------|------|
| Qwen3-Embedding-0.6B-4bit-DWQ | 0.8451 | 0.6867 | 0.3430 | 低内存首选 |
| Qwen3-Embedding-0.6B-mxfp8 | 0.8453 | 0.6849 | - | 与 4bit 精度相当，速度略慢 |
| Qwen3-Embedding-4B-4bit-DWQ | 0.8692 | - | - | STS 上明显优于 0.6B |

### 3.2 Reranker 模型

| 模型 | 任务 | 主指标 | 说明 |
|------|------|--------|------|
| Qwen3-Reranker-0.6B-mxfp8 (bi-encoder) | AskUbuntuDupQuestions | MAP@20 = 0.4275 | bi-encoder 形态，效果一般 |

### 3.3 关键发现

1. **0.6B 4bit 与 mxfp8 几乎没有精度差异**：SciFact 0.6867 vs 0.6849，STSBenchmark 0.8451 vs 0.8453。4bit 更快，是低内存场景首选。
2. **4B Embedding 在 STS 上明显优于 0.6B**：0.8692 vs 0.8451，提升约 2.4 个百分点，与调研文档中 "balanced 配置显著提升" 的结论一致。
3. **0.6B 模型输出维度为 1024**，不是调研文档中提到的 2048（可能 4B/8B 才输出更高维）。
4. **Retrieval 任务本地速度远低于调研文档预估**：
   - 0.6B 4bit 在 max_length=256 时约 **10-11 docs/s**。
   - SciFact（5183 docs）约 11 分钟；NFCorpus（3633 docs）约 7 分钟。
   - 按比例估算，FiQA2018（57,638 docs）单模型需 **~90 分钟**，4B/8B 更久。

---

## 4. 数据集下载与缓存状态

### 4.1 调研文档推荐数据集

| 优先级 | 数据集 | 类型 | 当前状态 | 备注 |
|--------|--------|------|----------|------|
| P0 | SciFact | Retrieval | ✅ 已缓存，已跑通 | corpus/queries/default 三配置齐全 |
| P0 | FiQA2018 | Retrieval | ✅ 已通过 HF Mirror 下载 | corpus: 57,638 / queries: 6,648 |
| P0 | NFCorpus | Retrieval | ✅ 已通过 HF Mirror 下载并跑通 | corpus: 3,633 / queries: 3,237 |
| P0 | SciDocsRR | Reranking | ✅ 已缓存 | 仅 default 配置，适合 Reranking |
| P1 | ArguAna | Retrieval | ⚠️ 旧缓存格式不兼容 | 只有 default 配置，运行会报错 |
| P1 | SCIDOCS | Retrieval | ⚠️ 旧缓存格式不兼容 | 同上 |
| P1 | TRECCOVID | Retrieval | ❌ 未下载 | 数据量大，优先级降低 |
| P1 | AskUbuntuDupQuestions | Reranking | ✅ 已缓存，已跑通 | - |
| P2 | STSBenchmark | STS | ✅ 已缓存，已跑通 | - |
| P2 | Banking77 | Classification | ❌ 未下载 | P2 优先级低 |

### 4.2 旧缓存格式问题

MTEB 2.16 的 Retrieval 任务需要数据集同时包含 `default`、`corpus`、`queries` 三个配置。当前本地缓存中：
- ✅ SciFact 已完整
- ✅ FiQA2018、NFCorpus 已通过 HF Mirror 刷新为完整格式
- ⚠️ ArguAna、SCIDOCS 等仍为旧版 `default` 单配置，运行时报错：
  ```
  Couldn't find cache for mteb/arguana for config 'corpus'
  ```

**建议**：如需跑 ArguAna / SCIDOCS，删除旧缓存后用 HF Mirror 重新下载：
```bash
rm -rf ~/.cache/huggingface/datasets/mteb___arguana
rm -rf ~/.cache/huggingface/datasets/mteb___scidocs
export HF_ENDPOINT=https://hf-mirror.com
python -c "from datasets import load_dataset; load_dataset('mteb/arguana', 'corpus'); load_dataset('mteb/arguana', 'queries')"
```

---

## 5. 尚未实现/仍存在的问题

### 5.1 尚未实现的评测维度

| 维度 | 状态 | 说明 |
|------|------|------|
| Matryoshka 维度对比 | ❌ 未实现 | 需测试 256/512/1024/2048 维对检索的影响 |
| 端到端 RAG 评测 | ❌ 未实现 | Embedding → Reranker 串联，计算 nDCG@10 提升 |
| 性能基准测试 | ⚠️ 不可靠 | `benchmark_performance.py` 暂存 `bak/`，README 已标注 |
| MTEB Cross-Encoder Reranker | ❌ 未实现 | `mteb_benchmark.py` 仅支持 bi-encoder，无法评测效果最好的 0.6B-4bit cross-encoder |

### 5.2 仍需关注的问题

1. **本地 Retrieval 速度过慢**：
   - 实际速度约 10 docs/s（0.6B，max_length=256），远低于调研文档预估。
   - 跑完整 FiQA2018 单模型需约 90 分钟，4B/8B 模型时间更长。
   - 建议：筛选 FiQA2018 子集做快速验证，或仅在 GPU 环境跑完整任务。

2. **Reranker 形态差异**：
   - 0.6B-4bit 是官方 cross-encoder，效果最好，但 MTEB 集成未实现。
   - mxfp8 系列是 bi-encoder，效果差（AskUbuntu MAP@20 仅 0.43）。
   - 当前 `benchmark_reranker.py` 的 cross-encoder 是逐条推理，未实现 batch，大模型评测极慢。

3. **网络依赖**：
   - 虽然已设置离线环境变量，但首次下载数据集仍需稳定网络。
   - 建议使用 `HF_ENDPOINT=https://hf-mirror.com` 加速。

4. **内存压力**：
   - 跑 8B + 8B high_quality 组合时，需确保系统内存充足，关闭其他应用。

---

## 6. 推荐后续执行计划

### 6.1 短期（已可执行）

1. **跑 NFCorpus 0.6B mxfp8 / 4B 4bit**：与 0.6B 4bit 做横向对比（各约 7-30 分钟）。
2. **跑 SciFact 4B 4bit**：验证 4B 在检索任务上的提升（预估 40-50 分钟）。
3. **跑 SciDocsRR / AskUbuntu 的 0.6B-4bit cross-encoder**：需要先在 `mteb_benchmark.py` 中实现 `MLXCrossEncoderRankerForMTEB`。
4. **实现 Matryoshka 维度测试**：在 SciFact 上测试 256/512/1024 维（注意 0.6B 实际输出 1024 维）。

### 6.2 中期

1. **实现端到端 RAG 评测**：复用 `mteb_benchmark.py` 中的 encoder 和 `benchmark_reranker.py` 的 cross-encoder 打分逻辑。
2. **恢复性能基准脚本**：基于 `mlx-embeddings` 重写 `benchmark_performance.py`，覆盖 batch 吞吐、延迟、内存峰值。
3. **补充中文评测**：如教育场景涉及中文，接入 C-MTEB（T2Retrieval、MMarcoReranking）。

### 6.3 长期

1. 在 GPU 或云端环境跑完整 P0/P1 数据集（FiQA2018 全量、TRECCOVID 等）。
2. 建立持续基准测试，跟踪不同模型版本/量化格式的效果。

---

## 7. 可直接复用的运行命令

```bash
cd benchmark/embedding_reranker
source ~/mlx-env/bin/activate

# 离线环境变量（必须）
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 推荐：使用 HF Mirror 下载缺失数据集
export HF_ENDPOINT=https://hf-mirror.com

# Embedding 模型 MTEB 测试
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks SciFact,NFCorpus \
  --max-length 256 \
  --output results/mteb_embedding_0.6b_4bit_rag.json

# 汇总报告
python scripts/summarize_mteb.py \
  --results-dir results \
  --output results/mteb_report.md
```

---

## 8. 结论

当前已建立一个可运行的 Qwen3 MLX Embedding/Reranker 基准测试流程，修复了脚本中的关键 bug，并跑通了 SciFact、NFCorpus、STSBenchmark 等核心 MTEB 任务。主要结论：

- **balanced 配置（4B Embedding + 0.6B Reranker）在 STS 上明显优于 low_memory**，符合调研文档推荐。
- **0.6B 4bit 与 mxfp8 精度基本持平，4bit 更快更省内存**，低内存场景首选 4bit。
- **查询 instruction 对 Retrieval 任务至关重要**，修复后 SciFact 提升 2.77 个百分点。
- **本地 Mac 跑完整 Retrieval 数据集（尤其 FiQA2018）非常耗时**，建议优先跑小数据集/子集，或在 GPU 环境跑全量。
- **MTEB Cross-Encoder Reranker 集成、Matryoshka 测试、端到端 RAG 评测、性能基准脚本** 仍是后续需要补齐的关键维度。

