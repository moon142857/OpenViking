# Qwen3 MLX Embedding/Reranker 评测现状与问题分析

> 基于调研文档 `benchmark/qwen3_mlx_evaluation_report/qwen3_mlx_evaluation_report.md` 与当前实现 `benchmark/embedding_reranker/` 的对比分析。  
> 生成时间：2026-06-27

---

## 1. 当前资源现状

### 1.1 已下载模型（~18.4 GB）

| 模型 | 大小 | 用途 | 状态 |
|------|------|------|------|
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 335M | Embedding low_memory_4bit | ✅ 已下载 |
| `mlx-community/Qwen3-Embedding-0.6B-mxfp8` | 600M | Embedding low_memory_mxfp8 | ✅ 已下载 |
| `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 2.1G | Embedding balanced | ✅ 已下载 |
| `mlx-community/Qwen3-Embedding-8B-mxfp8` | 7.3G | Embedding high_quality | ✅ 已下载 |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | 331M | Reranker low_memory_4bit / balanced | ✅ 已下载 |
| `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | 600M | Reranker low_memory_mxfp8 | ✅ 已下载 |
| `mlx-community/Qwen3-Reranker-8B-mxfp8` | 7.3G | Reranker high_quality | ✅ 已下载 |

### 1.2 磁盘与内存

- **磁盘**：460G 总计，已用 371G，**可用 55G**。
- **内存**：总容量 32G（系统报告可用约 28.2G），当前空闲紧张（~0.7G），但 Inactive 13.4G 可被系统回收。
- **数据集缓存**：`~/.cache/huggingface/datasets` 已占用 **3.6G**。

### 1.3 已安装依赖

- `mteb==2.16.1`
- `mlx-embeddings` 已安装
- `mlx-lm` 已安装
- `requirements.txt` 缺少 `mteb` 和 `mlx-embeddings` 的显式声明。

---

## 2. 调研文档推荐 vs 当前实现差距

### 2.1 推荐数据集覆盖

调研文档将 MTEB 数据集按 RAG 教育场景优先级分为 P0 / P1 / P2：

| 优先级 | 类型 | 数据集 | 当前缓存 | 当前已测试 | 问题 |
|--------|------|--------|----------|------------|------|
| P0 | Retrieval | **SciFact** | ✅ | 🔄 进行中 | 离线可跑 |
| P0 | Retrieval | **FiQA2018** | ❌ 未缓存 | ❌ | 需重新下载 |
| P0 | Retrieval | **NFCorpus** | ✅（仅 default） | ❌ | MTEB 2.16 需要 corpus/queries 分片，缓存格式不兼容 |
| P0 | Reranking | **SciDocsRR** | ✅（仅 default） | ❌ | 同上，缓存格式可能不兼容 |
| P1 | Retrieval | ArguAna | ✅（仅 default） | ❌ | 已验证 MTEB 2.16 无法使用旧缓存 |
| P1 | Retrieval | TRECCOVID | ❌ | ❌ | 未缓存，数据量较大 |
| P1 | Retrieval | SCIDOCS | ✅（仅 default） | ❌ | 缓存格式不兼容 |
| P1 | Reranking | AskUbuntu | ✅ | ✅（0.6B mxfp8） | 可跑 |
| P2 | STS | STSBenchmark | ✅ | ✅ | 可跑 |
| P2 | Classification | Banking77 | ❌ | ❌ | 未缓存 |

**关键发现**：
- 仅 **SciFact** 的本地缓存同时包含 `default` / `corpus` / `queries` 三个配置，能直接被 MTEB 2.16 用于 Retrieval 任务。
- 其余 Retrieval 数据集（NFCorpus、ArguAna、SCIDOCS、SciDocsRR 等）的旧缓存只有 `default` 配置，运行时会报错 `Couldn't find cache for mteb/xxx for config 'corpus'`。
- **FiQA2018** 是 P0 中唯一既关键又完全缺失的数据集，需要优先下载。

### 2.2 评测维度覆盖

| 维度 | 调研文档建议 | 当前实现 | 差距 |
|------|--------------|----------|------|
| 内置小样本功能测试 | Phase 1 快速验证 | ✅ `benchmark_embedding.py` / `benchmark_reranker.py` | 无 |
| MTEB 官方数据集轻量评测 | P0/P1/P2 任务 | ⚠️ `mteb_benchmark.py` 已接入，但受缓存/网络/格式限制 | 部分可跑 |
| Matryoshka 维度对比 | Phase 6，测试 256/512/1024/2048 维 | ❌ 未实现 | 缺失 |
| 端到端 RAG 评测 | Phase 5，Embedding → Reranker 串联 | ❌ 未实现 | 缺失 |
| 性能基准测试 | Phase 7，吞吐量/延迟/内存 | ⚠️ 脚本暂存 `bak/`，README 标注不可靠 | 缺失 |
| 量化格式精度对比 | 4bit-DWQ vs mxfp8 | ⚠️ 仅通过不同配置间接对比 | 未系统对比 |

---

## 3. 脚本实现中的具体问题

### 3.1 `scripts/benchmark_embedding.py` — 查询前缀被错误复用

**问题**：`embed_texts()` 对所有输入统一应用 `query_prefix`，导致：
- **Retrieval 文档**被加上检索指令前缀（`Instruct: ...\nQuery: <doc>`）。
- **Clustering 文本**在 `benchmark_clustering()` 中已被 `format_query` 处理一次，进入 `embed_texts()` 后又被 `format_query` 处理一次，形成双重前缀。

**影响**：内置数据集的检索、聚类指标不能反映真实用法，分数可能虚高或虚低。

**修复方向**：`embed_texts()` 应区分 query / document / sentence，分别使用对应的 prefix，或允许调用方传入 `prefix_template=None` 来禁用二次格式化。

### 3.2 `scripts/mteb_benchmark.py` — MTEB 2.16 下未按 prompt_type 应用 instruction

**问题**：`MLXEncoderForMTEB.encode()` 默认 `apply_instruction=False`，且未使用 `prompt_type`（query/passage）来决定是否加指令。Qwen3 Embedding 官方推荐对 query 加 instruction、对 document 不加。

**影响**：Retrieval 任务得分可能低于官方最佳实践。

**修复方向**：根据 `prompt_type` 自动选择 prefix：
- `prompt_type == "query"` → 使用 `query_prefix` + instruction
- `prompt_type == "passage"` 或无 → 原文本或 `document_prefix`

### 3.3 `scripts/mteb_benchmark.py` — ModelMeta 元数据不准确

**问题**：`build_model_meta()` 中 `framework=["PyTorch"]`，与 MLX 本地推理不符。

**影响**：元数据上报/结果展示不准确，长期可能影响 leaderboard 或对比报告。

**修复方向**：改为 `framework=["MLX"]`。

### 3.4 MTEB 运行时网络超时问题

**问题**：即使数据集已缓存，`mteb.evaluate()` 仍会尝试连接 HuggingFace Hub 检查更新，导致长时间重试（实测单个任务可能卡 10 分钟以上）。

**解决方案**：运行时必须设置离线环境变量：
```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 3.5 数据集缓存格式与 MTEB 2.16 不兼容

**问题**：MTEB 2.16 的 Retrieval / Reranking 任务要求数据集以 `corpus` / `queries` / `default` 多配置形式缓存。当前本地缓存中仅 SciFact 符合，其余均为旧版 `default` 单配置。

**验证结果**：
- ArguAna 运行报错：`Couldn't find cache for mteb/arguana for config 'corpus'`。
- NFCorpus / SCIDOCS / SciDocsRR 等大概率同样问题。

**解决方案**：
- 删除旧缓存，重新联网下载（需稳定 HuggingFace 连接或 HF-Mirror）。
- 或降级 MTEB 到支持旧缓存的版本（不推荐，会引入其他兼容性问题）。
- 或自行将旧缓存转换为新格式（工作量大）。

### 3.6 `scripts/summarize_mteb.py` — 未处理 NaN

**问题**：Reranker 结果中可能出现 `NaN`（如 `nauc_recall_at_20_max`），`json.dump` 默认会写入 `NaN`，但 Markdown 汇总时可能显示为 `nan` 或导致后续处理异常。

**修复方向**：汇总前将 `NaN`/`None` 替换为 `"-"` 或空值。

### 3.7 `requirements.txt` 缺失关键依赖

**问题**：`requirements.txt` 未列出 `mteb` 和 `mlx-embeddings`，但脚本已依赖它们。

**修复方向**：添加 `mteb>=2.0` 和 `mlx-embeddings`。

### 3.8 `scripts/benchmark_reranker.py` — 无 batch 推理

**问题**：cross-encoder reranker 逐条构造 prompt 并前向传播，未实现 batch。MTEB 中 8B reranker 的评测会非常慢。

**修复方向**：对 cross-encoder 实现 batch tokenize + batch forward。

### 3.9 MTEB Reranker 仅支持 bi-encoder 模式

**问题**：`mteb_benchmark.py` 中的 `MLXBiEncoderRankerForMTEB` 仅实现了 embedding similarity，无法评测官方 cross-encoder 的 `Qwen3-Reranker-0.6B-4bit`。

**影响**：无法对效果最好的 0.6B-4bit cross-encoder 跑 MTEB Reranking 任务。

**修复方向**：增加 `MLXCrossEncoderRankerForMTEB`，复用 `benchmark_reranker.py` 中的 yes/no logit 打分逻辑。

---

## 4. 基于资源约束的关键数据集下载建议

当前磁盘仍有 **55G 可用**，推荐按以下顺序有限下载：

| 优先级 | 数据集 | 类型 | 预估大小 | 下载必要性 | 说明 |
|--------|--------|------|----------|------------|------|
| 1 | **FiQA2018** | Retrieval P0 | ~15MB | 🔴 必须 | 调研文档 P0，政策/文案检索场景；当前完全缺失 |
| 2 | **NFCorpus** | Retrieval P0 | ~3MB | 🟡 建议重下载 | 已有旧缓存但格式不兼容，需刷新为新格式 |
| 3 | **SCIDOCS** | Retrieval P1 | ~50MB | 🟡 建议重下载 | 旧缓存格式不兼容 |
| 4 | **SciDocsRR** | Reranking P0 | ~50MB | 🟡 建议重下载 | 旧缓存格式可能不兼容 |
| 5 | **TRECCOVID** | Retrieval P1 | ~500MB-1G | 🟢 可选 | 长文档检索压力测试；数据量大，优先级降低 |
| 6 | **Banking77** | Classification P2 | ~2MB | 🟢 可选 | 基础语义理解参考，P2 优先级低 |

**不推荐现在下载**：
- MSMARCO：虽已在缓存中，但完整版数据量大，且非调研文档 P0/P1 推荐。
- TRECCOVID：除非明确需要测试长文档检索，否则可延后。

---

## 5. 推荐测试执行计划

### 5.1 第一阶段：修复脚本 + 跑通 P0（1-2 小时）

1. 设置离线环境变量。
2. 修复 `mteb_benchmark.py` 的 `prompt_type` 处理与 `framework` 元数据。
3. 修复 `benchmark_embedding.py` 的 query prefix 重复问题。
4. 下载/刷新 FiQA2018、NFCorpus、SciDocsRR。
5. 跑通以下任务：
   - Embedding: SciFact、FiQA2018、NFCorpus
   - Reranking: SciDocsRR、AskUbuntuDupQuestions
   - 模型：至少 0.6B 4bit、0.6B mxfp8、4B 4bit

### 5.2 第二阶段：补充 P1 + 中文/端到端（2-4 小时）

1. 跑 ArguAna、SCIDOCS、TRECCOVID（可选）。
2. 实现并跑端到端 RAG 评测（Embedding + Reranker 串联）。
3. 增加 Matryoshka 维度对比（512/1024/2048）。
4. 如教育场景涉及中文，补充 C-MTEB 中的 T2Retrieval / MMarcoReranking。

### 5.3 第三阶段：性能基准（1 小时）

1. 恢复/重写 `benchmark_performance.py`，基于 `mlx-embeddings` 实现 batch 吞吐与延迟测试。
2. 覆盖四组套件，记录内存峰值。

---

## 6. 已知风险与注意事项

1. **网络不稳定**：当前 HuggingFace 连接超时/拒绝，所有 MTEB 任务必须离线运行。若需下载新数据集，建议使用 HF-Mirror 或稳定网络时段。
2. **内存压力**：跑 8B + 8B high_quality 组合时，模型本身约 16G，加上向量索引可能接近 21G。建议关闭其他应用，并监控系统内存。
3. **MTEB 2.16 兼容性**：旧缓存数据集需要重新下载；`mteb.evaluate()` API 与旧版不同，脚本需持续适配。
4. **Reranker 形态差异**：0.6B-4bit 是 cross-encoder（效果最好），mxfp8 系列是 bi-encoder（效果差），不能混在一起比较。
5. **时间成本**：Retrieval 任务比 STS 慢一个数量级，4B/8B 模型更慢，建议先在小模型上验证脚本，再扩大模型。

---

## 7. 结论

当前实现已经完成了内置数据集评测和 MTEB 轻量接入，但与调研文档的完整方案相比，仍有明显差距：

- **最紧迫**：修复 `benchmark_embedding.py` 的 prefix 错误、`mteb_benchmark.py` 的 instruction 与元数据问题，并统一使用离线环境变量。
- **最关键**：优先下载/刷新 **FiQA2018、NFCorpus、SciDocsRR** 三个 P0 数据集。
- **最缺失**：Matryoshka 维度评测、端到端 RAG 评测、性能基准测试尚未实现。
- **最大阻塞**：旧版 MTEB 数据集缓存与 MTEB 2.16 不兼容，导致大部分 Retrieval / Reranking 任务无法直接离线运行。

建议先集中完成第一阶段（修复 + P0 数据集 + P0 测试），获得可复现的基准结果后，再逐步扩展第二阶段和第三阶段。
