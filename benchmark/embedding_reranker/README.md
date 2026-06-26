# Embedding / Reranker 模型基准测试

本目录包含一套可复用的 Embedding 和 Reranker 模型评测方案，基于 MLX 本地推理，用于对 `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` 和 `mlx-community/Qwen3-Reranker-0.6B-4bit` 进行功能与性能测试。

> 设计目标：作为基准（baseline），后续更换更大模型时，可用同一套方法做对比分析。

---

## 目录结构

```
benchmark/embedding_reranker/
├── README.md                       # 本说明文档
├── requirements.txt                # Python 依赖
├── datasets/                       # 内置测试数据集
│   ├── sts_sample.jsonl            # 语义相似度（STS）
│   ├── paraphrase_sample.jsonl     # 复述检测
│   ├── retrieval_corpus.jsonl      # 检索语料
│   ├── retrieval_queries.jsonl     # 检索查询
│   ├── rerank_pairs.jsonl          # 重排序 query-doc 对
│   └── clustering_sample.jsonl     # 聚类
├── scripts/                        # 测试脚本
│   ├── benchmark_embedding.py      # Embedding 功能测试
│   ├── benchmark_reranker.py       # Reranker 功能测试
│   ├── benchmark_performance.py    # 性能测试
│   └── run_all.py                  # 一键运行全部测试
└── results/                        # 测试结果
    ├── embedding_results.json
    ├── reranker_results.json
    ├── performance_results.json
    └── baseline_0.6b_report.md
```

---

## 1. 主流评测方法调研

### 1.1 Embedding 模型主流评测

| 评测方向 | 代表基准 | 常用指标 | 说明 |
|----------|---------|---------|------|
| 语义文本相似度 | STS-B, SICK-R, STS12-16 | Spearman correlation | 衡量向量余弦相似度与人类打分的一致性 |
| 复述检测 | Quora Question Pairs, PAWS | Accuracy / F1 | 判断两个句子是否同义 |
| 信息检索 | MS MARCO, BEIR, NQ | Recall@k, MRR, nDCG@k | 用向量召回相关文档 |
| 聚类 | 20 Newsgroups, StackExchange | V-measure, NMI | 按主题聚类 |
| 分类 | Amazon Reviews, Banking77 | Accuracy / F1 | 句子分类 |
| 跨语言 | XTREME, LaBSE eval | 多语言对齐指标 | 多语言语义对齐 |

### 1.2 Reranker 模型主流评测

| 评测方向 | 代表基准 | 常用指标 | 说明 |
|----------|---------|---------|------|
|  passage 重排序 | MS MARCO passage ranking | MRR@10, nDCG@10 | 对检索结果重新排序 |
| 成对排序 | MTEB Reranking | MAP, NDCG | query-doc 对相关性排序 |
| 相关性校准 | 人工标注数据集 | Spearman / Pearson | 模型分数与人类标注的相关性 |

### 1.3 本方案设计思路

为了兼顾**可复现性**和**本地运行效率**，本方案没有直接跑完整的 MTEB/BEIR（下载和运行成本高），而是：

1. 内置小样本数据集（中英双语），覆盖主流任务类型；
2. 使用与 OpenViking 服务一致的 `mlx_lm` 推理代码；
3. 输出结构化 JSON，方便后续与更大模型做横向对比；
4. 性能测试覆盖单条延迟、batch 吞吐、序列长度扩展。

> 如需更权威的评测，可将本脚本中的数据集替换为 MTEB/BEIR 子集，指标计算逻辑保持不变。

---

## 2. 环境准备

```bash
# 进入目录
cd benchmark/embedding_reranker

# 激活 MLX 环境（与 OpenViking 部署共用）
source ~/mlx-env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> 国内用户可设置 HF 镜像加速模型下载：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 3. 运行测试

### 3.1 一键运行全部

```bash
python scripts/run_all.py \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

运行结束后会在 `results/` 目录生成：
- `embedding_results.json`
- `reranker_results.json`
- `performance_results.json`
- `baseline_0.6b_report.md`

### 3.2 单独运行 Embedding 评测

```bash
python scripts/benchmark_embedding.py \
  --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --datasets ./datasets \
  --output ./results/embedding_results.json
```

### 3.3 单独运行 Reranker 评测

```bash
python scripts/benchmark_reranker.py \
  --model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --dataset ./datasets/rerank_pairs.jsonl \
  --output ./results/reranker_results.json
```

### 3.4 单独运行性能评测

```bash
python scripts/benchmark_performance.py \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --output ./results/performance_results.json
```

---

## 4. 测试方法详解

### 4.1 Embedding 功能测试

#### STS（Semantic Textual Similarity）

- 输入：句子对 + 人工相似度分数（0-5）
- 方法：计算两句子向量余弦相似度，缩放到 0-5
- 指标：**Spearman correlation**

#### 复述检测（Paraphrase Detection）

- 输入：句子对 + 是否复述标签（0/1）
- 方法：用余弦相似度做二分类，网格搜索最佳阈值
- 指标：**Accuracy / F1**

#### 检索召回（Retrieval Recall）

- 输入：语料库 + 查询 + 相关文档标注
- 方法：将查询和文档分别编码，按余弦相似度排序
- 指标：**Recall@1 / Recall@5 / Recall@10 / MRR**

#### 聚类（Clustering）

- 输入：带类别标签的句子
- 方法：先编码，再用类别中心做最近邻分类
- 指标：**V-measure**

### 4.2 Reranker 功能测试

#### 成对排序准确率（Pairwise Accuracy）

- 输入：同一查询下的相关文档和无关文档
- 方法：比较 reranker 分数，看相关文档是否更高
- 指标：**Accuracy**

#### NDCG@10

- 输入：同一查询下的多个 query-doc 对 + 分级相关度（0-3）
- 方法：按 reranker 分数排序，计算 DCG / IDCG
- 指标：**NDCG@10**

#### 相关性校准（Correlation）

- 输入：所有 query-doc 对 + 分级相关度
- 方法：计算 reranker 分数与人工标签的 Spearman 相关
- 指标：**Spearman correlation**

> 本方案使用 `yes` 与 `no` token 的 logit 差值作为相关性分数，比单用 `yes` logit 更稳定。

### 4.3 性能测试

| 测试项 | Embedding | Reranker |
|--------|-----------|----------|
| 单条延迟 | 1 条句子 | 1 个 query-doc 对 |
| Batch 吞吐 | batch=1,8,16,32 | batch=1,8,16,32 |
| 序列长度扩展 | 128/512/1024/2048 | - |

---

## 5. 0.6B 模型基线结果

### 5.1 Embedding 结果

| 任务 | 指标 | 数值 | 样本数 |
|------|------|------|--------|
| STS | Spearman correlation | **0.8927** | 20 |
| Paraphrase | Accuracy | **0.85** | 20 |
| Paraphrase | F1 | **0.88** | 20 |
| Retrieval | Recall@1 | **0.9333** | 15 |
| Retrieval | Recall@5 | **0.9333** | 15 |
| Retrieval | MRR | **0.9444** | 15 |
| Clustering | V-measure | **1.0000** | 20 |

### 5.2 Reranker 结果

| 任务 | 指标 | 数值 | 样本数 |
|------|------|------|--------|
| Pairwise Accuracy | Accuracy | **0.9091** | 11 |
| NDCG | NDCG@10 | **0.9124** | 9 |
| Correlation | Spearman | **0.1900** | 24 |

> Reranker 的 Pairwise Accuracy 和 NDCG 表现优秀，但 Spearman correlation 较低，说明模型能正确排序，但绝对分数与人工分级相关度不高。更大模型通常能改善分数校准。

### 5.3 性能结果

#### Embedding 吞吐

| Batch Size | 平均延迟 (ms) | 吞吐 (items/sec) |
|------------|--------------|------------------|
| 1 | 23.2 | 43.1 |
| 8 | 54.2 | 147.7 |
| 16 | 99.1 | 161.5 |
| 32 | 186.9 | 171.2 |

**单条延迟：** 23.0 ms

#### Reranker 吞吐

| Batch Size | 平均延迟 (ms) | 吞吐 (pairs/sec) |
|------------|--------------|------------------|
| 1 | 29.4 | 34.0 |
| 8 | 94.5 | 84.6 |
| 16 | 173.9 | 92.0 |
| 32 | 330.1 | 97.0 |

**单对延迟：** 29.8 ms

#### Embedding 序列长度扩展

| 序列长度 | 延迟 (ms) |
|----------|----------|
| 128 | 70.8 |
| 512 | 260.3 |
| 1024 | 532.2 |
| 2048 | 1143.8 |

---

## 6. 如何对比更大模型

更换模型后，只需重新运行：

```bash
python scripts/run_all.py \
  --embedding-model mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --reranker-model mlx-community/Qwen3-Reranker-8B-4bit
```

建议对比维度：

| 维度 | 关注指标 |
|------|---------|
| 语义质量 | STS Spearman、Retrieval Recall@1/5、Clustering V-measure |
| 重排序质量 | Pairwise Accuracy、NDCG@10 |
| 速度 | 单条延迟、Batch 吞吐 |
| 内存 | 模型加载时间、峰值显存/统一内存 |
| 扩展性 | 长序列延迟 |

---

## 7. 扩展建议

1. **接入 MTEB：** 将 `datasets` 依赖取消注释，用 `mteb` 库加载标准数据集替换内置小样本数据。
2. **增加中文评测：** 当前数据集已包含中英双语，可进一步补充 CLUE、LCQMC 等中文任务。
3. **增加内存监控：** 在性能脚本中加入 `mx.metal.get_peak_memory()` 或 `torch.mps` 内存统计。
4. **校准 Reranker 分数：** 对更大模型可尝试温度缩放或 Platt scaling 改善 Spearman correlation。

---

*最后更新: 2026-06-26*
