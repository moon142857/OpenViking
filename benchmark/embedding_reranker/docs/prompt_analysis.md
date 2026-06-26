# Qwen3 Embedding / Reranker 0.6B 官方提示词与当前测试对比分析

> 分析时间：2026-06-26  
> 分析对象：`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`、`mlx-community/Qwen3-Reranker-0.6B-4bit`  
> 说明：本文仅做分析，未修改任何代码。

---

## 0. 测试模型信息（本机缓存）

| 模型 | 本地缓存路径 | 缓存大小 |
|------|-------------|---------|
| Embedding | `/Users/xx/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-0.6B-4bit-DWQ` | **341 MB** |
| Reranker | `/Users/xx/.cache/huggingface/hub/models--mlx-community--Qwen3-Reranker-0.6B-4bit` | **337 MB** |

> 注：以上大小为 Hugging Face 缓存目录中的实际占用，用于在另一台机器上做模型占用与部署对比分析。

---

## 1. 官方推荐用法

### 1.1 Embedding 模型（Qwen3-Embedding-0.6B）

官方仓库与模型卡给出的标准推理流程：

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [get_detailed_instruct(task, 'What is the capital of China?')]
documents = ["The capital of China is Beijing."]

# 1. query 必须加 instruction，document 不加
# 2. tokenizer 使用 padding_side='left'
# 3. pooling 使用 last-token pooling（取最后一个有效 token 的 hidden state）
# 4. 对 embeddings 做 L2 normalize
# 5. 相似度用 normalized embeddings 的点积（等价于 cosine）
```

> 官方提示：该模型是 **Instruction Aware**，大多数下游任务中加 instruction 可提升 1%-5%；多语言场景建议用英文 instruction。

### 1.2 Reranker 模型（Qwen3-Reranker-0.6B）

官方用法是一个 **causal-LM 风格的 yes/no 打分器**，必须使用特定 chat template：

```python
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# 得分是 P("yes")，即在 yes/no 上的 softmax 概率
batch_scores = torch.stack([false_vector, true_vector], dim=1)
batch_scores = F.log_softmax(batch_scores, dim=1)
scores = batch_scores[:, 1].exp()
```

> 注意：该 reranker **不是** `bge-reranker-v2-m3` 那种 cross-encoder，必须依赖上述 chat template 才能正确输出 yes/no 概率。

---

## 2. 当前测试脚本与官方用法的差距

### 2.1 `benchmark_embedding.py`

| 维度 | 官方推荐 | 当前实现 | 是否匹配 |
|------|----------|----------|----------|
| Query prompt | `Instruct: ...\nQuery:...` | 无 instruction，直接用原始文本 | ❌ |
| Document prompt | 不加 instruction | 无 instruction（正确） | ✅ |
| Pooling 方式 | last-token pooling | mean pooling | ❌ |
| Embedding 归一化 | L2 normalize | 未归一化 | ❌ |
| Padding side | `left` | 未指定，默认通常 right | ❌ |
| 相似度计算 | cosine（归一化后点积） | sklearn cosine_similarity（未归一化） | ⚠️ |
| STS 缩放 | 无此操作 | 将 cosine 乘以 5 | ❌ |

**关键问题：**
- `mean pooling` 与官方 `last-token pooling` 会得到不同的向量表示。
- 未做 L2 normalize，虽然 `cosine_similarity` 内部会计算 cosine，但输出尺度与官方不完全一致。
- **未加 instruction 对性能影响最大**：官方明确说 embedding 模型是 instruction-aware，不加 instruction 通常会掉 1%-5%。

### 2.2 `benchmark_reranker.py`

| 维度 | 官方推荐 | 当前实现 | 是否匹配 |
|------|----------|----------|----------|
| Chat template | 必须加 system/user/assistant 模板 | 无，直接 tokenize `[query, doc]` | ❌ |
| 输入格式 | `<Instruct>:\n<Query>:\n<Document>:` | 无格式 | ❌ |
| 任务 instruction | `Given a web search query...` | 无 | ❌ |
| 得分计算 | `softmax(yes, no)[:, yes]` 概率 | `logit_yes - logit_no` 差值 | ❌ |

**关键问题：**
- Reranker 是 causal LM，**chat template 是核心**。没有 system prompt 和固定格式，模型不会按训练时的方式输出 yes/no。
- 当前用 `logit_yes - logit_no` 虽然比单用 yes logit 稳定，但与官方 `P(yes)` 概率仍不完全等价。
- 这很可能是当前报告中 **Spearman 相关系数仅 0.19** 的重要原因——模型能区分相关/无关（pairwise accuracy 90.91%），但绝对分数没有与人工标签对齐。

---

## 3. 是否达到最佳效果？

**结论：没有。**

当前测试更像是**功能/性能冒烟测试**，而不是官方标准的质量评估。具体表现：

| 模型 | 当前能达到的效果 | 理论上可提升空间 |
|------|------------------|------------------|
| Embedding | 基本功能正常，Retrieval Recall@1 0.93（小数据集） | 按官方格式加 instruction、last-token pooling、normalize，通常可提升 1-5% |
| Reranker | Pairwise accuracy 90.91%，NDCG@10 0.91 | 使用官方 chat template 和 yes/no 概率后，分数校准（Spearman）应显著改善 |

---

## 4. 与官方 MTEB 分数的间接对比

官方 `Qwen3-Embedding-0.6B` 在标准 MTEB 上的分数：

| 任务 | 官方 MTEB(eng, v2) | 当前测试 |
|------|---------------------|----------|
| STS | 86.57 | 89.27（Spearman，20 条样本，不可直接比） |
| Retrieval | 61.83 | Recall@1 0.9333（15 条查询，极小语料） |
| Clustering | 54.05 | V-measure 1.0（3 类 20 条，过于简单） |

**说明：**
- 当前测试数据集太小、太简单，无法与 MTEB 直接比较。
- 0.6B 官方 Retrieval 才 61.83，我们 0.9333 是因为语料只有 20 篇文档且主题差异极大。
- 官方 Clustering 54.05，我们 1.0 也是因为只有 3 个非常明显的类别。

---

## 5. 综合评估

| 方面 | 评价 |
|------|------|
| **提示词匹配度** | 不匹配。Embedding 缺 instruction、reranker 缺 chat template。 |
| **推理格式正确性** | 不完全正确。Pooling、归一化、padding、reranker 得分方式均有偏差。 |
| **测试有效性** | 作为功能验证和本地性能基线足够，但作为质量基准不够严谨。 |
| **是否达到最佳效果** | 没有。Embedding 至少损失 1-5% 的 instruction 收益；Reranker 的分数校准明显受 prompt 影响。 |
| **后续换大模型对比** | 如果保持当前写法，基线本身不是最优值，对比结论可能低估大模型提升空间。 |

---

## 6. 若要得到最佳效果需要改什么

> 以下仅列方向，本次不执行任何修改。

- **Embedding**：
  - query 加 `Instruct: {task}\nQuery:{text}`，document 不加；
  - 改用 last-token pooling；
  - L2 归一化；
  - tokenizer 设 `padding_side='left'`。

- **Reranker**：
  - 严格使用官方 chat template（system + user + assistant suffix）；
  - 输入格式为 `<Instruct>:\n<Query>:\n<Document>:`；
  - 得分用 yes/no 上的 softmax 概率。

- **数据集**：
  - 若要与官方报告对比，应使用 MTEB / C-MTEB / BEIR 等标准 benchmark，而不是自制小样本。

---

## 参考来源

- [QwenLM/Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding)
- [Qwen3 Embedding 论文 arXiv:2506.05176](https://arxiv.org/abs/2506.05176)
- [Hugging Face Model Card: Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
