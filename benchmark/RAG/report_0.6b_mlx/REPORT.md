# 0.6B Embedding+Reranker 本地服务 × benchmark/RAG (SyllabusQA) 端到端评测报告

> 生成时间：2026-06-28
> 被测组合（推荐配置）：**Qwen3-Embedding-0.6B-4bit-DWQ + Qwen3-Reranker-0.6B-4bit (cross-encoder, 已修复)**
> 评测套件：`benchmark/RAG`（真实 RAG 流水线：ingest→embed→retrieve→rerank→generate→LLM-judge）
> 数据集：SyllabusQA 小样本（5 份 syllabus 文档 / 20 道评测问题）

---

## 0. 结论

把两个 0.6B 模型做成本地 OpenAI 兼容服务，接入 OpenViking 跑通了完整 RAG 评测：

| 指标 | 值 | 含义 |
|------|----|----|
| **Average Recall** | **0.613** | 检索（embedding+rerank）把含答案证据召回的比例 ← **最能反映被测两模型** |
| **Average Accuracy（归一化）** | **0.738** | LLM 评判答案正确性（0–4 → 0–1） |
| Average Accuracy（0–4） | 2.95 / 4 | 同上原始分 |
| Average F1 | 0.332 | 答案词重叠（对 SyllabusQA 这类长答案天然偏低） |
| Avg 检索延迟 | 14.4 s/query | 本机串行（含 0.6B embed + rerank + 层级检索）|

### 0.1 对照实验：reranker 的端到端增益（关 vs 开）

同一份 SyllabusQA（20 题、topk=5、其余配置完全相同），仅切换 rerank：

| 指标 | rerank **关**（仅 embedding 召回） | rerank **开**（+0.6B cross-encoder） | 增益 |
|------|:---:|:---:|:---:|
| **Average Recall** | 0.489 | **0.613** | **+25.4%** |
| **Accuracy（归一化）** | 0.625 | **0.738** | **+18.1%** |
| Accuracy（0–4） | 2.50 | 2.95 | +0.45 |
| F1 | 0.305 | 0.332 | +9% |

**结论：修复后的 0.6B cross-encoder reranker 在真实 RAG 里带来确凿增益**——Recall +25%、答案正确率 +18%。这与单元/MTEB 层面的结论一致（SciFact e2e nDCG +6.6%），证明这套小模型组合"召回 + 精排"的两段式是有效的。原始数据见 `metrics_norerank.json`（关）与 `metrics.json`（开）。


- **Recall 0.61 / Accuracy 0.74**：0.6B 这套小模型在真实教育 RAG 里是**可用**的——大部分问题能召回正确证据并答对。
- F1 偏低是 SyllabusQA 答案较长、措辞自由导致的**指标特性**，不代表答错（Accuracy 由 LLM 语义评判更可信）。
- 生成/评判用的是 Kimi（kimi-2.6），**F1/Accuracy 同时反映 Kimi 的生成质量**；Recall 才是这两个被测模型的纯净信号。

---

## 1. 架构（实际跑通的形态）

```
┌─ MLX 服务 serve_mlx_openai.py (:11455, ~mlx-env) ─┐
│  POST /v1/embeddings → Qwen3-Embedding-0.6B-4bit  │ ← OpenViking add_resource / find
│  POST /v1/rerank     → Qwen3-Reranker-0.6B-4bit   │   (本次共服务 embed 116 / rerank 296 次)
└───────────────────────────────────────────────────┘
benchmark/RAG/ov.conf  → embedding(dim=1024)+rerank 指向 :11455; vlm=Kimi
benchmark/RAG run.py (.venv) → SyllabusQA: ingest→retrieve(+rerank)→generate(Kimi)→judge(Kimi)
```

- 服务进程：`benchmark/embedding_reranker/scripts/serve_mlx_openai.py`，复用已修复的 `embed_texts` / `rerank_scores`，query/document 用 `input_type` 区分前缀。
- 配置：`benchmark/RAG/ov.conf`（0.6B，dim 1024，独立 workspace）+ `config/syllabusqa_0.6b.yaml`（Kimi gen/judge）。**均含密钥，已 gitignore**。

---

## 2. 效率数据（本次运行）

| 项 | 值 |
|----|----|
| 入库总时长 | 298.7 s（5 文档 / 31 chunk） |
| 入库 embedding tokens | 42,650 |
| 平均检索延迟 | 14.4 s/query |
| 平均输入/输出 tokens | 2865 / 83 per query |
| 评测问题数 | 20 |

> 本机为保证 embedded 向量库单handle，**串行执行（max_workers=1）**，故延迟偏高；生产应走服务化 OpenViking + 并发。

---

## 3. 踩坑与修复（复现关键）

| # | 问题 | 修复 |
|---|------|------|
| 1 | RAG 运行时报 `openviking` ImportError | 用 **repo `.venv`** 跑（openviking 装在那；mlx-env 只跑 MLX 服务） |
| 2 | `LocalClient got unexpected kwarg 'agent_id'` | **OpenViking core 1 行 bug**：`async_client.py` 把 legacy 别名 `agent_id` 透传给不接受它的 `LocalClient` → 改为解析成 `actor_peer_id` |
| 3 | `query_param/document_param` 校验失败 | 该字段是**字符串**（如 `"query"`），不是 dict |
| 4 | Kimi `404 resource_not_found` | gen/judge 的 `base_url` 要带 `/v1`：`https://api.kimi.com/coding/v1` |
| 5 | `vectordb LOCK already held by process` | embedded 向量库不支持并发开库 → `max_workers: 1` 串行 |
| 6 | Kimi `400 invalid temperature` | kimi-2.6 只允许 `temperature: 1` |

> 另：reranker 本身的 cross-encoder bug（后缀截断 / 末 token gather）已在前序工作修复，本次服务直接复用修复版（冒烟：相关文档 top-1，无关 0.00001）。

---

## 4. 文件

| 文件 | 说明 |
|------|------|
| `benchmark/embedding_reranker/scripts/serve_mlx_openai.py` | MLX OpenAI 兼容服务（embed+rerank） |
| `benchmark/RAG/ov.conf` | 指向 0.6B 服务的 OpenViking 配置（gitignored） |
| `benchmark/RAG/config/syllabusqa_0.6b.yaml` | Kimi gen/judge 配置（gitignored） |
| `benchmark/RAG/report_0.6b_mlx/metrics.json` | 本次完整指标原始 JSON |
| `benchmark/RAG/Output/SyllabusQA/experiment_0.6b_top_5/` | benchmark.log + metrics |

---

## 5. 如何复现

```bash
# 1) 起 0.6B 服务（mlx 环境）
source ~/mlx-env/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python benchmark/embedding_reranker/scripts/serve_mlx_openai.py --port 11455 --max-length 512 &

# 2) 跑 RAG（OpenViking 的 .venv 环境）
cd benchmark/RAG
/Users/zhengxiaoxi/repo/OpenViking/.venv/bin/python run.py \
  --config config/syllabusqa_0.6b.yaml --step all
```

---

## 6. 后续可选

- **对照实验**：关掉 rerank（ov.conf 去掉 rerank 段）再跑一遍，量化 0.6B reranker 在真实 RAG 的端到端增益。
- **与 8B 对照**：你现有 :11436 的 8B 服务跑同一份 SyllabusQA，对比 Recall/Accuracy 与延迟，评估 0.6B 的性价比。
- **扩样本/换数据集**：增大 max_queries 或换 Qasper（长文档，更吃 reranker）。
