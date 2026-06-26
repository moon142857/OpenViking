# OpenViking 0.4.5 本地 MLX 部署与测试完整指南

> 本指南记录如何在 macOS Apple Silicon 上从零部署 OpenViking 0.4.5，配合本地 MLX 运行的 `Qwen3-Embedding-0.6B-4bit-DWQ` 和 `Qwen3-Reranker-0.6B-4bit`，并完成服务接口测试。  
> 目标：另一台机器按此文档可完整复刻。

---

## 目录

1. [环境准备](#1-环境准备)
2. [系统依赖安装](#2-系统依赖安装)
3. [OpenViking 源码与编译](#3-openviking-源码与编译)
4. [MLX 嵌入与重排序服务部署](#4-mlx-嵌入与重排序服务部署)
5. [OpenViking 配置](#5-openviking-配置)
6. [启动服务](#6-启动服务)
7. [服务接口测试](#7-服务接口测试)
8. [常见问题与排查](#8-常见问题与排查)
9. [附录：一键脚本](#9-附录一键脚本)
10. [Bot 模式](#10-bot-模式)

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 系统 | macOS 14+ | macOS 15+ |
| 芯片 | Apple Silicon (M1+) | M2 Pro/Max 或更高 |
| 内存 | 16 GB | 24 GB 或更高 |
| 磁盘 | 20 GB 可用空间 | 50 GB+ SSD |

> 0.6B 4-bit 量化模型在 Apple Silicon 上大约占用 1–2 GB 统一内存。若内存只有 8 GB，建议关闭其他大型应用。

### 1.2 网络要求

- 可访问 GitHub/HuggingFace（下载模型和源码）
- 若使用 Kimi VLM，需要能访问 `https://api.kimi.com/coding`

### 1.3 需要的账号/Key

- Kimi API Key（用于 VLM 摘要生成）：`sk-kimi-...`
- 可选：HuggingFace Token（部分模型需要）

---

## 2. 系统依赖安装

### 2.1 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2.2 安装基础工具

```bash
brew install git cmake llvm rust node uv curl
```

### 2.3 验证版本

```bash
git --version
cmake --version
clang --version
rustc --version
node --version
npm --version
uv --version
```

参考版本（2026-06-26 验证通过）：

```
Rust 1.94.0
CMake 4.2.3
Clang 21.0.0
Node.js 25.8.1 / npm 11.11.0
uv 0.10.8
```

### 2.4 配置 Rust 工具链

```bash
rustup default stable
rustup target add aarch64-apple-darwin
```

---

## 3. OpenViking 源码与编译

### 3.1 克隆源码

```bash
cd ~/repo  # 或你喜欢的目录
git clone https://github.com/volcengine/OpenViking.git
cd OpenViking
```

### 3.2 设置 Python 虚拟环境

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
```

### 3.3 处理版本号问题

OpenViking 当前 git tag 可能不连续，直接 `uv sync` 会触发 `setuptools_scm` 断言失败。需要显式指定版本：

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.4.5
```

### 3.4 安装依赖并编译

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.4.5
uv sync --all-extras
```

此过程会：
- 安装 Python 依赖
- 编译 Rust 扩展（`ragfs`、`ov_cli` 等）
- 构建 native C++ 扩展

耗时约 5–20 分钟，取决于网络和机器性能。

### 3.5 验证安装

```bash
openviking-server --help
ov --help
```

如果看到命令帮助，说明编译成功。

---

## 4. MLX 嵌入与重排序服务部署

`mlx_lm.server` 默认不支持 embedding 和 rerank 接口，因此需要一个自定义 MLX server。下面给出完整脚本。

### 4.1 创建 MLX 环境

```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
pip install --upgrade pip
pip install mlx mlx-lm fastapi uvicorn transformers numpy
```

> 若系统没有 `python3.13`，可直接用 `python3`（当前环境为 Python 3.14.6，MLX 同样可用）。

### 4.2 下载模型

推荐先配置 HF 镜像（国内）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载 0.6B MLX 量化模型：

```bash
source ~/mlx-env/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

hf download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
hf download mlx-community/Qwen3-Reranker-0.6B-4bit
```

模型会缓存到 `~/.cache/huggingface/hub/`。

### 4.3 创建 MLX Server 脚本

创建文件 `~/repo/qwen3_embedding/qwen3_mlx_server.py`：

```python
#!/usr/bin/env python3
"""MLX-based Qwen3 embedding + rerank server for OpenViking."""

import argparse
import logging
from typing import List, Union

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlx_lm import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3 MLX Embedding/Rerank Server")

DEFAULT_EMBEDDING_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
DEFAULT_RERANK_MODEL = "mlx-community/Qwen3-Reranker-0.6B-4bit"
DEFAULT_DIMENSION = 1024


def load_models(embedding_model: str, rerank_model: str):
    logger.info(f"Loading embedding model: {embedding_model}")
    embed_m, embed_t = load(embedding_model)
    logger.info(f"Loading rerank model: {rerank_model}")
    rerank_m, rerank_t = load(rerank_model)
    return (embed_m, embed_t), (rerank_m, rerank_t)


def embed_texts(texts: Union[str, List[str]], model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer._tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=8192,
    )
    input_ids = mx.array(inputs["input_ids"].astype(np.int32))
    attention_mask = mx.array(inputs["attention_mask"].astype(np.float32))
    # Use the underlying transformer to get hidden states.
    outputs = model.model(input_ids)
    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
    mask = attention_mask.astype(hidden.dtype)
    sum_embeddings = (hidden * mask[:, :, None]).sum(axis=1)
    embeddings = sum_embeddings / mx.maximum(mask.sum(axis=1), 1e-9)
    return np.array(embeddings.astype(mx.float32))


def rerank_scores(query: str, documents: List[str], model, tokenizer):
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer._tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=512,
    )
    input_ids = mx.array(inputs["input_ids"].astype(np.int32))
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    # Qwen3-Reranker: use the "yes" token logit at the last position.
    yes_id = tokenizer._tokenizer.convert_tokens_to_ids("yes")
    last_logits = logits[:, -1, :]
    yes_scores = last_logits[:, yes_id]
    return np.array(yes_scores.astype(mx.float32)).tolist()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    texts = body.get("input", [])
    model_id = body.get("model", DEFAULT_EMBEDDING_MODEL)
    embeddings = embed_texts(texts, embed_model, embed_tokenizer)
    data = [
        {"object": "embedding", "index": i, "embedding": emb.tolist()}
        for i, emb in enumerate(embeddings)
    ]
    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


@app.post("/v1/rerank")
async def rerank(req: Request):
    body = await req.json()
    query = body.get("query", "")
    documents = body.get("documents", [])
    model_id = body.get("model", DEFAULT_RERANK_MODEL)
    scores = rerank_scores(query, documents, rerank_model, rerank_tokenizer)
    results = [
        {"index": i, "relevance_score": score, "document": doc}
        for i, (score, doc) in enumerate(zip(scores, documents))
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return JSONResponse({
        "object": "list",
        "model": model_id,
        "results": results,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    args = parser.parse_args()

    (embed_model, embed_tokenizer), (rerank_model, rerank_tokenizer) = load_models(
        args.embedding_model, args.rerank_model
    )
    uvicorn.run(app, host=args.host, port=args.port)
```

> 注意：上述脚本是简化实现，用于验证部署。生产环境建议参考官方实现或社区优化版本，确保 pooling 方式和分数计算与训练目标一致。

### 4.4 启动 MLX Server

```bash
source ~/mlx-env/bin/activate
mkdir -p ~/repo/qwen3_embedding
python ~/repo/qwen3_embedding/qwen3_mlx_server.py \
  --host 127.0.0.1 \
  --port 11436
```

首次启动会自动检查/下载模型，耗时几分钟（取决于网络）。

验证是否启动成功：

```bash
curl http://127.0.0.1:11436/health
```

测试 embedding：

```bash
curl -X POST http://127.0.0.1:11436/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"}'
```

测试 rerank：

```bash
curl -X POST http://127.0.0.1:11436/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OpenViking",
    "documents": ["OpenViking is a context database.", "Unrelated text."],
    "model": "mlx-community/Qwen3-Reranker-0.6B-4bit"
  }'
```

---

## 5. OpenViking 配置

### 5.1 创建配置目录

```bash
mkdir -p ~/.openviking
```

### 5.2 编写 ov.conf

创建 `~/.openviking/ov.conf`：

```json
{
  "storage": {
    "workspace": "/Users/YOUR_USERNAME/.openviking/data-mlx"
  },
  "embedding": {
    "dense": {
      "provider": "openai",
      "api_base": "http://localhost:11436/v1",
      "api_key": "dummy",
      "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
      "dimension": 1024,
      "encoding_format": "float",
      "batch_size": 32,
      "input": "text"
    },
    "text_source": "content_only",
    "max_input_tokens": 4096,
    "allow_metadata_override": true
  },
  "vlm": {
    "provider": "kimi",
    "api_base": "https://api.kimi.com/coding",
    "api_key": "sk-kimi-YOUR_KEY_HERE",
    "model": "kimi-code",
    "temperature": 1.0
  },
  "rerank": {
    "provider": "openai",
    "api_base": "http://localhost:11436/v1/rerank",
    "api_key": "dummy",
    "model": "mlx-community/Qwen3-Reranker-0.6B-4bit",
    "timeout": 120,
    "threshold": 0.1
  },
  "server": {
    "host": "0.0.0.0",
    "port": 1933,
    "auth_mode": "dev",
    "cors_origins": ["*"]
  }
}
```

> 把 `YOUR_USERNAME` 和 `YOUR_KEY_HERE` 替换成实际值。`temperature` 固定为 1.0，避免部分 VLM 报错。`auth_mode: dev` 用于本地开发，免去每次传 API Key；生产环境建议改为 `api_key` 并设置 `root_api_key`。

### 5.3 配置文件说明

| 字段 | 说明 |
|------|------|
| `storage.workspace` | OpenViking 数据目录，首次启动会自动初始化 |
| `embedding.dense` | 指向本地 MLX embedding server |
| `embedding.allow_metadata_override` | 切换 embedding 模型时允许复用已有向量库 |
| `vlm` | 用于生成 L0/L1 摘要，当前使用 Kimi |
| `rerank` | 指向本地 MLX rerank server，`threshold` 控制过滤阈值 |
| `server.auth_mode` | `dev` 为本地免密模式，`api_key` 为正式模式 |

---

## 6. 启动服务

### 6.1 启动 MLX Server（终端 1）

```bash
source ~/mlx-env/bin/activate
python ~/repo/qwen3_embedding/qwen3_mlx_server.py \
  --host 127.0.0.1 \
  --port 11436
```

### 6.2 启动 OpenViking Server（终端 2）

```bash
cd ~/repo/OpenViking
source .venv/bin/activate
openviking-server --host 127.0.0.1 --port 1933
```

首次启动会初始化 workspace，可能需要几十秒。

### 6.3 验证服务

```bash
# OpenViking 健康检查
curl http://127.0.0.1:1933/health

# 就绪检查
curl http://127.0.0.1:1933/ready
```

`/ready` 应返回：

```json
{
  "status": "ready",
  "checks": {
    "agfs": { "status": "ok" },
    "vectordb": "ok",
    "embedding": "ok",
    "ollama": "not_configured"
  }
}
```

---

## 7. 服务接口测试

### 7.1 导入测试资源

```bash
curl -X POST http://127.0.0.1:1933/api/v1/resources \
  -H "Content-Type: application/json" \
  -d '{
    "path": "https://raw.githubusercontent.com/volcengine/OpenViking/main/README.md",
    "to": "viking://resources/test-readme",
    "wait": false
  }'
```

返回 `task_id`，说明异步导入已开始。

等待 5–30 秒让后台队列处理，然后检查队列：

```bash
curl http://127.0.0.1:1933/api/v1/observer/queue
```

### 7.2 文件系统接口测试

```bash
# 列目录
curl "http://127.0.0.1:1933/api/v1/fs/ls?uri=viking://resources/"

# 递归目录树
curl "http://127.0.0.1:1933/api/v1/fs/tree?uri=viking://resources/test-readme&depth=2"

# 文件元信息
curl "http://127.0.0.1:1933/api/v1/fs/stat?uri=viking://resources/test-readme"

# 读取内容
curl "http://127.0.0.1:1933/api/v1/content/read?uri=viking://resources/test-readme/Overview.md"
```

### 7.3 检索接口测试

```bash
# 语义检索（需等待 L0/L1 摘要和向量生成完成）
curl -X POST http://127.0.0.1:1933/api/v1/search/find \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is OpenViking",
    "target_uri": "viking://resources/test-readme",
    "limit": 5
  }'

# 文本 grep
curl -X POST http://127.0.0.1:1933/api/v1/search/grep \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": "OpenViking",
    "uri": "viking://resources/test-readme",
    "limit": 5
  }'

# Glob 路径匹配
curl -X POST http://127.0.0.1:1933/api/v1/search/glob \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": "**/*.md",
    "target_uri": "viking://resources/test-readme"
  }'
```

### 7.4 Observer 可观测性测试

```bash
curl http://127.0.0.1:1933/api/v1/observer/system
curl http://127.0.0.1:1933/api/v1/observer/queue
curl http://127.0.0.1:1933/api/v1/observer/models
curl http://127.0.0.1:1933/api/v1/observer/retrieval
```

### 7.5 会话接口测试

```bash
# 创建会话
curl -X POST http://127.0.0.1:1933/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-001"}'

# 追加消息
curl -X POST http://127.0.0.1:1933/api/v1/sessions/test-session-001/messages \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hello OpenViking"}'

# 获取会话上下文
curl "http://127.0.0.1:1933/api/v1/sessions/test-session-001/context"
```

### 7.6 自动化测试脚本

项目已提供一键测试脚本：

```bash
cd ~/repo/OpenViking
source .venv/bin/activate
python scripts/api_functional_perf_test.py
```

该脚本会：
- 测试健康检查、文件系统、内容读取、检索、Observer、控制台等接口
- 输出功能测试结果与性能统计（P50/P95/平均耗时）

---

## 8. 常见问题与排查

### 8.1 `uv sync` 报错 `AssertionError` 或版本号错误

**原因**：`setuptools_scm` 无法从 git tag 推断版本。  
**解决**：

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.4.5
uv sync --all-extras
```

### 8.2 MLX server 首次请求很慢（3–8 秒）

**原因**：MLX 首次运行需要编译 kernel 并加载模型。  
**解决**：启动后发送 1–2 个预热请求，或保持服务常驻。

```bash
curl -X POST http://127.0.0.1:11436/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "warmup", "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"}'
```

### 8.3 `/search/find` 返回空结果

**原因**：资源刚导入，L0/L1 摘要和语义向量还在后台队列处理。  
**解决**：

```bash
curl http://127.0.0.1:1933/api/v1/observer/queue
```

等待 `Embedding` 和 `Semantic-Nodes` pending 为 0 后再试。

### 8.4 `memory_pressure` 显示非 0

**原因**：统一内存不足，系统开始使用 swap。  
**解决**：
- 关闭不必要的应用
- 0.6B 模型已是最小推荐，若仍不足可关闭其他后台服务
- 升级内存更大的机器

### 8.5 OpenViking Server 启动失败：端口被占用

```bash
lsof -i :1933
kill -9 <PID>
```

### 8.6 `/ready` 显示 `embedding: error`

**原因**：OpenViking 无法连接 MLX server。  
**解决**：
- 检查 MLX server 是否启动：`curl http://127.0.0.1:11436/health`
- 检查 `ov.conf` 中 `api_base` 是否为 `http://localhost:11436/v1`
- 检查防火墙

### 8.7 VLM 摘要生成失败

**原因**：Kimi API key 无效或 `temperature` 不支持。  
**解决**：
- 确认 key 有效
- 在 `ov.conf` 中设置 `"temperature": 1.0`

### 8.8 切换 embedding 模型后报错 `EmbeddingRebuildRequiredError`

**原因**：已有向量库是用旧模型构建的，与新模型元数据不一致。  
**解决**：
- 在 `ov.conf` 的 `embedding` 段增加 `"allow_metadata_override": true`
- 或删除 `storage.workspace` 目录重新初始化

---

## 9. 附录：一键脚本

> 普通模式脚本（不含 Bot）：`start_mlx_server.sh`、`start_openviking.sh`  
> Bot 模式脚本：`start_openviking_with_bot.sh`

### 9.1 启动 MLX Server（后台）

创建 `~/repo/OpenViking/scripts/start_mlx_server.sh`：

```bash
#!/bin/bash
set -e

PORT=11436
LOG_FILE="/tmp/qwen3_mlx_server.log"

# 检查端口是否已被占用
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. MLX server may already be running."
    echo "Check: lsof -i :$PORT"
    exit 1
fi

source ~/mlx-env/bin/activate

# 国内用户建议开启 HF 镜像
export HF_ENDPOINT="https://hf-mirror.com"

echo "Starting Qwen3 MLX server on port $PORT..."
echo "Log: $LOG_FILE"

nohup python "${HOME}/repo/qwen3_embedding/qwen3_mlx_server.py" \
    --host 127.0.0.1 --port "$PORT" \
    > "$LOG_FILE" 2>&1 &

# 等待服务就绪（模型加载可能耗时数秒）
for i in {1..30}; do
    if curl -s http://127.0.0.1:"$PORT"/health > /dev/null 2>&1; then
        echo "MLX server started successfully."
        exit 0
    fi
    sleep 1
done

echo "Failed to start MLX server. Check log: $LOG_FILE"
exit 1
```

### 9.2 启动 OpenViking Server（后台）

创建 `~/repo/OpenViking/scripts/start_openviking.sh`：

```bash
#!/bin/bash
set -e

cd ~/repo/OpenViking
source .venv/bin/activate

# Anthropic-compatible LLM endpoint (Kimi coding API)
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"
export ANTHROPIC_API_KEY="sk-kimi-YOUR_KEY_HERE"

LOG_FILE="/tmp/openviking-server.log"
PORT=1933

if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. OpenViking server may already be running."
    echo "Check: lsof -i :$PORT"
    exit 1
fi

echo "Starting OpenViking server on port $PORT..."
echo "Log: $LOG_FILE"

nohup openviking-server --host 127.0.0.1 --port "$PORT" \
    > "$LOG_FILE" 2>&1 &

sleep 3
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "OpenViking server started successfully."
    echo "API:    http://127.0.0.1:$PORT"
    echo "Studio: http://127.0.0.1:$PORT/studio"
else
    echo "Failed to start OpenViking server. Check log: $LOG_FILE"
    exit 1
fi
```

> 注意：把 `ANTHROPIC_API_KEY` 中的 `sk-kimi-YOUR_KEY_HERE` 替换为你的真实 key。当前 `ov.conf` 仍使用 `provider: kimi`，这两个环境变量供你后续切到 `provider: anthropic` 时使用。

### 9.3 启动 OpenViking Server with Bot（后台）

创建 `~/repo/OpenViking/scripts/start_openviking_with_bot.sh`：

```bash
#!/bin/bash
# 启动 OpenViking HTTP Server（启用 Bot 模式，自动拉起 Vikingbot gateway）

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Anthropic-compatible LLM endpoint (Kimi coding API)
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"
export ANTHROPIC_API_KEY="sk-kimi-YOUR_KEY_HERE"

LOG_FILE="/tmp/openviking-server.log"
PORT=1933

# 检查端口是否已被占用
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. OpenViking server may already be running."
    echo "Check: lsof -i :$PORT"
    exit 1
fi

echo "Starting OpenViking server with bot on port $PORT..."
echo "Log: $LOG_FILE"

nohup openviking-server --host 127.0.0.1 --port "$PORT" --with-bot \
    > "$LOG_FILE" 2>&1 &

sleep 3
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "OpenViking server with bot started successfully."
    echo "API:      http://127.0.0.1:$PORT"
    echo "Studio:   http://127.0.0.1:$PORT/studio"
    echo "Bot API:  http://127.0.0.1:$PORT/bot/v1"
else
    echo "Failed to start OpenViking server. Check log: $LOG_FILE"
    exit 1
fi
```

### 9.4 停止所有服务

创建 `~/repo/OpenViking/scripts/stop_all.sh`：

```bash
#!/bin/bash
# 停止 OpenViking、Vikingbot gateway 和 Qwen3 MLX 服务

set -e

echo "Stopping OpenViking server (port 1933)..."
pkill -f "openviking-server" 2>/dev/null || true

echo "Stopping Vikingbot gateway (port 18790)..."
pkill -f "vikingbot gateway" 2>/dev/null || true

echo "Stopping Qwen3 MLX server (port 11436)..."
pkill -f "qwen3_mlx_server.py" 2>/dev/null || true

sleep 1

for port in 1933 18790 11436; do
    if lsof -i :"$port" > /dev/null 2>&1; then
        echo "Warning: port $port still in use"
    else
        echo "Port $port is free."
    fi
done

echo "All services stopped."
```

### 9.5 使用方式

```bash
chmod +x ~/repo/OpenViking/scripts/*.sh

# 普通模式启动
~/repo/OpenViking/scripts/start_mlx_server.sh
~/repo/OpenViking/scripts/start_openviking.sh

# Bot 模式启动（会自动拉起 Vikingbot gateway）
~/repo/OpenViking/scripts/start_mlx_server.sh
~/repo/OpenViking/scripts/start_openviking_with_bot.sh

# 停止
~/repo/OpenViking/scripts/stop_all.sh
```

---

## 10. Bot 模式

OpenViking 支持集成 Vikingbot，提供 `/bot/v1/chat`、`/bot/v1/chat/stream` 等 Agent 对话接口。

### 10.1 启动 Bot 模式

Bot 模式需要同时运行三个服务：

| 服务 | 启动命令 | 端口 |
|------|---------|------|
| MLX Server | `scripts/start_mlx_server.sh` | 11436 |
| OpenViking + Vikingbot | `scripts/start_openviking_with_bot.sh` | 1933 / 18790 |

或者手动启动：

```bash
# 1. 启动 MLX server（另开终端）
source ~/mlx-env/bin/activate
python ~/repo/qwen3_embedding/qwen3_mlx_server.py \
  --host 127.0.0.1 --port 11436

# 2. 启动 OpenViking with bot（另开终端）
cd ~/repo/OpenViking
source .venv/bin/activate
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"
export ANTHROPIC_API_KEY="sk-kimi-YOUR_KEY_HERE"
openviking-server --host 127.0.0.1 --port 1933 --with-bot
```

> 注意：必须使用 `--with-bot`，而不是 `--bot`。
> - `--with-bot`：启用 Bot API proxy，并自动启动 Vikingbot gateway。
> - `--bot`：只尝试启动 Vikingbot gateway，不会启用 API proxy。

### 10.2 Bot 接口

Bot API 路径前缀为 **`/bot/v1`**，不是 `/api/v1/bot`。

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/bot/v1/health` | Bot 服务健康 |
| 非流式对话 | POST | `/bot/v1/chat` | 单次 Agent 对话 |
| 流式对话 | POST | `/bot/v1/chat/stream` | SSE 流式输出 |
| 反馈 | POST | `/bot/v1/feedback` | 提交对话反馈 |

示例：

```bash
curl http://127.0.0.1:1933/bot/v1/health
# → {"status":"healthy","version":"0.4.5",...}

curl -X POST http://127.0.0.1:1933/bot/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello OpenViking"}'
```

### 10.3 Bot 模式常见问题

#### 报错：`Bot service not enabled. Start server with --with-bot option.`

**原因**：使用了 `--bot` 而不是 `--with-bot`，或 Bot API proxy 未启用。  
**解决**：使用 `openviking-server --with-bot` 启动。

#### 报错：`vikingbot gateway port 18790 is already in use`

**原因**：手动提前启动了 `vikingbot gateway`，而 `--with-bot` 会自己再启动一个。  
**解决**：先停止手动启动的 Vikingbot，再使用 `--with-bot`：

```bash
pkill -f "vikingbot gateway"
openviking-server --host 127.0.0.1 --port 1933 --with-bot
```

#### `/api/v1/bot/health` 返回 404

**原因**：Bot API 挂载路径是 `/bot/v1`，不是 `/api/v1/bot`。  
**解决**：访问 `http://127.0.0.1:1933/bot/v1/health`。

---

## 11. 总结

按本文档操作即可完成：

1. macOS Apple Silicon 环境准备
2. OpenViking 0.4.5 源码编译
3. 本地 MLX Qwen3 Embedding/Rerank 服务部署
4. OpenViking 配置与启动（普通模式 / Bot 模式）
5. 核心接口功能验证与性能测试

如在另一台机器复刻时遇到问题，优先检查：
- `memory_pressure`
- MLX server 是否成功启动并响应 embedding 请求
- `/ready` 中 `embedding` 状态是否为 `ok`
- `/observer/queue` 是否有积压
- Bot 模式下 Vikingbot gateway 是否在 18790 端口运行
