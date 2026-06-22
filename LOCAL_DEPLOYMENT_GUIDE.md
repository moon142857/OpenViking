# OpenViking 本地部署与运行指南

> 本指南基于 macOS + Apple Silicon 环境整理，覆盖项目依赖、模型部署、源码编译、服务启动和 API 测试。
>
> 验证时间：2026-06-22  
> 验证代码版本：OpenViking 0.4.4 + 4 commits（`cab0d525`）  
> 验证机器：`/Users/zhengxiaoxi`

---

## 一、环境概述

| 组件 | 用途 | 本机状态 |
|------|------|----------|
| Python 3.10+ | OpenViking 运行时 | ✅ 3.13.12（uv venv） |
| Rust 1.91.1+ | 编译 `ov` CLI 与 RAGFS Rust binding | ✅ 1.94.0 |
| CMake 3.15+ | 编译 C++ 向量引擎扩展 | ✅ 4.2.3 |
| Clang 11+ / GCC 9+ | C++17 编译器 | ✅ Clang 21.0.0 |
| Node.js + npm | 构建 Web Studio | ✅ 25.8.1 / 11.11.0 |
| uv | Python 依赖与环境管理 | ✅ 0.10.8 |
| Ollama | 备用本地 embedding（可选） | ✅ 已安装 |
| MLX 环境 | 运行 Qwen3 Embedding + Reranker | ✅ `~/mlx-env` |
| Kimi API Key | VLM（摘要、overview） | ✅ 已配置到 `~/.openviking/ov.conf` |

> **注意**：`CONTRIBUTING.md` 中提到的 Go 依赖已过时，当前 AGFS 实现已改为 Rust，无需安装 Go。

---

## 二、项目依赖

### 2.1 系统级依赖

| 依赖 | 最低版本 | 安装方式（macOS） |
|------|----------|-------------------|
| Python | 3.10 | 系统自带或 `brew install python` |
| Rust | 1.91.1 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| CMake | 3.15 | `brew install cmake` |
| Xcode CLI Tools | - | `xcode-select --install` |
| Node.js | 18+ | `brew install node` |
| uv | 最新 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

检查命令：

```bash
python3 --version
rustc --version
cargo --version
cmake --version
clang++ --version
node --version
npm --version
uv --version
```

### 2.2 Python 依赖

核心依赖见 `pyproject.toml`，包括：

- `fastapi`、`uvicorn`：HTTP 服务
- `openai`、`litellm`：多模型后端
- `volcengine`、`volcengine-python-sdk[ark]`：火山引擎/豆包
- `pydantic`、`httpx`、`pyyaml`：基础工具
- `pdfplumber`、`python-docx`、`openpyxl`、`ebooklib`：文档解析
- `tree-sitter` 系列：代码解析
- `opentelemetry-*`：可观测性
- `mcp`：MCP 协议支持

可选 extras：

```bash
uv sync --all-extras    # 包含 bot、langchain、gemini、ocr 等全部可选依赖
```

### 2.3 Rust 依赖

Rust CLI（`crates/ov_cli`）和 RAGFS Python binding（`crates/ragfs-python`）依赖 Cargo 自动下载，主要包括：

- `clap`、`reqwest`、`tokio`、`serde`：CLI 与 HTTP
- `pyo3`：Python 绑定
- `ratatui`、`crossterm`：TUI 交互

### 2.4 模型依赖

OpenViking 运行需要两类模型能力：

| 能力 | 本机部署方式 | 端口/路径 |
|------|-------------|-----------|
| Embedding | MLX 本地模型 `Qwen3-Embedding-8B-4bit-DWQ` | `http://localhost:11436/v1` |
| Rerank | MLX 本地模型 `Qwen3-Reranker-8B-MLX-4bit` | `http://localhost:11436/v1/rerank` |
| VLM | Kimi API（`kimi-2.6`） | `https://api.kimi.com/coding` |

---

## 三、模型部署

### 3.1 启动 MLX Embedding + Rerank 服务

你本地已有自定义 MLX 服务脚本，同时加载两个模型：

```bash
source ~/mlx-env/bin/activate
python /Users/zhengxiaoxi/repo/qwen3_embedding/qwen3_mlx_server.py
```

该脚本会：
- 加载 `~/.cache/huggingface/hub/Qwen3-Embedding-8B-4bit-DWQ`
- 加载 `~/.cache/huggingface/hub/Qwen3-Reranker-8B-MLX-4bit`
- 在 `0.0.0.0:11436` 暴露两个端点：
  - `POST /v1/embeddings`
  - `POST /v1/rerank`

**长期运行建议：**

```bash
source ~/mlx-env/bin/activate
nohup python /Users/zhengxiaoxi/repo/qwen3_embedding/qwen3_mlx_server.py \
  > /tmp/qwen3_mlx_server.log 2>&1 &
```

**验证：**

```bash
# Embedding
curl http://127.0.0.1:11436/v1/embeddings -X POST \
  -H "Content-Type: application/json" \
  -d '{"input": "hello"}'

# Rerank
curl http://127.0.0.1:11436/v1/rerank -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "what is OpenViking", "documents": ["OpenViking is a context database", "unrelated"]}'
```

### 3.2 准备 Kimi API Key

从 `~/.zshrc` 中获取：

```bash
grep ANTHROPIC_API_KEY ~/.zshrc
```

> 你的 Claude Code 配置复用了 Kimi endpoint，key 以 `ANTHROPIC_API_KEY` 命名但对应 Kimi 服务。

将该 key 填入 `~/.openviking/ov.conf` 的 `vlm.api_key`。

---

## 四、源码编译

### 4.1 克隆/进入仓库

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
```

### 4.2 解决 Git 标签版本解析问题

当前仓库 latest reachable tag 为 `python-sdk/v0.1.2`，不符合 `setuptools_scm` 的 `^vX.Y.Z` 正则，需要手动指定版本：

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.4.4
```

### 4.3 安装 Python 依赖并构建本地包

```bash
source .venv/bin/activate
uv sync --all-extras
```

`uv sync --all-extras` 会触发 `setup.py`，自动完成：
- 构建 `ov` Rust CLI → `openviking/bin/ov`
- 构建 RAGFS Python binding → `openviking/lib/ragfs_python.abi3.so`
- 构建 C++ 向量引擎扩展
- 构建并打包 Web Studio → `openviking/web_studio/dist/`

### 4.4 验证编译结果

```bash
openviking-server --version   # openviking-server 0.4.4
ov language zh-CN             # 首次使用需设置语言
ov --version                  # openviking 0.4.4
python -c "import openviking; print(openviking.__version__)"
```

### 4.5 Makefile 方式（可选）

```bash
make check-deps    # 检查系统依赖
make build         # 完整构建（含 Web Studio）
make build-cli     # 仅构建 Rust CLI
make build-studio  # 仅构建 Web Studio
make clean         # 清理构建产物
```

> 如果不需要 Web Studio，可设置 `OV_SKIP_STUDIO_BUILD=1` 跳过前端构建。

---

## 五、部署方式

### 5.1 本地独立服务（本机验证方式）

启动 OpenViking HTTP 服务：

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
source .venv/bin/activate
openviking-server --host 127.0.0.1 --port 1933
```

长期运行：

```bash
nohup openviking-server --host 127.0.0.1 --port 1933 \
  > /tmp/openviking-server.log 2>&1 &
```

### 5.2 Docker 部署（生产/共享环境）

```bash
# 准备配置目录
mkdir -p ~/.openviking
touch ~/.openviking/ov.conf

# 运行
docker run -d \
  --name openviking \
  -p 1933:1933 \
  -v ~/.openviking:/app/.openviking \
  --restart unless-stopped \
  ghcr.io/volcengine/openviking:latest
```

> Docker 镜像默认绑定 `0.0.0.0:1933`，必须在 `ov.conf` 中配置 `server.root_api_key`。

### 5.3 Docker Compose 部署

项目已提供 `docker-compose.yml`：

```bash
docker compose up -d
```

默认暴露：
- `1933`：OpenViking API + Web Studio
- `1934`：Caddy 反向代理（兼容旧部署）

---

## 六、运行指令

项目已提供三个 helper 脚本（`scripts/` 目录）：

| 脚本 | 用途 |
|------|------|
| `scripts/start_mlx_server.sh` | 后台启动 Qwen3 MLX 服务 |
| `scripts/start_openviking.sh` | 后台启动 OpenViking 服务 |
| `scripts/stop_all.sh` | 停止两个服务 |

一键启动：

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
./scripts/start_mlx_server.sh
./scripts/start_openviking.sh
```

停止：

```bash
./scripts/stop_all.sh
```

---

### 6.1 启动 MLX 模型服务

```bash
source ~/mlx-env/bin/activate
nohup python /Users/zhengxiaoxi/repo/qwen3_embedding/qwen3_mlx_server.py \
  > /tmp/qwen3_mlx_server.log 2>&1 &
```

### 6.2 启动 OpenViking 服务

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
source .venv/bin/activate

# 使用默认配置 ~/.openviking/ov.conf
openviking-server

# 或指定配置
openviking-server --config /path/to/ov.conf --host 127.0.0.1 --port 1933
```

### 6.3 健康检查

```bash
openviking-server doctor
curl http://127.0.0.1:1933/health
```

### 6.4 CLI 常用命令

```bash
# 设置语言（首次使用）
ov language zh-CN

# 查看状态
ov status

# 列出资源
ov ls viking://resources/

# 添加资源
ov add-resource https://example.com/README.md --wait

# 语义搜索
ov find "what is OpenViking"
```

---

## 七、推荐配置（`~/.openviking/ov.conf`）

```json
{
  "storage": {
    "workspace": "/Users/zhengxiaoxi/.openviking/data-mlx"
  },
  "log": {
    "level": "INFO",
    "output": "stdout"
  },
  "embedding": {
    "dense": {
      "provider": "openai",
      "api_base": "http://localhost:11436/v1",
      "api_key": "dummy",
      "model": "Qwen3-Embedding-8B-4bit-DWQ",
      "dimension": 4096
    },
    "max_concurrent": 1,
    "text_source": "content_only",
    "max_input_tokens": 4096
  },
  "vlm": {
    "provider": "kimi",
    "api_base": "https://api.kimi.com/coding",
    "api_key": "<YOUR_KIMI_API_KEY>",
    "model": "kimi-2.6",
    "temperature": 1.0,
    "max_retries": 2,
    "max_concurrent": 100,
    "timeout": 300
  },
  "rerank": {
    "provider": "openai",
    "api_base": "http://localhost:11436/v1/rerank",
    "api_key": "dummy",
    "model": "Qwen3-Reranker-8B-MLX-4bit",
    "threshold": 0.01
  }
}
```

> **注意**：`kimi-2.6` 要求 `temperature` 必须为 `1.0`，否则报错 `invalid temperature: only 1 is allowed for this model`。

---

## 八、API 测试

### 8.1 导入网络资源

```bash
curl -X POST http://127.0.0.1:1933/api/v1/resources \
  -H "Content-Type: application/json" \
  -d '{
    "path": "https://raw.githubusercontent.com/volcengine/OpenViking/refs/heads/main/README.md",
    "wait": true,
    "to": "viking://resources/readme-main"
  }'
```

> 使用 MLX 本地模型时，`wait: true` 可能需要 **5–10 分钟** 完成 embedding。

### 8.2 语义搜索

```bash
curl -X POST http://127.0.0.1:1933/api/v1/search/find \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is OpenViking",
    "target_uri": "viking://resources/readme-main",
    "limit": 5
  }'
```

### 8.3 文件系统浏览

```bash
curl "http://127.0.0.1:1933/api/v1/fs/ls?uri=viking://resources/readme-main"
```

### 8.4 Python SDK 测试

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
source .venv/bin/activate
python - <<'PY'
import openviking as ov

client = ov.SyncHTTPClient(url="http://127.0.0.1:1933")
client.initialize()

result = client.add_resource(
    path="https://raw.githubusercontent.com/volcengine/OpenViking/refs/heads/main/README.md",
    wait=True,
    to="viking://resources/readme-py"
)
print("root_uri:", result["root_uri"])

results = client.find("what is OpenViking", target_uri=result["root_uri"], limit=5)
for r in results.resources:
    print(f"  {r.uri} (score: {r.score:.4f})")

client.close()
PY
```

---

## 九、故障排查

### 9.1 `uv sync` 报 `AssertionError`（版本解析失败）

**原因**：Git tag `python-sdk/v0.1.2` 不符合 setuptools_scm 正则。  
**解决**：

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.4.4
uv sync --all-extras
```

### 9.2 Embedding 报 `Connection error`

**原因**：`localhost:11436` 没有运行 MLX 服务，或被 `mlx_lm.server` 占用（不支持 embedding）。  
**解决**：

```bash
# 检查端口占用
lsof -i :11436

# 杀掉错误的 mlx_lm.server 进程，启动正确的服务
source ~/mlx-env/bin/activate
python /Users/zhengxiaoxi/repo/qwen3_embedding/qwen3_mlx_server.py
```

### 9.3 VLM 报 `invalid temperature`

**原因**：`kimi-2.6` 要求 `temperature=1.0`。  
**解决**：将 `vlm.temperature` 改为 `1.0`。

### 9.4 切换 embedding 模型后启动失败

**错误**：`EmbeddingRebuildRequiredError`  
**原因**：workspace 中的向量集合元数据与新模型不匹配。  
**解决**：

- 方案 A：使用新的 workspace 路径
- 方案 B：设置 `"embedding": {"allow_metadata_override": true}`（维度必须相同）
- 方案 C：删除旧 workspace 数据重新导入

### 9.5 `find` 返回空结果

可能原因：
- 资源尚未处理完成（embedding 未完成）
- `rerank.threshold` 过高
- 查询与文档语义不匹配

建议将 `rerank.threshold` 从 `0.1` 降低到 `0.01`。

---

## 十、相关文件位置

| 文件/目录 | 说明 |
|-----------|------|
| `/Users/zhengxiaoxi/repo/OpenViking/` | 项目源码 |
| `/Users/zhengxiaoxi/repo/OpenViking/.venv/` | Python 虚拟环境 |
| `/Users/zhengxiaoxi/repo/OpenViking/openviking/bin/ov` | Rust CLI 二进制 |
| `/Users/zhengxiaoxi/repo/OpenViking/openviking/lib/ragfs_python.abi3.so` | RAGFS Rust binding |
| `/Users/zhengxiaoxi/repo/qwen3_embedding/qwen3_mlx_server.py` | MLX 模型服务脚本 |
| `~/.openviking/ov.conf` | OpenViking 主配置 |
| `~/.openviking/ov.conf.backup.*` | 原配置备份 |
| `~/.openviking/data-mlx/` | 当前使用的数据工作区 |
| `~/.openviking/data-api-test/` / `data-mlx-test2/` | 测试用工作区 |

---

## 十一、可选进阶

### 启用 VikingBot

```bash
openviking-server --with-bot
```

### 启用 API Key 认证

```json
{
  "server": {
    "root_api_key": "your-secret-root-key"
  }
}
```

### 连接 OpenClaw

```bash
# 在 OpenClaw 中安装插件
openclaw plugins install clawhub:@openviking/openclaw-plugin
openclaw openviking setup --base-url http://127.0.0.1:1933 --api-key sk-xxx --json
openclaw gateway restart
```
