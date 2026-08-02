# OpenViking 单机部署配置（当前环境快照）

> 生成时间：2026-08-02  
> 来源机器：`82.156.210.246`（ubuntu 用户，`/home/ubuntu`）  
> 用途：将当前正在运行的 OpenViking 服务复刻到另一台机器上部署。

## 一、当前运行的服务

| 服务名 | 类型 | 监听端口 | 进程命令 | 说明 |
|--------|------|----------|----------|------|
| `openviking-textbooks` | OpenViking Server + Bot | `1933`（server）<br>`18791`（bot gateway） | `.venv/bin/python -m openviking_cli.server_bootstrap --config ~/.openviking/ov.conf --port 1933 --with-bot --bot-port 18791` | 教材库检索服务 |
| `openviking-curriculum` | OpenViking Server + Bot | `1934`（server）<br>`18790`（bot gateway） | `.venv/bin/python -m openviking_cli.server_bootstrap --config ~/.openviking/ov-curriculum.conf --port 1934 --with-bot --bot-port 18790` | 课标库检索服务 |

> 注：两个 `vikingbot gateway` 子进程分别由对应 `server_bootstrap` 的 `--with-bot --bot-port` 拉起，不需要单独写 systemd unit。

### 本机端口占用速查

```text
127.0.0.1:1933   python      (openviking-textbooks server)
127.0.0.1:1934   python      (openviking-curriculum server)
127.0.0.1:18791  vikingbot   (textbooks bot gateway)
127.0.0.1:18790  vikingbot   (curriculum bot gateway)
```

## 二、目录与数据

| 路径 | 作用 | 部署时是否需要 |
|------|------|----------------|
| `~/.openviking/data-textbooks-0724` | 教材库 workspace（向量库、元数据、bot sessions 等） | **需要**，可整体 tar 迁移到新机器同路径 |
| `~/.openviking/data-curriculum-0727` | 课标库 workspace | **需要**，同上 |
| `~/.openviking/state` | 运行状态、ws-peer 信息、统计 | 可选，新部署可重建 |
| `~/.openviking/logs` | 日志 | 可选 |
| `~/.openviking/codex-plugin-state` | Codex memory plugin 状态 | 按需迁移 |
| `~/.openviking/memory-plugin-marketplace` | 各 IDE memory plugin 代码/配置 | 按需迁移 |
| `~/.openviking/openviking-ca.crt` | 本地 CA 证书 | 需要 |

## 三、配置文件清单

本包已将 `~/.openviking` 下的核心配置拷贝到 `configs/` 并做脱敏处理（所有 API Key、Token、Hash 均替换为 `<REDACTED>`）。在新机器部署前必须填入真实值。

| 文件 | 用途 | 重要字段 |
|------|------|----------|
| `configs/ov.conf` | 教材库主配置 | `server.port=1933`、`storage.workspace`、`embedding`、`vlm`、`bot.agents` |
| `configs/ov-curriculum.conf` | 课标库主配置 | `server.port=1934`、`storage.workspace`、`embedding`、`vlm` |
| `configs/ov-textbooks.conf` | 教材库备用配置（host=0.0.0.0） | 旧模板，通常用 `ov.conf` |
| `configs/config-curriculum.ov.conf` | 课标库备用配置 | 旧模板，通常用 `ov-curriculum.conf` |
| `configs/ovcli-textbooks.conf` | ovcli 连接教材库 | `url=http://127.0.0.1:1933`、`api_key` |
| `configs/ovcli-curriculum.conf` | ovcli 连接课标库 | `url=http://127.0.0.1:1934`、`api_key` |
| `configs/ovcli.conf` | ovcli 默认云端配置 | 指向 `https://api.vikingdb.cn-beijing.volces.com/openviking`，本机未使用 |
| `configs/ovcli.settings.conf` | ovcli 语言设置 | `language=zh-CN` |
| `configs/.keys.env.example` | Bot Agent 所需环境变量 | `GLM52_*`、`KIMI_SEARCH_API_KEY`、各 viewer/admin key |
| `configs/openviking-ca.crt` | 本地 CA 证书 | 直接使用 |

## 四、部署步骤（新机器）

### 1. 准备环境

```bash
# 创建运行用户（建议与源码同用户）
sudo useradd -m -s /bin/bash ubuntu
# 或直接使用现有用户

# 克隆代码并创建虚拟环境
cd /home/ubuntu/repo/OpenViking
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 创建配置目录并拷贝配置

```bash
mkdir -p ~/.openviking

# 拷贝并修改配置（先修改 configs/*.conf 里的占位符）
cp deploy/standalone-current/configs/ov.conf              ~/.openviking/
cp deploy/standalone-current/configs/ov-curriculum.conf   ~/.openviking/
cp deploy/standalone-current/configs/ovcli-textbooks.conf ~/.openviking/
cp deploy/standalone-current/configs/ovcli-curriculum.conf ~/.openviking/
cp deploy/standalone-current/configs/ovcli.conf           ~/.openviking/
cp deploy/standalone-current/configs/ovcli.settings.conf  ~/.openviking/
cp deploy/standalone-current/configs/openviking-ca.crt    ~/.openviking/

# 创建 keys 文件（chmod 600，勿提交 git）
cp deploy/standalone-current/configs/.keys.env.example ~/.openviking/.keys.env
chmod 600 ~/.openviking/.keys.env
```

> **必须替换的占位符**：所有 `<REDACTED>` 以及 `~` 路径、用户名、`public_base_url` 等。

### 3. 迁移数据目录（如需保留原有向量库）

```bash
# 在原机器打包
ssh ubuntu@82.156.210.246 \
  'tar czf - ~/.openviking/data-textbooks-0724 ~/.openviking/data-curriculum-0727' \
  > openviking-data.tar.gz

# 在新机器解压
tar xzf openviking-data.tar.gz -C /
```

### 4. 安装并启动 systemd 服务

```bash
sudo cp deploy/standalone-current/systemd/openviking-textbooks.service  /etc/systemd/system/
sudo cp deploy/standalone-current/systemd/openviking-curriculum.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now openviking-textbooks
sudo systemctl enable --now openviking-curriculum
```

### 5. 验证

```bash
sudo systemctl status openviking-textbooks openviking-curriculum
ss -tlnp | grep -E '1933|1934|18790|18791'
curl -s http://127.0.0.1:1933/health || echo 'health endpoint not exposed'
```

## 五、环境变量说明

两个 systemd service 文件中通过 `Environment=` 注入以下变量（已在包中脱敏）：

| 变量 | 用途 | 当前配置来源 |
|------|------|--------------|
| `GLM52_API_KEY` | Bot 大模型 API Key（`ov.conf` 中 `${GLM52_API_KEY}` 展开） | `.keys.env` |
| `GLM52_BASE_URL` | Bot 大模型 Base URL | `.keys.env` |
| `GLM52_MODEL` | Bot 大模型名 | `.keys.env` |
| `KIMI_SEARCH_API_KEY` | Kimi 联网搜索后端 Key | `.keys.env` |
| `OPENVIKING_CLI_CONFIG_FILE` | 指定该服务使用哪一份 ovcli 配置 | 分别指向课标/教材 ovcli |
| `PYTHONUNBUFFERED=1` | Python 日志无缓冲 | 固定 |

> 注：`.keys.env` 文件并未被 systemd 直接读取，当前部署是通过 `Environment=` 逐条写在 service 文件里的。如需改为 `EnvironmentFile=~/.openviking/.keys.env`，可自行调整。

## 六、网络/安全注意事项

1. 当前服务均监听 `127.0.0.1`，外网访问需通过 Nginx/Caddy/Clash 等反向代理。  
2. `ov.conf` 中 `auth_mode: null` 表示未启用鉴权；生产环境建议改为 `dev` 或更严格的模式并配置 `root_api_key`。  
3. 所有 `<REDACTED>` 字段必须替换为真实值，否则服务启动后会调用失败。  
4. CA 证书 `openviking-ca.crt` 有效期为 2026-07-30 至 2036-07-27，到期前需重新生成。

## 七、服务管理命令速查

```bash
# 查看状态
sudo systemctl status openviking-textbooks openviking-curriculum

# 重启
sudo systemctl restart openviking-textbooks openviking-curriculum

# 查看日志
sudo journalctl -u openviking-textbooks -f
sudo journalctl -u openviking-curriculum -f

# 停止
sudo systemctl stop openviking-textbooks openviking-curriculum
```

## 八、脱敏说明

本包内所有 `.conf`、`.service`、`.keys.env.example` 均已将真实 API Key、Token、Hash、路径中的用户名替换为占位符。部署前请按上表逐项替换。
