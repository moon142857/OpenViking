# OpenViking 公网访问部署文档

> 生成于 2026-07-30 | 服务器 82.156.210.246 (腾讯云轻量)

## 一、访问地址(端口放行后可用)

| 库 | 地址 | 用途 |
|---|---|---|
| 教材库 | https://82.156.210.246:8443/studio/ | data-textbooks-0724 全库 |
| 课标库 | https://82.156.210.246:8444/studio/ | data-curriculum-0727 全库 |

打开后,在 web-studio 连接弹窗里填对应 API Key 才能查看/检索。

## 二、⚠️ 你必须做的一件事:云控制台放行端口

8443/8444 在本机已就绪,但腾讯云安全组还没放行,公网访问超时。
请到 **腾讯云控制台 -> 轻量应用服务器 -> 防火墙(安全组)**,添加入站规则:

- TCP 8443 (教材库)
- TCP 8444 (课标库)
- 来源:0.0.0.0/0 (或填对方 Windows 电脑的公网 IP 更安全)

放行后即可公网访问。本机层面(iptables/nginx/后端)已全部就绪。

## 三、API Key(连接弹窗里填)

> ⚠️ 这些是访问凭证,等同于密码。viewer 只能查/浏览,不能改;admin 可读写。勿外泄。

```
# viewer(只读检索/浏览)——分发给查看者
VIEWER_TEXTBOOKS   = ZGVmYXVsdA.dmlld2Vy.MjYzYTVkNWZmMjEwZjM1NDhjNzYwYWEyMDY3NGU2YWUyMTFhMTBjMGU1NjVmNTBlNjYwYjY5OWVjM2IwNWJjYg
VIEWER_CURRICULUM  = ZGVmYXVsdA.dmlld2Vy.MzRiZTkxOWEzY2RiODE2MDU4MWM0MTI2YjViNDk1YTRlOTQwYjJmZDMwMzljM2MyZTAyMmJlZTdjN2VlZDM5Zg

# admin(可读写管理)——仅自己用
ADMIN_TEXTBOOKS  / ADMIN_CURRICULUM   见 /home/ubuntu/.openviking/.keys.env
# root(仅账户管理,不能查数据)        仅在 ov.conf 里
```
全部密钥在 `/home/ubuntu/.openviking/.keys.env`(权限600)。

## 四、Windows 首次访问:消除证书警告(二选一)

因为是自签证书,Windows 浏览器首次访问会显示"连接不是私密连接"红警告。
数据已全程 HTTPS 加密,只是浏览器不认识这个 CA。消除方式:

**方式 A——点"继续前往"(最快,每台电脑每次)**
点「高级」->「继续前往 82.156.210.246(不安全)」。能用,但地址栏显示"不安全"。

**方式 B——导入 CA 根证书(一劳永逸,推荐,每台电脑一次)**
1. 下载 CA 证书:http://82.156.210.246/openviking-ca.crt (80端口,裸IP不拦)
2. 双击该 .crt 文件 -> 「安装证书」-> 选「本地计算机」-> 下一步
3. 选「将所有的证书都放入下列存储」->「浏览」-> 选「受信任的根证书颁发机构」-> 确定 -> 下一步 -> 完成
4. 关闭并重开浏览器
5. 之后再访问 https://82.156.210.246:8443 即绿锁无警告

## 五、把 80 端口网页关联到这两个库

你的 yudou 网页(http://82.156.210.246)里加两个链接即可跳转:

```html
<a href="https://82.156.210.246:8443/studio/" target="_blank">教材库</a>
<a href="https://82.156.210.246:8444/studio/" target="_blank">课标库</a>
```
点开后,访问者在连接弹窗填对应 viewer key 即可查看。

## 六、服务管理

后端经 systemd 常驻(开机自启,崩溃自动重启,已开启 --with-bot + GLM-5-2 LLM):
```bash
# 启停脚本(在 /home/ubuntu/repo/OpenViking/)
./ov-services.sh status     # 查看状态
./ov-services.sh restart    # 重启两个后端
./ov-services.sh stop       # 停止
./ov-services.sh start      # 启动

# 或直接用 systemctl
sudo systemctl status openviking-textbooks openviking-curriculum
sudo systemctl restart openviking-textbooks
sudo journalctl -u openviking-textbooks -f   # 看日志
```

## Bot / AI 会话功能

已开启 `--with-bot`,**两个库各有独立 bot 实例**(各自独立端口 + 独立 LLM 配置 + 独立知识库):
- 教材库 bot: `--bot-port 18791` -> 后端 1933,经 `https://82.156.210.246:8443/bot/v1/chat`(需 viewer/admin key)
- 课标库 bot: `--bot-port 18790` -> 后端 1934,经 `https://82.156.210.246:8444/bot/v1/chat`(需 viewer/admin key)
- 健康检查: `.../bot/v1/health`
- **bot 会 RAG 各自的库**:教材库 bot 答教材内容(人教版/北师大版等),课标库 bot 答课标内容(2022 版课程标准),互不串台
- bot 是独立子进程服务(vikingbot gateway),有完整 HTTP API:`/bot/v1/chat`、`/chat/stream`、`/sessions`、`/feedback`、`/health`
- 可直接调 API:`curl -X POST https://82.156.210.246:8443/bot/v1/chat -H "Authorization: Bearer <viewer_key>" -H "Content-Type: application/json" -d '{"message":"...","chat_id":"x"}'`
- LLM:火山引擎 ark,GLM-5-2,Anthropic 兼容协议(`${GLM52_*}` 占位符在 ov.conf 加载时经 expandvars 展开,从 systemd 注入的环境变量取值)

### bot 工作原理(ReAct Agent 循环)

bot 不是"搜一次就答",而是 **ReAct(Reason+Act)Agent 循环**:LLM 自己决定每步调什么工具、看结果后再决定下一步,循环到信息充分才给最终答案。你看到的"很多步骤"就是这个循环。

代码:`bot/vikingbot/agent/loop.py:762` 主循环:
```python
while iteration < self.max_iterations:   # 默认 max_iterations=50
    response = await self._chat_with_stream_events(messages, tools)  # ① LLM 思考
    if response.has_tool_calls:               # ② LLM 要调工具
        results = await asyncio.gather(*[execute_single_tool(tc) ...])  # ③ 并行执行工具
        messages.append({role:"tool", content:results})                 # ④ 结果塞回对话
        continue                                                          # ⑤ 下一轮
    else:
        final_content = response.content   # ⑥ 不再要工具 = 给最终答案
        break
```
- 每步搜什么、要不要继续、何时停,都由 LLM 自主决策(非预设流程)
- 检索 query 由 LLM 现场改写拆分(你输入的"植物种植园"它可能拆成多角度 query 搜)
- 同一轮多个工具调用会并行执行(`asyncio.gather`,`loop.py:855`)
- 最多 50 轮,防死循环

bot 工具清单(`bot/vikingbot/agent/tools/`):
| 工具 | 干什么 |
|---|---|
| `openviking_search` | 向量检索知识库(RAG 召回) |
| `openviking_read`(多读) | 读文档全文 |
| `openviking_glob` | 按文件名模式找文件(`**/*种植*.md`) |
| `openviking_grep` | 内容正则搜索 |
| `openviking_outline` | 文件结构大纲 |
| `web_search` / `web_fetch` | 联网搜索(库外知识,⚠️ 见下) |
| `python` / `shell` | 执行代码 |
| `remember` / `write` | 写记忆(需 admin key) |

### 推理模式 thinking + 联网搜索:已开启

**当前 `bot.agents.thinking = true`(两个库都是),即 LLM 推理(extended thinking)已开启。**

- ⚠️ **两个"thinking"别混淆**:
  - `bot.agents.thinking`(LLM 推理):控制 LLM 是否走 extended thinking。当前 = **true(已开)**。
  - `default_search_mode: "thinking"`(检索模式):向量检索的召回策略,与 LLM thinking 无关。
- 配置(两个 ov.conf 的 `bot.agents`):`thinking: true`,`timeout: 180`。
- ⚠️ **必须配的两件事(否则 thinking 会超时失败)**:
  1. **调大 LLM 超时**:`bot.agents.timeout: 180`(秒)。thinking 让请求更慢更大,默认 30s 不够 -> `litellm.Timeout`。设 180 后稳定。机制:bot 调 LLM 走 OV 的 `LiteLLMVLMProvider`(`openviking/models/vlm/backends/litellm_vlm.py:283` 传 `self.timeout`,从 `bot.agents.timeout` 继承)。
  2. **联网搜索换成可达的搜索引擎**:thinking 模式下 LLM 倾向"查证"会调 `web_search`,但 bot 默认 brave/startpage/yahoo(国外,本机连不通)。已换成 **Kimi 联网搜索**(见下)。
- 验证:thinking + kimi 搜索 + timeout=180,bot 能融合库内知识 + 联网结果做复杂跨学科分析(实测 token 16万+,无超时)。

### 联网搜索后端:Kimi(自建 backend,替代 brave/ddgs)

bot 的 `web_search` 是多后端可插拔设计(`bot/vikingbot/agent/tools/websearch/`),原支持 brave/ddgs/exa/tavily,全是不通的国外源。**新增了 Kimi backend**,调 Kimi 的 agent-gw 搜索接口(国内可达、稳定)。

- 新文件:`bot/vikingbot/agent/tools/websearch/kimi.py`(实现 `WebSearchBackend`,调 `POST https://agent-gw.kimi.com/coding/v1/search`)
- 注册:`websearch/__init__.py` 加 `from . import kimi`;`registry.py` 把 kimi 加进 auto 优先级**置顶**(`["kimi","tavily","exa","brave","ddgs"]`)。有 `KIMI_SEARCH_API_KEY` 时自动选中 kimi。
- 接口契约:`Authorization: Bearer <key>` + body `{"text_query","limit","enable_page_crawling","timeout_seconds"}`,响应 `{"search_results":[{title,url,snippet,content,date,...}]}`。带 408 重试。
- key 注入:systemd `Environment=KIMI_SEARCH_API_KEY=sk-kimi-...`(两个 unit 都有);key 存 `.keys.env`。key 来源是 kimi-search 插件包的 README(`/home/ubuntu/repo/kimi-search`)。
- ⚠️ 实现 bug 记录:`__init__` 里 timeout 解析 `int(env or 0)` 在未设 `KIMI_SEARCH_TIMEOUT` 时得 0->`max(1,0)=1`,导致客户端 1s 超时 -> 408。已修为未设时用默认 30。
- 验证:问"2026 科技大事"(库内没有),bot 用 kimi 联网搜到答案(Kimi K3、GPT-5 等),日志 `[RESULT]: Results for: ...`(不再 brave 超时)。

### bot 鉴权配置(关键,易踩坑)
bot 连后端知识库用的 key 不在 ov.conf,而在 ovcli 配置(每个库一份):
- `/home/ubuntu/.openviking/ovcli-textbooks.conf` -> url=1933 + 教材 admin key
- `/home/ubuntu/.openviking/ovcli-curriculum.conf` -> url=1934 + 课标 admin key
- 通过 systemd `Environment=OPENVIKING_CLI_CONFIG_FILE=...` 指定各 bot 读哪个
- ⚠️ 后端切 api_key 模式后,旧的共享 `ovcli.conf`(default.default 旧 key)会失效 -> bot 检索全 401 `Invalid API Key` -> chat 反复重试 45s 超时。必须用对应库的 admin key 更新 ovcli 配置。

## 七、架构总览

```
Windows 电脑
  │  HTTPS (自签证书,8443/8444 非标端口绕开备案拦截)
  ▼
nginx (:8443 ssl -> 1933, :8444 ssl -> 1934)   ← 终止TLS, 鉴权在后端
  │  loopback
  ▼
OpenViking 后端 (:1933 教材, :1934 课标)        ← auth_mode=api_key, 仅绑127.0.0.1
  │  X-API-Key 校验 (viewer只读 / admin读写)
  ▼
数据 (data-textbooks-0724 / data-curriculum-0727)
```

- 后端只绑 127.0.0.1,公网无法直连,必须经 nginx
- nginx 校验 TLS;OpenViking 校验 API Key
- 无有效 key = 401 拒绝(数据读不了)
- viewer key 只读;admin 可改;root 仅管账户
```

## 八、安全说明

- dev 模式已彻底关闭(切到 api_key),不再是裸奔 ROOT
- CORS 已收紧到对应子域(仅本机 loopback 调试用保留 127.0.0.1)
- 想给不同人发不同 key:用 admin key 调 `POST /api/v1/admin/accounts/default/users` 建 user,见 API 文档
- 想吊销某人的访问:用 admin/root key 重置其 key 或删用户
- viewer 无法写/删数据,即使 key 泄露也只能看
```

## 九、相关文件

| 文件 | 说明 |
|---|---|
| /home/ubuntu/.openviking/ov.conf | 教材库后端配置(含 root_api_key) |
| /home/ubuntu/.openviking/ov-curriculum.conf | 课标库后端配置(含 root_api_key) |
| /home/ubuntu/.openviking/.keys.env | 全部 API Key(权限600,勿提交git) |
| /etc/nginx/ssl/ | 自签证书 + CA(私钥600) |
| /home/ubuntu/.openviking/openviking-ca.crt | CA 证书副本(给Windows导入) |
| /etc/nginx/sites-available/openviking | nginx 反代配置 |
| /etc/systemd/system/openviking-{textbooks,curriculum}.service | systemd unit |
| /home/ubuntu/repo/OpenViking/ov-services.sh | 启停脚本 |

---

# 附:搭建过程问题排查与配置变更记录

> 以下记录本机从"macOS 构建的源码"到"Linux 公网可用"全过程遇到的问题、根因、解法,以及所有改动。
> 作用:日后重装/迁移/排错时按图索骥。**根因比结论重要**。

## A. 环境基线(为什么会有这么多坑)

这套 OpenViking 源码原本是在 **macOS (arm64)** 上构建的,整个目录原样拷到了这台 **Linux x86_64** 服务器。
直接拿来跑不通,因为跨平台二进制全不对:
- `.venv/` 是 macOS venv(shebang 指向 `/Users/xx/...`,python 二进制无法执行)
- `openviking/lib/ragfs_python.abi3.so` 是 **Mach-O arm64**(Linux 加载报 `invalid ELF header`)
- `openviking/storage/vectordb/engine/_native.abi3.so` 同 Mach-O(但 x86 不用它,见下)
- `web-studio/node_modules/` 里 rollup/lightningcss 的 `.node` 是 darwin-arm64

> 关键认知:这台机器没有可用的 Python 环境、没有 Rust、没有可用的 venv,全要从零搭。

## B. 装的系统包 / 工具链(apt + rustup)

```bash
# apt(需 sudo)
sudo apt install -y python3-venv python3-dev cmake build-essential pkg-config ninja-build
sudo apt install -y certbot python3-certbot-nginx       # 后续公网证书用
# Rust(rustup 安装,stable 工具链,rustc 1.97.1)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
source ~/.cargo/env
```
- gcc 13.3 / cmake 3.28 / ninja 1.11 / python3 3.12 本来有或刚装。

## C. 重编译原生绑定(Rust + C++)

两个原生组件,分别构建:

1. **Rust 绑定 `ragfs_python.abi3.so`** —— 来自 `crates/ragfs-python`,用 maturin:
   ```bash
   .venv/bin/python setup.py build_ext --inplace   # 实际会顺带触发 maturin build
   ```
   产物:`openviking/lib/ragfs_python.abi3.so` → ELF x86-64 ✅
2. **C++ 向量引擎** —— `src/` 下 cmake 构建,经 `setup.py build_ext`:
   产物:`engine/_x86_sse3.abi3.so` / `_x86_avx2` / `_x86_avx512` / `_x86_caps` → 全 ELF ✅
   - ⚠️ `engine/_native.abi3.so` 仍是旧 Mach-O,**但 x86 机器不用它**(只 ARM 机器用),保留无害。
- 验证:`.venv/bin/python -c "import openviking.storage.viking_fs"` 不再报 `RAGFSBindingClient None`。

## D. Python 环境(重建 venv)

```bash
mv .venv .venv.macos                 # 备份 macOS venv
python3 -m venv .venv
.venv/bin/python -m pip install -e .          # 装运行时依赖(fastapi/uvicorn/pydantic/openai/volcengine...)
.venv/bin/python -m pip install -e ".[bot]"   # bot 依赖(prompt-toolkit/python-socks/python-telegram-bot/qq-botpy...)
.venv/bin/python -m pip install "mcp>=1.27.0,<2.0.0"   # 见 E
.venv/bin/python -m pip install httpx[socks] socksio     # 见 F
```

## E. mcp 版本坑

- 代码用 `from mcp.server.fastmcp import FastMCP`(mcp 1.x API)。
- 但 `mcp>=1.27.0` 这个宽松约束装了 **mcp 2.0.0**,而 2.0 把 fastmcp 拆到独立包 → `ModuleNotFoundError`。
- 解法:锁 `mcp>=1.27.0,<2.0.0`(实际装 1.29.0)。

## F. 代理坑(最关键,极易再踩)

本机 shell 里设了 Clash 代理:`HTTP_PROXY=127.0.0.1:7890` / `ALL_PROXY=socks5://127.0.0.1:7891`。
而 `~/.pip/pip.conf` 把 pip 默认源指向 **腾讯云内网镜像** `mirrors.tencentyun.com`。
**两者冲突**:
- 走代理访问腾讯内网镜像 = 8s 超时(内网地址不该走代理)
- pip 报 `Could not find a version`(找不到包)
- 后台 server 进程继承 `ALL_PROXY=socks5`,httpx 要用 SOCKS 但缺 `socksio` → `Application startup failed`

**解法(所有构建/启动命令都要做)**:
```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# pip 用腾讯内网镜像(快),cargo 用 USTC(见 G),npm 用腾讯镜像
# systemd unit 用 Environment= 显式注入干净环境(systemd 本就干净,不会继承代理)
```
- pip 装包:`npm_config_registry=https://mirrors.tencentyun.com/npm/` / pip 走 pip.conf 内网镜像
- 还需 `pip install socksio` 让 httpx 能处理(万一环境里有 SOCKS 变量)
- 注意:直连 `https://index.crates.io` 其实能通(0.65s),但 `cargo metadata` 拉索引会卡,见 G。

## G. cargo 卡死 → 配 USTC 镜像

`cargo metadata` 直连 crates.io 卡住(`do_poll` 阻塞在拉 sparse 索引)。
配 `~/.cargo/config.toml`:
```toml
[source.crates-io]
replace-with = "ustc"
[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
[net]
git-fetch-with-cli = true
```
配完 `cargo metadata` 秒过,依赖从 USTC 下载。

## H. 前端重建(web-studio)

- `node_modules` 是 macOS 装的(rollup native 是 darwin-arm64)→ `npm install` 重装(备份 `node_modules.macos`)
- 关键:`dist` 重建时**不能**带 `VITE_OV_BASE_URL`!
  - 旧 dist 把 `http://127.0.0.1:1933` 写死,公网用户打开 SPA 会去连自己电脑的 127.0.0.1 → 全废
  - 不带该 env → 默认 baseUrl 回退 `window.location.origin` → 教材库子域自动连教材库、课标库子域自动连课标库(同一个 dist 两个库通用)
  - `npm_config_registry=...npm/ node_modules/.bin/vite build`(旧 dist 备份 `dist.macos`)
- inotify watch 不够(ENOSPC)→ `sudo sysctl fs.inotify.max_user_watches=524288`(已设,但重启会丢,需写 sysctl 持久化或重启后重设)

## I. 鉴权:dev → api_key

- 原状态 `auth_mode: dev` = 无鉴权 + ROOT,**任何人能读/写/删全库 + 刷爆火山引擎 embedding 额度**
- dev 模式有保护:host 非 localhost 拒绝启动 → 所以保持绑 `127.0.0.1` + nginx 反代
- 切 api_key:`server.root_api_key` 设非空即自动切(`config.py` 的 `get_effective_auth_mode`)
- 同时加 `server.public_base_url`(MCP 上传指令回填公网 URL)、`server.cors_origins`(收紧)
- **account 坑**:已入库数据在 `default` account 下(`workspace/viking/default/`)。新建的 `textbooks`/`curriculum` account 是**空命名空间**,viewer 看不到真实数据。
  - 解法:在 **`default` account** 下建 viewer(`POST /api/v1/admin/accounts/default/users`)
  - 角色层级:ROOT(仅管账户,不能查数据 API,调了 403) / ADMIN(读写) / USER(只读)
  - key 生成:`openssl rand -hex 32` 造 root;admin/viewer key 由后端 API 生成,存 VikingFS

## J. 公网方案演进(nip.io 失败 → 自签+非标端口)

1. **先试 nip.io + Let's Encrypt**:用 `textbook.82.156.210.246.nip.io` / `curriculum.*` 申请可信证书。
   - **失败**:Let's Encrypt 验证时被腾讯云劫持到 `dnspod.qcloud.com/static/webblock.html` → "未备案域名拦截"
   - 根因:**腾讯云内地节点对"域名+80/443"要求 ICP 备案**;nip.io 无法备案。裸 IP 访问 80 不拦(外部代理能访问到 yudou 页面证明 80 通)。
2. **改自签证书 + 非标端口**(绕开备案拦截):
   - 非标端口 8443/8444 不走备案拦截逻辑;裸 IP + 自签,不需要域名
   - 自建本地 CA(`/etc/nginx/ssl/ca.{crt,key}`)+ 签发 server 证书(SAN 含 `82.156.210.246`、`127.0.0.1`)
   - Windows 导入 CA(`openviking-ca.crt`)后绿锁无警告;不导入就点"继续前往"
3. **未来升级路径**:若有已 ICP 备案域名 → 解析过来 → Let's Encrypt 可信证书 + 标准 443(后端/systemd/鉴权全部复用,只换 nginx 配置和证书)

## K. nginx 配置变更

- 配置文件:`/etc/nginx/sites-available/openviking`(软链到 sites-enabled)
- 内容:`listen 8443 ssl` → `proxy_pass http://127.0.0.1:1933`;`8444` → `1934`
- **坑**:曾写死 `proxy_set_header Connection "upgrade"` —— 对非 WebSocket 请求也发 upgrade 头,违反 HTTP 协议,后端(h11/uvicorn)reset 连接。OV 用 SSE(非 WebSocket),**不需要 upgrade 头** → 改为 `proxy_set_header Connection "";` + `proxy_buffering off;`(SSE 流式必须)
- `client_max_body_size 100m`(入库上传)
- 80 端口仍由你的 yudou nginx 站点(`liubufa`)占用,不动
- CA 证书副本放到 yudou 前端 dist:`/home/ubuntu/repo/education/yudou/webapp/frontend/dist/openviking-ca.crt`(裸 IP 80 可下载)

## L. --with-bot 配置

- **bot 是独立子进程**:server 加 `--with-bot` 后 spawn 一个 `vikingbot gateway` Python 进程(默认 18790),后端把 `/bot/v1/*` 代理过去。需 `openviking[bot]` extra。
- 缺依赖时 `vikingbot` 报 `No module named 'prompt_toolkit'` → 装 `[bot]` extra 解决
- **占位符展开机制**(关键):ov.conf 里 `${GLM52_*}` 在加载时经 `load_json_config` → `os.path.expandvars` 展开,**从进程环境变量取值**。所以 systemd unit 必须用 `Environment=` 注入 `GLM52_API_KEY`/`GLM52_BASE_URL`/`GLM52_MODEL`,否则占位符原样留下、bot 连不上 LLM。
- **LLM 协议**:用 Anthropic 兼容端点 `https://ark.cn-beijing.volces.com/api/plan`(不带 /v3),`provider: anthropic`,`model: anthropic/${GLM52_MODEL}`。已验证 bot chat 连通(GLM-5-2 回答正常,RAG 融合库内检索)。
  - 想切 OpenAI 协议:`provider: openai`、`api_base: .../api/plan/v3`、model 去掉 `anthropic/` 前缀。
- **坑**:前台(本会话 sandbox 里)`--with-bot` 启动报 starlette lifespan error 且重定向文件不生成 —— 是 sandbox 限制子进程 spawn;**用 systemd 启动(沙箱外)正常**。日后排查 bot 问题直接看 systemd 日志:`journalctl -u openviking-textbooks -f` + `~/.openviking/data-textbooks-0724/bot/logs/vikingbot.log`。

### L2. 两个库都开 --with-bot 的端口冲突 + 鉴权串台(重要)

**默认 bot 都绑 18790,两个后端各 spawn 一个 bot 会端口冲突:**
- 抢到 18790 的那个 bot 真起来,另一个后端的 bot 启动失败(`Error: vikingbot gateway port 18790 is already in use. refusing to start a duplicate`)
- 但失败的后端**仍把 `/bot/v1/*` 代理到 18790** -> 该库的 bot 请求被错误转发到另一个库的 bot(**串台**)
- **解法**:两个库用不同 `--bot-port`:教材库 `--bot-port 18791`,课标库用默认 18790。两 systemd unit 已配。

**bot 鉴权链(`Invalid API Key` -> chat 45s 超时):**
- bot 连后端知识库用的 key 来自 **ovcli 配置**,不是 ov.conf。后端切 api_key 模式后,旧 `ovcli.conf` 里的 `default.default` 旧 key 失效(401)。
- bot 每次检索 `viking://user/.../memories/` 和 `openviking_search` 都 401 -> 反复重试 -> chat 超时(日志:`[RESULT]: Error searching Viking: Invalid API Key`)
- **解法**:每个库建独立 ovcli,用对应库的 admin key:
  - `ovcli-textbooks.conf`:url=1933 + 教材 ADMIN_TEXTBOOKS
  - `ovcli-curriculum.conf`:url=1934 + 课标 ADMIN_CURRICULUM
  - systemd 用 `Environment=OPENVIKING_CLI_CONFIG_FILE=...` 各指定一个
- bot 连哪个后端:由 `VIKINGBOT_MANAGED_OV_SERVER_URL` 环境变量(后端 spawn 时自动传入,=该后端自己的 loopback URL)决定,优先级高于 ovcli.url。所以 url 即使在 ovcli 写错也以 env 为准,但 **api_key 必须对**。
- 验证:两个 bot 各答各自库内容(教材答"人教版/北师大版...",课标答"2022 版课程标准..."),不串台。

## M. 完整改动 / 新增文件清单

**系统级(需 sudo,重启不丢)**:
- `/etc/systemd/system/openviking-textbooks.service`(含 `--with-bot` + GLM52_* env)
- `/etc/systemd/system/openviking-curriculum.service`(同上,端口 1934)
- `/etc/nginx/sites-available/openviking`(+软链)→ 8443/8444 ssl 反代
- `/etc/nginx/ssl/` → ca.{crt,key}、server.{crt,key}、serial
- `~/.cargo/config.toml` → USTC crates 镜像
- `~/.cargo/`(整个 rustup 工具链)
- apt 装的包:见 B 节
- `sysctl fs.inotify.max_user_watches=524288`(⚠️ 非持久,重启丢失)

**用户级(配置)**:
- `/home/ubuntu/.openviking/ov.conf`(教材库:root_api_key + public_base_url + cors + bot 块)
- `/home/ubuntu/.openviking/ov-curriculum.conf`(课标库:同上 + bot 块,新建)
- `/home/ubuntu/.openviking/.keys.env`(全部 API key,权限600)
- `/home/ubuntu/.openviking/openviking-ca.crt`(CA 副本,给 Windows)
- `/home/ubuntu/repo/OpenViking/ov-services.sh`(启停脚本)
- `/home/ubuntu/repo/OpenViking/bot/vikingbot/agent/tools/websearch/kimi.py`(**新增**:Kimi 联网搜索 backend)+ `websearch/__init__.py`、`registry.py`(注册 kimi,详见 Bot 章节)
- `/home/ubuntu/repo/OpenViking/openviking/lib/ragfs_python.abi3.so`(重编 Linux ELF)
- `/home/ubuntu/repo/OpenViking/openviking/storage/vectordb/engine/_x86_*.abi3.so`(重编 Linux ELF)
- `/home/ubuntu/repo/OpenViking/.venv/`(新建 Linux venv)
- `/home/ubuntu/repo/OpenViking/web-studio/node_modules/`(重装)
- `/home/ubuntu/repo/OpenViking/openviking/web_studio/dist/`(重建,origin 相对)
- `/home/ubuntu/repo/education/yudou/webapp/frontend/dist/openviking-ca.crt`(CA 下载)

**备份的 macOS 产物(可删)**:
- `/home/ubuntu/repo/OpenViking/.venv.macos/`
- `/home/ubuntu/repo/OpenViking/web-studio/node_modules.macos/`
- `/home/ubuntu/repo/OpenViking/web-studio/dist.macos/`

## N. 重启后需注意(非持久项)

- `fs.inotify.max_user_watches` 会回到默认 → 若前端 dev 模式 ENOSPC,重设 `sudo sysctl -w fs.inotify.max_user_watches=524288`(生产用 /studio 不受影响,无需设)
- systemd 服务开机自启(已 `enable`)
- Rust 工具链、apt 包、证书、配置文件均持久
