#!/usr/bin/env python3
"""诊断 OpenViking 性能瓶颈：拆分 embedding / rerank / vector search / fs 耗时"""

import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlencode

BASE_URL = "http://127.0.0.1:1933"
EMBED_URL = "http://127.0.0.1:11436/v1/embeddings"
RERANK_URL = "http://127.0.0.1:11436/v1/rerank"


def http_call(url, method="GET", body=None, timeout=60):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
            return {"ok": True, "status": resp.status, "elapsed_ms": (time.time() - t0) * 1000}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "elapsed_ms": (time.time() - t0) * 1000, "error": e.read().decode()[:200]}
    except Exception as e:
        return {"ok": False, "status": None, "elapsed_ms": (time.time() - t0) * 1000, "error": str(e)}


def main():
    root_uri = "viking://resources/api-test-readme"
    file_uri = f"{root_uri}/Overview.md"
    query = "OpenViking filesystem"

    results = {}

    # 1. 基础 HTTP 开销
    results["health"] = http_call(f"{BASE_URL}/health")

    # 2. 直接测 embedding server
    results["embed_direct_1st"] = http_call(EMBED_URL, "POST", {"input": query, "model": "Qwen3-Embedding-8B-4bit-DWQ"}, timeout=120)
    time.sleep(1)
    results["embed_direct_2nd"] = http_call(EMBED_URL, "POST", {"input": query, "model": "Qwen3-Embedding-8B-4bit-DWQ"}, timeout=60)

    # 3. 直接测 rerank server
    results["rerank_direct_1st"] = http_call(RERANK_URL, "POST", {
        "model": "Qwen3-Reranker-8B-MLX-4bit",
        "query": query,
        "documents": ["OpenViking is a context database for AI agents.", "This is unrelated."]
    }, timeout=120)
    time.sleep(1)
    results["rerank_direct_2nd"] = http_call(RERANK_URL, "POST", {
        "model": "Qwen3-Reranker-8B-MLX-4bit",
        "query": query,
        "documents": ["OpenViking is a context database for AI agents.", "This is unrelated."]
    }, timeout=60)

    # 4. OpenViking 内部接口
    results["fs_ls"] = http_call(f"{BASE_URL}/api/v1/fs/ls?{urlencode({'uri': root_uri})}")
    results["content_read"] = http_call(f"{BASE_URL}/api/v1/content/read?{urlencode({'uri': file_uri})}")
    results["search_find"] = http_call(f"{BASE_URL}/api/v1/search/find", "POST", {
        "query": query, "target_uri": root_uri, "limit": 3
    }, timeout=120)
    results["observer_system"] = http_call(f"{BASE_URL}/api/v1/observer/system")

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
