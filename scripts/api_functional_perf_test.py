#!/usr/bin/env python3
"""
OpenViking API 功能与性能测试脚本（修正版）
输出 JSON 报告到 stdout
"""

import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlencode
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:1933"


def call(method, path, body=None, query=None, timeout=60):
    url = f"{BASE_URL}{path}"
    if query:
        url += "?" + urlencode(query)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = (time.time() - t0) * 1000
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw[:500]
            return {"ok": True, "status": resp.status, "elapsed_ms": elapsed, "data": payload}
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - t0) * 1000
        raw = e.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw[:500]
        return {"ok": False, "status": e.code, "elapsed_ms": elapsed, "error": payload}
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return {"ok": False, "status": None, "elapsed_ms": elapsed, "error": str(e)}


def perf(method, path, body=None, query=None, n=20, timeout=60):
    times = []
    ok_count = 0
    for _ in range(n):
        r = call(method, path, body=body, query=query, timeout=timeout)
        times.append(r["elapsed_ms"])
        if r["ok"]:
            ok_count += 1
        time.sleep(0.05)
    times.sort()
    return {
        "n": n,
        "ok": ok_count,
        "min_ms": round(times[0], 2),
        "p50_ms": round(times[len(times)//2], 2),
        "p95_ms": round(times[int(len(times)*0.95)], 2),
        "p99_ms": round(times[int(len(times)*0.99)] if len(times) >= 2 else times[-1], 2),
        "max_ms": round(times[-1], 2),
        "avg_ms": round(sum(times)/len(times), 2),
    }


def main():
    results = {}
    root_uri = "viking://resources/api-test-readme"
    file_uri = f"{root_uri}/Overview.md"

    # System / Health
    results["health"] = call("GET", "/health")
    results["system_status"] = call("GET", "/api/v1/system/status")
    results["system_ready"] = call("GET", "/ready")

    # Filesystem
    results["fs_ls_root"] = call("GET", "/api/v1/fs/ls", query={"uri": "viking://resources/"})
    results["fs_ls_resource"] = call("GET", "/api/v1/fs/ls", query={"uri": root_uri})
    results["fs_tree"] = call("GET", "/api/v1/fs/tree", query={"uri": root_uri, "depth": "2"})
    results["fs_stat"] = call("GET", "/api/v1/fs/stat", query={"uri": root_uri})

    # Content
    results["content_read"] = call("GET", "/api/v1/content/read", query={"uri": file_uri})
    results["content_abstract"] = call("GET", "/api/v1/content/abstract", query={"uri": root_uri})
    results["content_overview"] = call("GET", "/api/v1/content/overview", query={"uri": root_uri})

    # Search
    results["search_find"] = call(
        "POST",
        "/api/v1/search/find",
        body={"query": "what is OpenViking", "target_uri": root_uri, "limit": 3},
        timeout=120,
    )
    results["search_grep"] = call(
        "POST",
        "/api/v1/search/grep",
        body={"pattern": "OpenViking", "uri": root_uri, "limit": 3},
        timeout=60,
    )
    results["search_glob"] = call(
        "POST",
        "/api/v1/search/glob",
        body={"pattern": "**/*.md", "target_uri": root_uri, "limit": 5},
        timeout=60,
    )

    # Observer
    results["observer_system"] = call("GET", "/api/v1/observer/system")
    results["observer_filesystem"] = call("GET", "/api/v1/observer/filesystem")
    results["observer_queue"] = call("GET", "/api/v1/observer/queue")
    results["observer_models"] = call("GET", "/api/v1/observer/models")

    # Stats
    results["sessions"] = call("GET", "/api/v1/sessions")

    # Skills
    results["skills_list"] = call("GET", "/api/v1/skills")

    # Console
    results["console_dashboard"] = call("GET", "/api/v1/console/dashboard/summary")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    results["console_tokens"] = call(
        "GET",
        "/api/v1/console/tokens",
        query={"start_date": today, "end_date": today},
    )

    # Watches
    results["watches_list"] = call("GET", "/api/v1/watches")

    # Relations
    results["relations_list"] = call("GET", "/api/v1/relations", query={"uri": root_uri})

    # Privacy configs
    results["privacy_configs"] = call("GET", "/api/v1/privacy-configs")

    # Code tools
    results["code_outline"] = call(
        "POST",
        "/api/v1/code/outline",
        body={"uri": file_uri, "language": "markdown"},
        timeout=60,
    )

    # Performance tests
    results["perf_health"] = perf("GET", "/health", n=50)
    results["perf_fs_ls"] = perf("GET", "/api/v1/fs/ls", query={"uri": root_uri}, n=20)
    results["perf_content_read"] = perf("GET", "/api/v1/content/read", query={"uri": file_uri}, n=20)
    results["perf_search_find"] = perf(
        "POST",
        "/api/v1/search/find",
        body={"query": "OpenViking filesystem", "target_uri": root_uri, "limit": 3},
        n=10,
        timeout=120,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
