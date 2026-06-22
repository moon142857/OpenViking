#!/usr/bin/env python3
"""
Import teaching_kb test data into OpenViking via API.
Creates directories and writes files one by one.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

CONSOLE_URL = "http://127.0.0.1:8020/console/api/v1"
SERVER_URL = "http://127.0.0.1:1933/api/v1"
SOURCE_DIR = Path("/tmp/teaching_kb")
TARGET_BASE = "viking://resources/teaching_kb_test"


async def console_post(client: httpx.AsyncClient, path: str, json_data=None):
    resp = await client.post(f"{CONSOLE_URL}{path}", json=json_data)
    return resp.json()


async def console_get(client: httpx.AsyncClient, path: str):
    resp = await client.get(f"{CONSOLE_URL}{path}")
    return resp.json()


async def server_post(client: httpx.AsyncClient, path: str, json_data=None):
    resp = await client.post(f"{SERVER_URL}{path}", json=json_data)
    return resp.json()


async def server_get(client: httpx.AsyncClient, path: str):
    resp = await client.get(f"{SERVER_URL}{path}")
    return resp.json()


async def mkdir(client: httpx.AsyncClient, uri: str, description: str = ""):
    result = await server_post(client, "/fs/mkdir", {
        "uri": uri,
        "description": description
    })
    return result


async def write_content(client: httpx.AsyncClient, uri: str, content: str):
    # Use wait=False for fast bulk import, then wait for queue separately
    result = await server_post(client, "/content/write", {
        "uri": uri,
        "content": content,
        "mode": "create",
        "wait": False
    })
    return result


async def import_teaching_kb():
    async with httpx.AsyncClient(timeout=300) as client:
        files = sorted(SOURCE_DIR.rglob("*.md"))
        total = len(files)
        print(f"Found {total} files to import")

        # Create root directory
        print(f"Creating root directory: {TARGET_BASE}")
        await mkdir(client, TARGET_BASE, "教学知识库测试数据集")

        imported = 0
        errors = []

        for file_path in files:
            rel_path = file_path.relative_to(SOURCE_DIR)
            target_uri = f"{TARGET_BASE}/{rel_path}"
            parent_uri = target_uri.rsplit("/", 1)[0]

            # Create parent directories if needed
            if parent_uri != TARGET_BASE:
                result = await mkdir(client, parent_uri)
                if result.get("status") == "error" and "already exists" not in str(result.get("error", "")).lower():
                    pass  # Might already exist, ignore

            content = file_path.read_text(encoding="utf-8")

            try:
                result = await write_content(client, target_uri, content)
                if result.get("status") == "ok":
                    imported += 1
                    print(f"  [{imported}/{total}] OK {target_uri}")
                else:
                    err = result.get("error", {}).get("message", str(result))
                    errors.append((str(rel_path), err))
                    print(f"  [{imported}/{total}] FAIL {target_uri}: {err}")
            except Exception as e:
                errors.append((str(rel_path), str(e)))
                print(f"  [{imported}/{total}] ERROR {target_uri}: {e}")

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.5)

        print(f"\nImport complete: {imported}/{total} files imported")
        if errors:
            print(f"Errors ({len(errors)}):")
            for path, err in errors:
                print(f"  {path}: {err}")

        return imported, total, errors


async def wait_for_processing(timeout=600):
    """Wait for all background queue processing to complete."""
    async with httpx.AsyncClient(timeout=300) as client:
        print(f"\nWaiting for queue processing (up to {timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            result = await server_post(client, "/system/wait", {"timeout": 10})
            if result.get("status") == "ok":
                print("Queue processing complete!")
                return True
            await asyncio.sleep(5)
        print("Queue processing timeout")
        return False


async def check_vector_count():
    async with httpx.AsyncClient(timeout=30) as client:
        result = await server_get(client, "/debug/vector/count?uri=viking%3A%2F%2Fresources%2Fteaching_kb_test")
        print(f"\nVector count for teaching_kb:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result


async def main():
    imported, total, errors = await import_teaching_kb()
    if imported > 0:
        await wait_for_processing(timeout=600)
        await check_vector_count()


if __name__ == "__main__":
    asyncio.run(main())
