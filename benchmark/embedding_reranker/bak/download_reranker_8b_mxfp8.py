#!/usr/bin/env python3
"""Resume download mlx-community/Qwen3-Reranker-8B-mxfp8 with progress."""

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print("Resuming download: mlx-community/Qwen3-Reranker-8B-mxfp8")
    local_dir = snapshot_download(
        repo_id="mlx-community/Qwen3-Reranker-8B-mxfp8",
        resume_download=True,
    )
    print("Downloaded to:", local_dir)
