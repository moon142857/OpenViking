#!/bin/bash
# 启动 Qwen3 MLX Embedding + Reranker 服务
# 监听 0.0.0.0:11436

set -e

cd "$(dirname "$0")/.."

source ~/mlx-env/bin/activate

# Use HuggingFace mirror for mainland China
export HF_ENDPOINT="https://hf-mirror.com"

LOG_FILE="/tmp/qwen3_mlx_server.log"
PORT=11436

# 检查端口是否已被占用
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. MLX server may already be running."
    echo "Check: lsof -i :$PORT"
    exit 1
fi

echo "Starting Qwen3 MLX server on port $PORT..."
echo "Log: $LOG_FILE"

nohup python "${HOME}/repo/qwen3_embedding/qwen3_mlx_server.py" \
    > "$LOG_FILE" 2>&1 &

for i in {1..30}; do
    if curl -s http://127.0.0.1:"$PORT"/health > /dev/null 2>&1; then
        echo "MLX server started successfully."
        exit 0
    fi
    sleep 1
done
echo "Failed to start MLX server. Check log: $LOG_FILE"
exit 1
