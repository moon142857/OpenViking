#!/bin/bash
# 启动 OpenViking HTTP Server（启用 Bot 模式，自动拉起 Vikingbot gateway）

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

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
