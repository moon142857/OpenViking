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
