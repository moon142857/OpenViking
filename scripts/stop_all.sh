#!/bin/bash
# 停止 OpenViking 和 Qwen3 MLX 服务

echo "Stopping OpenViking server (port 1933)..."
pkill -f "openviking-server --host 127.0.0.1 --port 1933" 2>/dev/null || true

echo "Stopping Qwen3 MLX server (port 11436)..."
pkill -f "qwen3_mlx_server.py" 2>/dev/null || true

sleep 1

if lsof -i :1933 > /dev/null 2>&1; then
    echo "Warning: port 1933 still in use"
else
    echo "Port 1933 is free."
fi

if lsof -i :11436 > /dev/null 2>&1; then
    echo "Warning: port 11436 still in use"
else
    echo "Port 11436 is free."
fi
