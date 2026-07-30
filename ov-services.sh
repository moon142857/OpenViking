#!/usr/bin/env bash
# OpenViking 服务启停脚本 (两个后端经 systemd 常驻;前端走 nginx -> 后端 /studio)
# 用法:
#   ov-services.sh start    启动两个后端
#   ov-services.sh stop     停止两个后端
#   ov-services.sh restart  重启
#   ov-services.sh status   查看状态 (含 /health)
set -euo pipefail

SERVICES=(openviking-textbooks openviking-curriculum)

case "${1:-status}" in
  start)
    for s in "${SERVICES[@]}"; do
      sudo systemctl enable --now "$s" >/dev/null 2>&1 || true
      echo "started: $s"
    done
    ;;
  stop)
    for s in "${SERVICES[@]}"; do
      sudo systemctl stop "$s" >/dev/null 2>&1 || true
      echo "stopped: $s"
    done
    ;;
  restart)
    for s in "${SERVICES[@]}"; do
      sudo systemctl restart "$s" >/dev/null 2>&1 || true
      echo "restarted: $s"
    done
    ;;
  status)
    for s in "${SERVICES[@]}"; do
      state=$(systemctl is-active "$s" 2>/dev/null || echo unknown)
      port=$([ "$s" = "openviking-textbooks" ] && echo 1933 || echo 1934)
      name=$([ "$s" = "openviking-textbooks" ] && echo "教材库" || echo "课标库")
      h=$(curl -sS --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'{d.get(\"status\")}/{d.get(\"auth_mode\")}')" 2>/dev/null || echo "无响应")
      printf "  %-26s %-8s  %s :%s->%s\n" "$s" "$state" "$name" "$port" "$h"
    done
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
