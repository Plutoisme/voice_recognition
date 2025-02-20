#!/bin/bash

# 启动 FastAPI 应用
PORT=8002
echo "Starting FastAPI application on port $PORT..."
nohup python ./infer_fastapi.py --port $PORT > fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI started with PID: $FASTAPI_PID"

# 切换目录并启动 frpc
cd /home/lizhinan/project/voice_recognition/deploy/frp/frp_0.51.3_linux_amd64/
echo "Starting frpc..."
nohup ./frpc -c frpc.ini > frpc.log 2>&1 &
FRPC_PID=$!
echo "frpc started with PID: $FRPC_PID"

# 将进程 ID 保存到文件中，方便后续管理
echo $FASTAPI_PID > fastapi.pid
echo $FRPC_PID > frpc.pid

echo "All services started. Use 'kill \$(cat fastapi.pid)' or 'kill \$(cat frpc.pid)' to stop services"