#!/bin/bash

# 停止 FastAPI
if [ -f fastapi.pid ]; then
    kill $(cat fastapi.pid)
    rm fastapi.pid
    echo "FastAPI stopped"
fi

# 停止 frpc
if [ -f frpc.pid ]; then
    kill $(cat frpc.pid)
    rm frpc.pid
    echo "frpc stopped"
fi

# 确保端口被释放
PORT=8002
fuser -k $PORT/tcp 2>/dev/null
echo "Port $PORT cleared"