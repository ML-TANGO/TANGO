#!/bin/bash
set -e
cd "$(dirname "$0")/.."

ARGS=$1
PORT=$2

if [ "$ARGS" = "inference" ] ; then
    echo "Run Inference Server..."
    if [ -n "$PORT" ] ; then
        uvicorn deploy_server:app --port "$PORT"
    fi
        python deploy_server.py
fi
