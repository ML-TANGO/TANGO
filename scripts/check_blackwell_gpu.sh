#!/usr/bin/env bash

set -e

CU130_FILE=".compose/docker-compose.cu130.yml"

# -------------------------------
# GPU 이름 감지
# -------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)

if [[ -z "$GPU_NAME" ]]; then
    echo "[INFO] No NVIDIA GPU detected or nvidia-smi unavailable. Skipping cu130 override."
    rm -f "$CU130_FILE"
    exit 0
fi

# Blackwell 계열 감지 (B100/B200/RTX 50xx/GB코드 등)
if echo "$GPU_NAME" | grep -Ei 'Blackwell|B100|B200|RTX *50[0-9]0|GB[0-9]{3}' >/dev/null; then
    IS_BLACKWELL=1
else
    IS_BLACKWELL=0
fi

# -------------------------------
# 드라이버 버전 major 추출
# -------------------------------
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | sed 's/[^0-9].*$//')

# -------------------------------
# 로직 본문
# -------------------------------
if [[ "$IS_BLACKWELL" -eq 1 ]]; then
    echo "[INFO] Detected Blackwell GPU: $GPU_NAME"
    echo "[INFO] NVIDIA Driver major version: $DRIVER_MAJOR"

    if [[ "$DRIVER_MAJOR" -ge 580 ]]; then
        echo "[INFO] Driver is compatible with CUDA 13.0. Generating cu130 override."

        mkdir -p .compose

        cat > "$CU130_FILE" <<'EOF'
version: "3.9"

services:
  autonn:
    build:
      dockerfile: Dockerfile.cu130

  autonn_cl:
    build:
      dockerfile: Dockerfile.cu130
EOF

        echo "[INFO] Created $CU130_FILE"
        exit 0
    else
        echo "[WARN] Blackwell GPU detected BUT driver is too old (<580)"
        echo "[WARN] Please upgrade NVIDIA driver to 580+ for CUDA 13.0 support."
        echo "[WARN] Using default Dockerfile (NOT using Dockerfile.cu130)."

        rm -f "$CU130_FILE"
        exit 0
    fi
else
    echo "[INFO] Non-Blackwell GPU: $GPU_NAME"
    echo "[INFO] Using default Dockerfile."

    rm -f "$CU130_FILE"
    exit 0
fi

