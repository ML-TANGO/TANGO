#!/usr/bin/env bash
# Usage:
#   scripts/check_docker_compose.sh [output_yaml]
# Default:
#   .compose/docker-compose.v1.yml
#
# 역할:
# - v1 환경(docker-compose)에서 사용할 override compose를 생성
# - autonn / autonn_cl 서비스에 runtime: nvidia + env만 덮어쓰는 얇은 파일을 만든다.
# - 원본 docker-compose.yml은 건드리지 않고, merge로 적용된다.

set -euo pipefail

OUT_YAML="${1:-.compose/docker-compose.v1.yml}"

mkdir -p "$(dirname "$OUT_YAML")"

cat > "$OUT_YAML" << 'YAML'
services:
  autonn:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

  autonn_cl:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
YAML

echo "v1 override written to ${OUT_YAML}"

