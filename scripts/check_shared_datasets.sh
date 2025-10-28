#!/usr/bin/env bash
# Usage:
#   scripts/check_shared_datasets.sh <compose_project_name> [output_yaml]
# Example:
#   scripts/check_shared_datasets.sh tango .compose/docker-compose.datasets.yml
#
# 동작:
# - <project>_shared 볼륨의 Mountpoint 하위 /datasets/* 검사
# - 항상 x-vol-* 앵커는 정의하되, services의 volumes에는 "비어있는 데이터셋만" 앵커 추가
# - 아무 것도 비어있지 않으면 services가 비어있게 생성됨(= override 없어도 됨)

set -euo pipefail

PROJECT_NAME="${1:-tango}"
OUT_YAML="${2:-.compose/docker-compose.datasets.yml}"
VOLUME_NAME="${PROJECT_NAME}_shared"

mkdir -p "$(dirname "$OUT_YAML")"

# 볼륨 마운트 경로
MOUNTPOINT="$(docker volume inspect "$VOLUME_NAME" --format '{{ .Mountpoint }}' 2>/dev/null || true)"

# 키와 shared 상대경로
declare -A SHARED_PATHS=(
  [coco]="datasets/coco"
  [coco128]="datasets/coco128"
  [imagenet]="datasets/imagenet"
  [voc]="datasets/VOC"
  [chestxray]="datasets/ChestXRay"
)

DATASET_ORDER=(coco coco128 imagenet voc chestxray)

is_empty_dir() {
  local p="$1"
  # 파일 하나라도 있으면 non-empty로 간주 (디렉토리만 있는 빈 구조는 empty)
  if [[ ! -d "$p" ]]; then
    return 0
  fi
  if find "$p" -type f -mindepth 1 -print -quit | grep -q . ; then
    return 1  # non-empty
  else
    return 0  # empty
  fi
}

NEED_KEYS=()

if [[ -z "$MOUNTPOINT" || ! -d "$MOUNTPOINT" ]]; then
  # shared 볼륨이 아직 없으면 모든 데이터셋을 외부 바인딩 대상으로
  NEED_KEYS=("${DATASET_ORDER[@]}")
else
  for key in "${DATASET_ORDER[@]}"; do
    target="${MOUNTPOINT}/${SHARED_PATHS[$key]}"
    if is_empty_dir "$target"; then
      NEED_KEYS+=("$key")
    fi
  done
fi

# YAML 생성 시작
{
  cat <<'YAML'
# --- Anchors: env 기반 host-dataset bindings ---
# .env 에서 COCODIR, COCO128DIR, IMAGENETDIR, VOCDIR, CHESTXRAYDIR 설정 가능
x-vol-coco: &vol_coco
  type: bind
  source: ${COCODIR:-./autonn/autonn/autonn_core/datasets/coco}
  target: /shared/datasets/coco
  read_only: false

x-vol-coco128: &vol_coco128
  type: bind
  source: ${COCO128DIR:-./autonn/autonn/autonn_core/datasets/coco128}
  target: /shared/datasets/coco128
  read_only: false

x-vol-imagenet: &vol_imagenet
  type: bind
  source: ${IMAGENETDIR:-./autonn/autonn/autonn_core/datasets/imagenet}
  target: /shared/datasets/imagenet
  read_only: false

x-vol-voc: &vol_voc
  type: bind
  source: ${VOCDIR:-./autonn/autonn/autonn_core/datasets/voc}
  target: /shared/datasets/VOC
  read_only: false

x-vol-chestxray: &vol_chestxray
  type: bind
  source: ${CHESTXRAYDIR:-./autonn/autonn/autonn_core/datasets/ChestXRay}
  target: /shared/datasets/ChestXRay
  read_only: false

services:
YAML

  add_block_for_service() {
    local svc="$1"
    local first=1
    for key in "${NEED_KEYS[@]}"; do
      # 필요할 때만 volumes 섹션과 항목 출력
      if [[ $first -eq 1 ]]; then
        echo "  ${svc}:"
        echo "    volumes:"
        first=0
      fi
      case "$key" in
        coco)      echo "      - *vol_coco" ;;
        coco128)   echo "      - *vol_coco128" ;;
        imagenet)  echo "      - *vol_imagenet" ;;
        voc)       echo "      - *vol_voc" ;;
        chestxray) echo "      - *vol_chestxray" ;;
      esac
    done
  }

  # 필요한 데이터셋이 하나라도 있으면 필요한 서비스에만 앵커 참조 추가
  if [[ ${#NEED_KEYS[@]} -gt 0 ]]; then
    add_block_for_service "project_manager"
    echo
    add_block_for_service "autonn"
    echo
    add_block_for_service "autonn_cl"
  fi
} > "$OUT_YAML"

if [[ ${#NEED_KEYS[@]} -gt 0 ]]; then
  echo "overrides_written"
else
  echo "no_overrides"
fi
