#!/usr/bin/env bash
# Usage:
#   scripts/check_shared_datasets.sh <compose_project_name> [output_yaml]
# Example:
#   scripts/check_shared_datasets.sh tango .compose/docker-compose.datasets.yml
#
# 동작:
# - <project>_shared 볼륨의 /datasets/* 를 "임시 컨테이너"로 마운트해 파일 유무 검사(호스트 권한 이슈 회피)
# - 항상 x-vol-* 앵커는 정의하되, services의 volumes에는 "비어있는(없거나 파일 없는) 데이터셋만" 앵커 추가
# - 아무 것도 비어있지 않으면 services가 비어있게 생성됨(= override 없어도 됨)

set -euo pipefail

DOCKER="${DOCKER:-docker}"

# 프로젝트명 해석 우선순위: 인자 > 환경변수 > 현재 디렉토리명
PROJECT_NAME="${1:-${COMPOSE_PROJECT_NAME:-$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]')}}"
OUT_YAML="${2:-.compose/docker-compose.datasets.yml}"
VOLUME_NAME="${PROJECT_NAME}_shared"

mkdir -p "$(dirname "$OUT_YAML")"

echo ">> PROJECT_NAME resolved to: '${PROJECT_NAME}'"
echo ">> Using Docker volume name: '${VOLUME_NAME}'"

# 키와 shared 상대경로
declare -A SHARED_PATHS=(
  [coco]="datasets/coco"
  [coco128]="datasets/coco128"
  [coco128seg]="datasets/coco128_seg"
  [imagenet]="datasets/imagenet"
  [voc]="datasets/VOC"
  [chestxray]="datasets/ChestXRay"
)

DATASET_ORDER=(coco coco128 coco128seg imagenet voc chestxray)

# 임시 컨테이너로 볼륨 내부 검사: 파일이 하나라도 있으면 true(0), 없으면 false(1)
dir_has_files() {
  # $1 = dataset relative path, e.g., "datasets/coco"
  # -t로 TTY 요구하면 CI에서 실패할 수 있으니 비TTY 실행
  $DOCKER run --rm -v "${VOLUME_NAME}:/mnt:ro" --entrypoint sh alpine:3.20 -lc \
    "test -d \"/mnt/$1\" && find \"/mnt/$1\" -type f -mindepth 1 -print -quit | grep -q ."
}

NEED_KEYS=()

# 1) 볼륨 존재 여부를 상태코드로 판단(출력을 버리고 실패/성공만)
if ! $DOCKER volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
  echo "🔍 Docker volume '${VOLUME_NAME}' not found or not accessible via '${DOCKER}'."
  echo "➡️  Treating as empty: all datasets will be considered for external binding."
  NEED_KEYS=("${DATASET_ORDER[@]}")
else
  echo "🔍 Docker volume '${VOLUME_NAME}' is present. Inspecting contents via helper container..."
  for key in "${DATASET_ORDER[@]}"; do
    rel="${SHARED_PATHS[$key]}"
    if dir_has_files "$rel"; then
      echo "   • ${key}: already has files → skip host binding"
    else
      echo "   • ${key}: missing or empty → will bind from host (.env)"
      NEED_KEYS+=("$key")
    fi
  done
fi

# YAML 생성 시작
{
  cat <<'YAML'
# --- Anchors: env 기반 host-dataset bindings ---
# .env에서 COCODIR, COCO128DIR, COCO128SEGDIR, IMAGENETDIR, VOCDIR, CHESTXRAYDIR 설정 가능
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

x-vol-coco128seg: &vol_coco128seg
  type: bind
  source: ${COCO128SEGDIR:-./autonn_cl/autonn_cl/autonn_cl_core/datasets/coco128_seg}
  target: /shared/datasets/coco128_seg
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
      if [[ $first -eq 1 ]]; then
        echo "  ${svc}:"
        echo "    volumes:"
        first=0
      fi
      case "$key" in
        coco)       echo "      - *vol_coco" ;;
        coco128)    echo "      - *vol_coco128" ;;
        coco128seg) echo "      - *vol_coco128seg" ;;
        imagenet)   echo "      - *vol_imagenet" ;;
        voc)        echo "      - *vol_voc" ;;
        chestxray)  echo "      - *vol_chestxray" ;;
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
