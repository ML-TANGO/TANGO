#!/bin/sh
set -eu

# ----------------------------------------
# Detect docker compose command (v2 > v1)
# ----------------------------------------
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif docker-compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "❌ Neither 'docker compose' nor 'docker-compose' is available."
  exit 1
fi

# ----------------------------------------
# Project name (fallback to "tango")
# If you have .env with COMPOSE_PROJECT_NAME, compose will read it automatically.
# This script also respects an exported env var if set.
# ----------------------------------------
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tango}"
COMPOSE="${COMPOSE_CMD} --project-name ${PROJECT_NAME}"

echo "==> Using compose: ${COMPOSE_CMD}"
echo "==> Project name : ${PROJECT_NAME}"
echo "==> Disk usage before:"
docker system df || true
echo

# ----------------------------------------
# 1) Down only THIS compose project
#    --rmi local     : compose가 빌드한 로컬 이미지만
#    --volumes       : 이 프로젝트의 named volumes만 (bind-mount는 영향 없음)
#    --remove-orphans: 남아있는 고아 컨테이너 정리
# ----------------------------------------
echo "==> Bringing down project '${PROJECT_NAME}' (containers, networks, local images, project volumes)"
${COMPOSE} down --rmi local --volumes --remove-orphans

# ----------------------------------------
# 2) Extra safety cleanup by explicit name filters
#    - Images   : ${PROJECT_NAME}_* 레퍼런스만
#    - Volumes  : 이름에 ${PROJECT_NAME}_ 포함만
#    - Networks : 이름에 ${PROJECT_NAME}_ 포함만
# ----------------------------------------
echo "==> Removing dangling images with reference '${PROJECT_NAME}_*' (if any)"
IMG_IDS="$(docker images --filter="reference=${PROJECT_NAME}_*" -q || true)"
[ -n "${IMG_IDS}" ] && docker rmi -f ${IMG_IDS} || echo "No extra images to remove."

echo "==> Removing leftover volumes named '*${PROJECT_NAME}_*' (if any)"
VOL_IDS="$(docker volume ls -q --filter "name=${PROJECT_NAME}_")"
[ -n "${VOL_IDS}" ] && docker volume rm ${VOL_IDS} || echo "No extra volumes to remove."

echo "==> Removing leftover networks named '*${PROJECT_NAME}_*' (if any)"
NET_IDS="$(docker network ls -q --filter "name=${PROJECT_NAME}_")"
[ -n "${NET_IDS}" ] && docker network rm ${NET_IDS} || echo "No extra networks to remove."

# ----------------------------------------
# 3) Prune only project-specific build cache
#    - 빌더가 없으면 생성(--use로 현재 세션 기본 빌더로 설정)
#    - 캐시 필터:
#        --filter "until=24h"            : 24시간 이전 캐시만
#        --filter "type=source.local"    : 로컬 소스 캐시만
#        --filter "type!=exec.cachemount": 런타임 캐시마운트 제외
#    - --keep-storage로 상한 유지 (10GB)
# ----------------------------------------
# docker builder prune
BUILDER_NAME="${PROJECT_NAME}-builder"

if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  echo "==> creating buildx builder: ${BUILDER_NAME}"
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
else
  echo "==> using existing buildx builder: ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}"
fi

echo "==> buildx prune for builder '${BUILDER_NAME}' (project-scoped)"
docker buildx prune \
  --builder "${BUILDER_NAME}" \
  --filter "until=24h" \
  --filter "type!=exec.cachemount" \
  --keep-storage 10gb \
  -f

echo
echo "==> Disk usage after:"
docker system df || true

echo
echo "✅ Done. Only resources under project '${PROJECT_NAME}' were affected."
echo "   (Global builder cache and non-project volumes/images were NOT touched.)"


