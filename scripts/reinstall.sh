#!/bin/sh
set -eu

# ----------------------------------------
# Load .env (if exists) so COMPOSE_PROJECT_NAME is available
# ----------------------------------------
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

# ----------------------------------------
# Determine project name
# ----------------------------------------
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tango}"

# ----------------------------------------
# Detect docker compose command automatically
# ----------------------------------------
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    IMAGE_PREFIX="${PROJECT_NAME}-"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
    IMAGE_PREFIX="${PROJECT_NAME}_"
else
    echo "❌ Neither 'docker compose' nor 'docker-compose' is installed."
    exit 1
fi
COMPOSE="${COMPOSE_CMD} --project-name ${PROJECT_NAME}"

echo "==> Using compose command: ${COMPOSE_CMD}"
echo "==> Project name: ${PROJECT_NAME}"

# ----------------------------------------
# Stop and remove containers, networks, and local images
# ----------------------------------------
echo "==> Stopping & removing containers for project: ${PROJECT_NAME}"
${COMPOSE} down --rmi local --remove-orphans

# ----------------------------------------
# Remove only images matching project prefix
# ----------------------------------------
echo "==> Removing only images that match ${IMAGE_PREFIX}*"
IMAGES="$(docker images --filter="reference=${IMAGE_PREFIX}*" -q || true)"
if [ -n "${IMAGES}" ]; then
  docker rmi -f ${IMAGES}
else
  echo "No images matching ${IMAGE_PREFIX}*"
fi

# ----------------------------------------
# Clean local bind-mount directories
# ----------------------------------------
if [ -d "labelling/datadb" ]; then
    echo "==> Deleting labelling/datadb to avoid permission error"
  sudo rm -rf -- labelling/datadb
fi

# ----------------------------------------
# Rebuild all docker containers
# ----------------------------------------
echo "==> Building all docker containers for project: ${PROJECT_NAME}"
${COMPOSE} build