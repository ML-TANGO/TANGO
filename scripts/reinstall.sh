#!/bin/sh
set -eu

# ----------------------------------------
# Detect docker compose command automatically
# ----------------------------------------
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Neither 'docker compose' nor 'docker-compose' is installed."
    exit 1
fi

# ----------------------------------------
# Determine project name
# ----------------------------------------
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tango}"
COMPOSE="docker compose --project-name ${PROJECT_NAME}"

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
echo "==> Removing only images that match ${PROJECT_NAME}_*"
IMAGES="$(docker images --filter="reference=${PROJECT_NAME}_*" -q || true)"
if [ -n "${IMAGES}" ]; then
  docker rmi -f ${IMAGES}
else
  echo "No images matching ${PROJECT_NAME}_*"
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