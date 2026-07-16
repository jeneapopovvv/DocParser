#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

echo ">> Pulling latest images"
docker compose -f "$COMPOSE_FILE" pull

echo ">> Rebuilding service images"
docker compose -f "$COMPOSE_FILE" build

echo ">> Recreating and starting containers"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate

echo ">> Pruning unused images"
docker image prune -f

echo ">> Redeploy complete"
docker compose -f "$COMPOSE_FILE" ps
