#!/usr/bin/env bash
set -euo pipefail


echo ">> Recreating and starting containers"
docker compose up -d --build

echo ">> Redeploy complete"
docker compose ps
