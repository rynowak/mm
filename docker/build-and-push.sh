#!/usr/bin/env bash
# Build + push the custom bufo Ray image (cross-arch: Apple Silicon -> cluster linux/amd64).
#
#   REG=myacr.azurecr.io ./docker/build-and-push.sh          # build + push
#   REG=myacr.azurecr.io NO_PUSH=1 ./docker/build-and-push.sh # local build + verify only
#
# Prereqs: docker + buildx; for a private ACR: `az acr login --name <acr>`.
set -euo pipefail

REG="${REG:?set REG to your registry, e.g. myacr.azurecr.io}"
IMAGE="${REG}/ray-bufo:2.40.0-gpu"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${NO_PUSH:-}" ]; then
  # Local-only build + import smoke (verifies py3.12 dep resolution before involving the cluster).
  docker buildx build --platform linux/amd64 --load -t "$IMAGE" "$HERE"
  echo "Built $IMAGE (loaded locally, not pushed)."
else
  docker buildx build --platform linux/amd64 --push -t "$IMAGE" "$HERE"
  echo "Pushed $IMAGE"
fi
