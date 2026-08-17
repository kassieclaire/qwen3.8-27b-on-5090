#!/usr/bin/env bash
# One-shot launcher: downloads weights if missing, builds the image, serves on :30000.
# Env overrides: STEPS=4 TOPK=2 DRAFT=6 (MTP depth) | MFS=0.95 MMC=16 MRR=2 MTT=280000
set -euo pipefail
cd "$(dirname "$0")"
if ! docker volume inspect qwen38-models >/dev/null 2>&1 || \
   [ -z "$(docker volume ls -q -f name=qwen38-models)" ]; then
  echo "[serve] fetching RadixArk/Qwen3.8-27B-NVFP4 (21GB, one time)..."
  docker run --rm -v qwen38-models:/models python:3.12-slim bash -lc \
    'pip -q install huggingface_hub && python -c "from huggingface_hub import snapshot_download; snapshot_download(\"RadixArk/Qwen3.8-27B-NVFP4\", local_dir=\"/models/Qwen3.8-27B-NVFP4\")"'
fi
docker compose up -d --build
echo "[serve] up: http://localhost:30000 (logs: docker compose logs -f server)"
