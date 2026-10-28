#!/usr/bin/env bash
set -euo pipefail

model_cache_available() {
  local model_cache_dir="models/huggingface/hub/models--${1//\//--}/snapshots"
  local snapshot
  for snapshot in "$model_cache_dir"/*; do
    [[ -d "$snapshot" && -f "$snapshot/config.json" ]] || continue
    if compgen -G "$snapshot/*.safetensors" > /dev/null || compgen -G "$snapshot/*.bin" > /dev/null; then
      return 0
    fi
  done
  return 1
}
if model_cache_available "Qwen/Qwen3.5-2B"; then
  export EXTRACTOR_HF_HUB_OFFLINE=1
  echo ">> Using cached extractor model in Hugging Face offline mode"
else
  export EXTRACTOR_HF_HUB_OFFLINE=0
  echo ">> Extractor model cache not found; allowing Hugging Face downloads"
fi
if model_cache_available "Qwen/Qwen3.5-0.8B"; then
  export CLASSIFIER_HF_HUB_OFFLINE=1
  echo ">> Using cached classifier model in Hugging Face offline mode"
else
  export CLASSIFIER_HF_HUB_OFFLINE=0
  echo ">> Classifier model cache not found; allowing Hugging Face downloads"
fi


echo ">> Recreating and starting containers"
docker compose up -d --build

echo ">> Redeploy complete"
docker compose ps
