#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)" || true
export LD_PRELOAD="${TCMALLOC}"

# Write GCS credentials from env var if provided
if [ -n "${GCS_KEY_JSON_B64}" ]; then
    echo "${GCS_KEY_JSON_B64}" | base64 -d > /app/gcs-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
fi

echo "Starting ComfyUI..."
python3 /ComfyUI/main.py --listen --use-sage-attention &

echo "Waiting for ComfyUI to be ready..."
until curl --silent --fail http://127.0.0.1:8188 --output /dev/null; do
    sleep 2
done
echo "ComfyUI is ready"

echo "Starting RunPod handler..."
python3 -u /app/handler.py
