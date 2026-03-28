#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)" || true
export LD_PRELOAD="${TCMALLOC}"

echo "Starting ComfyUI..."
python3 /workspace/ComfyUI/main.py --listen --use-sage-attention &

echo "Waiting for ComfyUI to be ready..."
until curl --silent --fail http://127.0.0.1:8188 --output /dev/null; do
    sleep 2
done
echo "ComfyUI is ready"

echo "Starting RunPod handler..."
python3 -u /app/handler.py
