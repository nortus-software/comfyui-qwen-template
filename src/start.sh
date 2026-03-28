#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

NETWORK_VOLUME="/workspace"
BOOT_START=$(date +%s)

echo "======================================"
echo "  Starting up...  $(date '+%H:%M:%S')"
echo "======================================"

# Run user-provided startup overrides
if [ -f "/workspace/additional_params.sh" ]; then
    echo "[init] Running additional_params.sh..."
    chmod +x /workspace/additional_params.sh
    /workspace/additional_params.sh
fi

if ! which aria2 > /dev/null 2>&1; then
    echo "[init] Installing aria2..."
    apt-get update && apt-get install -y aria2
fi

# Build SageAttention in background (cached on network volume after first build)
SAGE_CACHE="$NETWORK_VOLUME/.sage_attention_built"
(
    if [ -f "$SAGE_CACHE" ] && python -c "import sageattention" 2>/dev/null; then
        echo "[sage] Already installed (cached)."
    else
        echo "[sage] Building SageAttention (this can take ~5 min on first boot)..."
        export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
        cd /tmp
        rm -rf SageAttention
        git clone https://github.com/thu-ml/SageAttention.git
        cd SageAttention
        git reset --hard 68de379
        if pip install -q .; then
            touch "$SAGE_CACHE"
            echo "[sage] Build completed successfully."
        else
            echo "[sage] Build FAILED. See /tmp/sage_build.log"
        fi
        rm -rf /tmp/SageAttention
    fi
) > /tmp/sage_build.log 2>&1 &
SAGE_PID=$!
echo "[sage] SageAttention build started in background (PID: $SAGE_PID)"

# Check if NETWORK_VOLUME exists; if not, use root directory
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "[init] NETWORK_VOLUME '$NETWORK_VOLUME' does not exist. Using '/' instead."
    NETWORK_VOLUME="/"
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    echo "[init] Network volume found at $NETWORK_VOLUME"
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi
echo "[init] JupyterLab started in background"

# Extra model paths for dev mode
USE_EXTRA_MODEL_PATHS=false
if [ "$IS_DEV" = "true" ] && [ "$NETWORK_VOLUME" = "/workspace" ]; then
    mkdir -p /models/diffusion_models
    if [ -d "/workspace/ComfyUI/models/diffusion_models" ]; then
        echo "[dev] Copying diffusion models to local disk in background..."
        (
            find /workspace/ComfyUI/models/diffusion_models -name "*.safetensors" -type f | while read -r file; do
                cp "$file" "/models/diffusion_models/disk_$(basename "$file")"
            done
        ) > /tmp/model_copy.log 2>&1 &
        USE_EXTRA_MODEL_PATHS=true
    fi
fi

COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$NETWORK_VOLUME/ComfyUI/user/default/workflows"
DIFFUSION_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/diffusion_models"
LORAS_DIR="$NETWORK_VOLUME/ComfyUI/models/loras"
TEXT_ENCODERS_DIR="$NETWORK_VOLUME/ComfyUI/models/text_encoders"
VAE_DIR="$NETWORK_VOLUME/ComfyUI/models/vae"
CUSTOM_NODES_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes"

if [ ! -d "$COMFYUI_DIR" ]; then
    echo "[init] Moving ComfyUI to $COMFYUI_DIR..."
    mv /ComfyUI "$COMFYUI_DIR"
else
    echo "[init] ComfyUI already at $COMFYUI_DIR"
fi

# Update ComfyUI only if explicitly requested
if [ "$UPDATE_COMFYUI" = "true" ]; then
    echo "[init] Updating ComfyUI..."
    cd "$COMFYUI_DIR" && git checkout master && git pull
    pip install -q -r "$COMFYUI_DIR/requirements.txt"
fi

mkdir -p "$CUSTOM_NODES_DIR"

# Clone private custom nodes and workflows in parallel if GITHUB_PAT is set
if [ -n "$GITHUB_PAT" ]; then
    echo "[nodes] GITHUB_PAT detected, cloning private repos in parallel..."
    for repo_name in ComfyUI-HMNodes airobust_custom_nodes; do
        target_dir="$CUSTOM_NODES_DIR/$repo_name"
        if [ ! -d "$target_dir" ]; then
            (
                echo "[nodes] Cloning $repo_name..."
                git clone "https://${GITHUB_PAT}@github.com/nortus-software/${repo_name}.git" "$target_dir" && \
                    [ -f "$target_dir/requirements.txt" ] && pip install -q -r "$target_dir/requirements.txt"
                echo "[nodes] $repo_name done."
            ) &
        else
            echo "[nodes] $repo_name already exists, skipping."
        fi
    done

    # Clone workflow repo
    WORKFLOW_REPO_DIR="/tmp/z_image_first_frame_match"
    rm -rf "$WORKFLOW_REPO_DIR"
    (
        echo "[workflows] Cloning z_image_first_frame_match..."
        if git clone "https://${GITHUB_PAT}@github.com/nortus-software/z_image_first_frame_match.git" "$WORKFLOW_REPO_DIR" 2>&1; then
            mkdir -p "$WORKFLOW_DIR"
            cp "$WORKFLOW_REPO_DIR"/*.json "$WORKFLOW_DIR/"
            find "$WORKFLOW_REPO_DIR" -maxdepth 1 -name "*.json" | head -n 1 > /tmp/default_workflow_path
            echo "[workflows] Workflow repo cloned and copied."
        else
            echo "[workflows] Failed to clone z_image_first_frame_match."
        fi
    ) &

    echo "[nodes] Waiting for all clones to finish..."
    wait
    [ -f /tmp/default_workflow_path ] && DEFAULT_WORKFLOW=$(cat /tmp/default_workflow_path)
    echo "[nodes] All private repos ready."
else
    echo "[nodes] GITHUB_PAT not set, skipping private repos."
fi

# Download models in parallel
download_model() {
    local url="$1"
    local full_path="$2"
    local destination_dir=$(dirname "$full_path")
    local destination_file=$(basename "$full_path")

    mkdir -p "$destination_dir"

    if [ -f "$full_path" ]; then
        local size_bytes=$(stat -c%s "$full_path" 2>/dev/null || stat -f%z "$full_path" 2>/dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))
        if [ "$size_bytes" -lt 10485760 ]; then
            echo "[download] Removing corrupted $destination_file (${size_mb}MB < 10MB)"
            rm -f "$full_path"
        else
            echo "[download] $destination_file already exists (${size_mb}MB), skipping."
            return 0
        fi
    fi

    # Clean up partial downloads
    rm -f "${full_path}.aria2" "$full_path"

    echo "[download] Starting download: $destination_file"
    aria2c -x 16 -s 16 -k 1M --continue=true -d "$destination_dir" -o "$destination_file" "$url" &
}

echo "--------------------------------------"
echo "[download] Downloading models..."
echo "--------------------------------------"
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/ae.safetensors"

# Download LoRA (requires HF_TOKEN)
if [ ! -f "$LORAS_DIR/linaZ.safetensors" ]; then
    mkdir -p "$LORAS_DIR"
    echo "[download] Starting download: linaZ.safetensors (LoRA)"
    wget --header="Authorization: Bearer $HF_TOKEN" \
        -O "$LORAS_DIR/linaZ.safetensors" \
        "https://huggingface.co/Maverkk/linaZ/resolve/main/adapter_model-2.safetensors" &
else
    echo "[download] linaZ.safetensors already exists, skipping."
fi

# Track which files are being downloaded for progress reporting
DOWNLOAD_FILES=(
    "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
    "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    "$VAE_DIR/ae.safetensors"
    "$LORAS_DIR/linaZ.safetensors"
)

get_file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

# Wait for all downloads with per-file progress and ETA
echo "[download] Waiting for downloads to complete..."
declare -A PREV_SIZES
while pgrep -x "aria2c" > /dev/null || pgrep -x "wget" > /dev/null; do
    for f in "${DOWNLOAD_FILES[@]}"; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        current=$(get_file_size "$f")
        current_mb=$((current / 1024 / 1024))
        prev=${PREV_SIZES[$f]:-$current}
        speed_bytes=$(( (current - prev) / 5 ))  # bytes per second (5s interval)
        if [ "$speed_bytes" -gt 0 ]; then
            speed_mb=$(( speed_bytes / 1024 / 1024 ))
            # Check if .aria2 control file exists (download incomplete)
            if [ -f "${f}.aria2" ]; then
                echo "[download] $fname: ${current_mb}MB downloaded (${speed_mb}MB/s)"
            else
                echo "[download] $fname: ${current_mb}MB (finalizing)"
            fi
        else
            if [ -f "${f}.aria2" ]; then
                echo "[download] $fname: ${current_mb}MB downloaded (waiting...)"
            fi
        fi
        PREV_SIZES[$f]=$current
    done
    sleep 5
done
echo "[download] All models downloaded."

# Copy bundled workflows
mkdir -p "$WORKFLOW_DIR"
SOURCE_DIR="/comfyui-qwen-template/workflows"
for file in "$SOURCE_DIR"/*.json; do
    [ -f "$file" ] || continue
    cp -n "$file" "$WORKFLOW_DIR/"
done

grep -qxF "cd $NETWORK_VOLUME" ~/.bashrc || echo "cd $NETWORK_VOLUME" >> ~/.bashrc

# ComfyUI Manager config
CONFIG_FILE="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Manager/config.ini"
mkdir -p "$(dirname "$CONFIG_FILE")"
if [ ! -f "$CONFIG_FILE" ]; then
    cat <<EOL > "$CONFIG_FILE"
[default]
preview_method = auto
git_exe =
use_uv = False
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
component_policy = workflow
update_policy = stable-comfyui
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
security_level = normal
skip_migration_check = False
always_lazy_install = False
network_mode = public
db_mode = cache
EOL
fi

# Wait for SageAttention build
echo "[sage] Waiting for SageAttention build to finish..."
while kill -0 "$SAGE_PID" 2>/dev/null; do
    echo "[sage] Still building..."
    sleep 10
done

SAGE_FLAG=""
if python -c "import sageattention" 2>/dev/null; then
    SAGE_FLAG="--use-sage-attention"
    echo "[sage] SageAttention available, enabling."
else
    echo "[sage] SageAttention not available, launching without it."
fi

# Build ComfyUI launch command
COMFYUI_CMD="python3 $NETWORK_VOLUME/ComfyUI/main.py --listen $SAGE_FLAG"

if [ "$USE_EXTRA_MODEL_PATHS" = "true" ]; then
    COMFYUI_CMD="$COMFYUI_CMD --extra-model-paths-config /comfyui-qwen-template/src/extra_model_paths.yaml"
fi

# Set default workflow if available
if [ -n "$DEFAULT_WORKFLOW" ] && [ -f "$DEFAULT_WORKFLOW" ]; then
    DEFAULT_GRAPH_DIR="$NETWORK_VOLUME/ComfyUI/web/assets"
    mkdir -p "$DEFAULT_GRAPH_DIR"
    cp "$DEFAULT_WORKFLOW" "$DEFAULT_GRAPH_DIR/defaultGraph.json"
    echo "[init] Default workflow set: $(basename "$DEFAULT_WORKFLOW")"
fi

BOOT_END=$(date +%s)
BOOT_DURATION=$(( BOOT_END - BOOT_START ))
echo "======================================"
echo "  Boot completed in ${BOOT_DURATION}s"
echo "  Starting ComfyUI...  $(date '+%H:%M:%S')"
echo "======================================"

nohup $COMFYUI_CMD > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &

until curl --silent --fail http://127.0.0.1:8188/ --output /dev/null; do
    sleep 2
done

echo "======================================"
echo "  ComfyUI is ready!  $(date '+%H:%M:%S')"
echo "======================================"
sleep infinity
