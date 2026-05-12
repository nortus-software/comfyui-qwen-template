#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

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
        git reset --hard 5a6f53c7
        if pip install --no-build-isolation -q .; then
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
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ > /tmp/jupyter.log 2>&1 &
else
    echo "[init] Network volume found at $NETWORK_VOLUME"
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace > /tmp/jupyter.log 2>&1 &
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
    echo "[init] Cloning ComfyUI to $COMFYUI_DIR..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    pip install -q -r "$COMFYUI_DIR/requirements.txt"
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
CLONE_PIDS=()

# Public custom nodes — deps are pre-installed in /opt/venv at image build
PUBLIC_NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager.git"
    "https://github.com/kijai/ComfyUI-KJNodes.git"
    "https://github.com/rgthree/rgthree-comfy.git"
    "https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git"
    "https://github.com/ClownsharkBatwing/RES4LYF.git"
    "https://github.com/cubiq/ComfyUI_essentials.git"
    "https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git"
    "https://github.com/M1kep/ComfyLiterals.git"
    "https://github.com/city96/ComfyUI-GGUF.git"
    "https://github.com/crystian/ComfyUI-Crystools.git"
)
for repo in "${PUBLIC_NODES[@]}"; do
    repo_dir=$(basename "$repo" .git)
    target_dir="$CUSTOM_NODES_DIR/$repo_dir"
    if [ ! -d "$target_dir" ]; then
        (
            echo "[nodes] Cloning $repo_dir..."
            git clone --depth 1 "$repo" "$target_dir" 2>&1 | tail -n 1
            echo "[nodes] $repo_dir done."
        ) &
        CLONE_PIDS+=($!)
    fi
done

# Private custom nodes and workflow repo (require GITHUB_PAT)
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
            CLONE_PIDS+=($!)
        else
            echo "[nodes] $repo_name already exists, skipping."
        fi
    done

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
    CLONE_PIDS+=($!)
else
    echo "[nodes] GITHUB_PAT not set, skipping private repos."
fi

echo "[nodes] Waiting for clones to finish..."
for pid in "${CLONE_PIDS[@]}"; do
    wait "$pid"
done
[ -f /tmp/default_workflow_path ] && DEFAULT_WORKFLOW=$(cat /tmp/default_workflow_path)
echo "[nodes] All custom nodes ready."

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
    aria2c -x 4 -s 4 -k 1M --continue=true --console-log-level=warn --summary-interval=0 \
        -d "$destination_dir" -o "$destination_file" "$url" &
}

echo "--------------------------------------"
echo "[download] Downloading models..."
echo "--------------------------------------"
# Z-Image stack (existing workflow_first_frame_image)
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/ae.safetensors"

# Qwen-Image stack (new workflow_qwen_i2i_minimal / _faithful)
download_model "https://huggingface.co/city96/Qwen-Image-gguf/resolve/main/qwen-image-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/qwen-image-Q8_0.gguf"
download_model "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q8_K_XL.gguf" "$TEXT_ENCODERS_DIR/Qwen2.5-VL-7B-Instruct-UD-Q8_K_XL.gguf"
# Vision projector (mmproj) for the VL encoder — required for Qwen-Image-Edit workflows.
# Filename prefix must match the encoder so ComfyUI-GGUF auto-pairs them.
download_model "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-F16.gguf" "$TEXT_ENCODERS_DIR/Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf"
download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"

# Qwen LoRAs (public)
download_model "https://huggingface.co/BauDevs/Lora_qwen_styles/resolve/main/1GIRL_QWEN_V3.safetensors" "$LORAS_DIR/1girl-qwen_v3.safetensors"
download_model "https://huggingface.co/BauDevs/Lora_qwen_styles/resolve/main/nicegirls_alternative.safetensors" "$LORAS_DIR/NiceGirls_qwen.safetensors"
download_model "https://huggingface.co/Danrisi/Qwen-image_SamsungCam_UltraReal/resolve/main/Samsung.safetensors" "$LORAS_DIR/samsungcam_qwen.safetensors"

# Character LoRAs (require HF_TOKEN)
download_lora_with_token() {
    local dest="$1"
    local url="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "[download] $label already exists, skipping."
    else
        mkdir -p "$(dirname "$dest")"
        echo "[download] Starting download: $label"
        aria2c -x 4 -s 4 -k 1M --continue=true --console-log-level=warn --summary-interval=0 \
            --header="Authorization: Bearer $HF_TOKEN" \
            -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url" &
    fi
}
download_lora_with_token "$LORAS_DIR/linaZ.safetensors" \
    "https://huggingface.co/Maverkk/linaZ/resolve/main/adapter_model-2.safetensors" \
    "linaZ.safetensors (Z-Image character LoRA)"
download_lora_with_token "$LORAS_DIR/lina_qwen.safetensors" \
    "https://huggingface.co/Maverkk/loralina/resolve/main/adapter_model.safetensors" \
    "lina_qwen.safetensors (Qwen-Image character LoRA)"

# Track which files are being downloaded for progress reporting
DOWNLOAD_FILES=(
    "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
    "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    "$VAE_DIR/ae.safetensors"
    "$LORAS_DIR/linaZ.safetensors"
    "$DIFFUSION_MODELS_DIR/qwen-image-Q8_0.gguf"
    "$TEXT_ENCODERS_DIR/Qwen2.5-VL-7B-Instruct-UD-Q8_K_XL.gguf"
    "$TEXT_ENCODERS_DIR/Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf"
    "$VAE_DIR/qwen_image_vae.safetensors"
    "$LORAS_DIR/1girl-qwen_v3.safetensors"
    "$LORAS_DIR/NiceGirls_qwen.safetensors"
    "$LORAS_DIR/samsungcam_qwen.safetensors"
    "$LORAS_DIR/lina_qwen.safetensors"
)

get_file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

# Wait for all downloads with per-file progress and ETA
echo "[download] Waiting for downloads to complete..."
declare -A PREV_SIZES
while pgrep -x "aria2c" > /dev/null; do
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
# COMFY_VERBOSE=true (default in dev / when IS_DEV=true) enables --verbose DEBUG
# and live preview, useful for per-node timing in the captured nohup log.
if [ "${COMFY_VERBOSE:-${IS_DEV:-false}}" = "true" ]; then
    VERBOSE_FLAGS="--verbose DEBUG --preview-method auto"
else
    VERBOSE_FLAGS=""
fi

COMFYUI_CMD="python3 $NETWORK_VOLUME/ComfyUI/main.py --listen $SAGE_FLAG $VERBOSE_FLAGS"

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
