#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

NETWORK_VOLUME="/workspace"

# Run user-provided startup overrides
if [ -f "/workspace/additional_params.sh" ]; then
    chmod +x /workspace/additional_params.sh
    echo "Executing additional_params.sh..."
    /workspace/additional_params.sh
fi

if ! which aria2 > /dev/null 2>&1; then
    apt-get update && apt-get install -y aria2
fi

# Build SageAttention in background (cached on network volume after first build)
SAGE_CACHE="$NETWORK_VOLUME/.sage_attention_built"
(
    if [ -f "$SAGE_CACHE" ] && python -c "import sageattention" 2>/dev/null; then
        echo "SageAttention already installed (cached)."
    else
        echo "Building SageAttention..."
        export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
        cd /tmp
        rm -rf SageAttention
        git clone https://github.com/thu-ml/SageAttention.git
        cd SageAttention
        git reset --hard 68de379
        if pip install .; then
            touch "$SAGE_CACHE"
            echo "SageAttention build completed."
        else
            echo "SageAttention build failed. See /tmp/sage_build.log"
        fi
        rm -rf /tmp/SageAttention
    fi
) > /tmp/sage_build.log 2>&1 &
SAGE_PID=$!

# Check if NETWORK_VOLUME exists; if not, use root directory
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "NETWORK_VOLUME '$NETWORK_VOLUME' does not exist. Using '/' instead."
    NETWORK_VOLUME="/"
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi

# Extra model paths for dev mode
USE_EXTRA_MODEL_PATHS=false
if [ "$IS_DEV" = "true" ] && [ "$NETWORK_VOLUME" = "/workspace" ]; then
    mkdir -p /models/diffusion_models
    if [ -d "/workspace/ComfyUI/models/diffusion_models" ]; then
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
    mv /ComfyUI "$COMFYUI_DIR"
fi

# Update ComfyUI only if explicitly requested
if [ "$UPDATE_COMFYUI" = "true" ]; then
    echo "Updating ComfyUI..."
    cd "$COMFYUI_DIR" && git checkout master && git pull
    pip install -r "$COMFYUI_DIR/requirements.txt"
fi

mkdir -p "$CUSTOM_NODES_DIR"

# Clone private custom nodes and workflows in parallel if GITHUB_PAT is set
if [ -n "$GITHUB_PAT" ]; then
    for repo_name in ComfyUI-HMNodes airobust_custom_nodes; do
        target_dir="$CUSTOM_NODES_DIR/$repo_name"
        if [ ! -d "$target_dir" ]; then
            (
                echo "Cloning $repo_name..."
                git clone "https://${GITHUB_PAT}@github.com/nortus-software/${repo_name}.git" "$target_dir" && \
                    [ -f "$target_dir/requirements.txt" ] && pip install -r "$target_dir/requirements.txt"
            ) &
        fi
    done

    # Clone workflow repo
    WORKFLOW_REPO_DIR="/tmp/z_image_first_frame_match"
    rm -rf "$WORKFLOW_REPO_DIR"
    (
        if git clone "https://${GITHUB_PAT}@github.com/nortus-software/z_image_first_frame_match.git" "$WORKFLOW_REPO_DIR" 2>&1; then
            mkdir -p "$WORKFLOW_DIR"
            cp "$WORKFLOW_REPO_DIR"/*.json "$WORKFLOW_DIR/"
            find "$WORKFLOW_REPO_DIR" -maxdepth 1 -name "*.json" | head -n 1 > /tmp/default_workflow_path
        fi
    ) &

    wait
    [ -f /tmp/default_workflow_path ] && DEFAULT_WORKFLOW=$(cat /tmp/default_workflow_path)
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
        if [ "$size_bytes" -lt 10485760 ]; then
            rm -f "$full_path"
        else
            return 0
        fi
    fi

    # Clean up partial downloads
    rm -f "${full_path}.aria2" "$full_path"

    aria2c -x 16 -s 16 -k 1M --continue=true -d "$destination_dir" -o "$destination_file" "$url" &
}

download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/ae.safetensors"

# Download LoRA (requires HF_TOKEN)
if [ ! -f "$LORAS_DIR/linaZ.safetensors" ]; then
    mkdir -p "$LORAS_DIR"
    wget --header="Authorization: Bearer $HF_TOKEN" \
        -O "$LORAS_DIR/linaZ.safetensors" \
        "https://huggingface.co/Maverkk/linaZ/resolve/main/adapter_model-2.safetensors" &
fi

# Wait for all downloads
echo "Waiting for model downloads..."
while pgrep -x "aria2c" > /dev/null || pgrep -x "wget" > /dev/null; do
    sleep 5
done
echo "All models downloaded."

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
else
    sed -i 's/^preview_method = .*/preview_method = auto/' "$CONFIG_FILE"
fi

# Wait for SageAttention build
echo "Waiting for SageAttention build..."
while kill -0 "$SAGE_PID" 2>/dev/null; do
    sleep 10
done

SAGE_FLAG=""
if python -c "import sageattention" 2>/dev/null; then
    SAGE_FLAG="--use-sage-attention"
    echo "SageAttention available."
else
    echo "SageAttention not available, launching without it."
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
fi

echo "Starting ComfyUI..."
nohup $COMFYUI_CMD > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &

until curl --silent --fail http://127.0.0.1:8188/ --output /dev/null; do
    sleep 2
done
echo "ComfyUI is ready"
sleep infinity
