#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Set the network volume path
NETWORK_VOLUME="/workspace"

# This is in case there's any special installs or overrides that needs to occur when starting the machine before starting ComfyUI
if [ -f "/workspace/additional_params.sh" ]; then
    chmod +x /workspace/additional_params.sh
    echo "Executing additional_params.sh..."
    /workspace/additional_params.sh
else
    echo "additional_params.sh not found in /workspace. Skipping..."
fi

if ! which aria2 > /dev/null 2>&1; then
    echo "Installing aria2..."
    apt-get update && apt-get install -y aria2
else
    echo "aria2 is already installed"
fi

echo "SageAttention is pre-built in the Docker image"

# Check if NETWORK_VOLUME exists; if not, use root directory instead
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "NETWORK_VOLUME directory '$NETWORK_VOLUME' does not exist. You are NOT using a network volume. Setting NETWORK_VOLUME to '/' (root directory)."
    NETWORK_VOLUME="/"
    echo "NETWORK_VOLUME directory doesn't exist. Starting JupyterLab on root directory..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    echo "NETWORK_VOLUME directory exists. Starting JupyterLab..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi

# Check if NETWORK_VOLUME is /workspace and set up extra model paths (only if IS_DEV is true)
USE_EXTRA_MODEL_PATHS=false
if [ "$IS_DEV" = "true" ] && [ "$NETWORK_VOLUME" = "/workspace" ]; then
    echo "IS_DEV is true and NETWORK_VOLUME is /workspace. Setting up extra model paths..."
    
    # Create /models/diffusion_models directory
    mkdir -p /models/diffusion_models
    
    # Copy all .safetensors files from /workspace/ComfyUI/models/diffusion_models to /models/diffusion_models in background
    if [ -d "/workspace/ComfyUI/models/diffusion_models" ]; then
        echo "Copying .safetensors files from /workspace/ComfyUI/models/diffusion_models to /models/diffusion_models in background..."
        (
            find /workspace/ComfyUI/models/diffusion_models -name "*.safetensors" -type f | while read -r file; do
                filename=$(basename "$file")
                cp "$file" "/models/diffusion_models/disk_${filename}"
            done
            echo "✅ Finished copying .safetensors files to /models/diffusion_models"
        ) > /tmp/model_copy.log 2>&1 &
        USE_EXTRA_MODEL_PATHS=true
    else
        echo "⚠️  Source directory /workspace/ComfyUI/models/diffusion_models does not exist. Skipping copy."
    fi
else
    if [ "$IS_DEV" != "true" ]; then
        echo "IS_DEV is not set to true. Skipping extra model paths setup."
    elif [ "$NETWORK_VOLUME" != "/workspace" ]; then
        echo "NETWORK_VOLUME is not /workspace. Skipping extra model paths setup."
    fi
fi

COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$NETWORK_VOLUME/ComfyUI/user/default/workflows"
MODEL_WHITELIST_DIR="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt"
DIFFUSION_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/diffusion_models"
LORAS_DIR="$NETWORK_VOLUME/ComfyUI/models/loras"
TEXT_ENCODERS_DIR="$NETWORK_VOLUME/ComfyUI/models/text_encoders"
VAE_DIR="$NETWORK_VOLUME/ComfyUI/models/vae"
UPSCALE_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/upscale_models"

if [ ! -d "$COMFYUI_DIR" ]; then
    mv /ComfyUI "$COMFYUI_DIR"
else
    echo "Directory already exists, skipping move."
fi

# Update ComfyUI only if UPDATE_COMFYUI is set
if [ "$UPDATE_COMFYUI" = "true" ]; then
    echo "Updating ComfyUI repository..."
    cd "$COMFYUI_DIR"
    git checkout master
    git pull
    /opt/venv/bin/python3 -m pip install -r "$NETWORK_VOLUME/ComfyUI/requirements.txt"
    echo "✅ ComfyUI updated"
else
    echo "Using pre-built ComfyUI from image (set UPDATE_COMFYUI=true to update)"
fi

CUSTOM_NODES_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

# Clone ComfyUI-HMNodes custom node if GITHUB_PAT is set
if [ -n "$GITHUB_PAT" ]; then
    HMNODES_DIR="$CUSTOM_NODES_DIR/ComfyUI-HMNodes"
    if [ ! -d "$HMNODES_DIR" ]; then
        echo "📥 GITHUB_PAT detected. Cloning ComfyUI-HMNodes custom node..."
        cd "$CUSTOM_NODES_DIR"
        if git clone "https://${GITHUB_PAT}@github.com/nortus-software/ComfyUI-HMNodes.git" 2>&1 | tee /tmp/hmnodes_clone.log; then
            echo "✅ ComfyUI-HMNodes cloned successfully"
        else
            echo "❌ Failed to clone ComfyUI-HMNodes. Error details:"
            cat /tmp/hmnodes_clone.log
        fi
    else
        echo "✅ ComfyUI-HMNodes already exists, skipping clone."
    fi
else
    echo "⏭️  GITHUB_PAT not set. Skipping ComfyUI-HMNodes clone."
fi

# Clone airobust_custom_nodes if GITHUB_PAT is set
if [ -n "$GITHUB_PAT" ]; then
    AIROBUST_DIR="$CUSTOM_NODES_DIR/airobust_custom_nodes"
    if [ ! -d "$AIROBUST_DIR" ]; then
        echo "📥 Cloning airobust_custom_nodes..."
        cd "$CUSTOM_NODES_DIR"
        if git clone "https://${GITHUB_PAT}@github.com/nortus-software/airobust_custom_nodes.git" 2>&1 | tee /tmp/airobust_clone.log; then
            echo "✅ airobust_custom_nodes cloned successfully"
            if [ -f "$AIROBUST_DIR/requirements.txt" ]; then
                pip install -r "$AIROBUST_DIR/requirements.txt"
            fi
        else
            echo "❌ Failed to clone airobust_custom_nodes. Error details:"
            cat /tmp/airobust_clone.log
        fi
    else
        echo "✅ airobust_custom_nodes already exists, skipping clone."
    fi
else
    echo "⏭️  GITHUB_PAT not set. Skipping airobust_custom_nodes clone."
fi

# Clone z_image_first_frame_match workflow repo
if [ -n "$GITHUB_PAT" ]; then
    WORKFLOW_REPO_DIR="/tmp/z_image_first_frame_match"
    rm -rf "$WORKFLOW_REPO_DIR"
    echo "📥 Cloning z_image_first_frame_match workflow repo..."
    if git clone "https://${GITHUB_PAT}@github.com/nortus-software/z_image_first_frame_match.git" "$WORKFLOW_REPO_DIR" 2>&1; then
        mkdir -p "$WORKFLOW_DIR"
        cp "$WORKFLOW_REPO_DIR"/*.json "$WORKFLOW_DIR/"
        # Set the first JSON file as the default workflow
        DEFAULT_WORKFLOW=$(find "$WORKFLOW_REPO_DIR" -maxdepth 1 -name "*.json" | head -n 1)
        echo "✅ Workflow cloned and copied. Default workflow: $DEFAULT_WORKFLOW"
    else
        echo "❌ Failed to clone z_image_first_frame_match repo"
    fi
else
    echo "⏭️  GITHUB_PAT not set. Skipping z_image_first_frame_match workflow clone."
fi

download_model() {
    local url="$1"
    local full_path="$2"

    local destination_dir=$(dirname "$full_path")
    local destination_file=$(basename "$full_path")

    mkdir -p "$destination_dir"

    # Simple corruption check: file < 10MB or .aria2 files
    if [ -f "$full_path" ]; then
        local size_bytes=$(stat -f%z "$full_path" 2>/dev/null || stat -c%s "$full_path" 2>/dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))

        if [ "$size_bytes" -lt 10485760 ]; then  # Less than 10MB
            echo "🗑️  Deleting corrupted file (${size_mb}MB < 10MB): $full_path"
            rm -f "$full_path"
        else
            echo "✅ $destination_file already exists (${size_mb}MB), skipping download."
            return 0
        fi
    fi

    # Check for and remove .aria2 control files
    if [ -f "${full_path}.aria2" ]; then
        echo "🗑️  Deleting .aria2 control file: ${full_path}.aria2"
        rm -f "${full_path}.aria2"
        rm -f "$full_path"  # Also remove any partial file
    fi

    echo "📥 Downloading $destination_file to $destination_dir..."
    aria2c -x 16 -s 16 -k 1M --continue=true -d "$destination_dir" -o "$destination_file" "$url" &

    echo "Download started in background for $destination_file"
}


download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/ae.safetensors"

echo "Finished downloading models!"

# Wait for all aria2c downloads to complete
echo "⏳ Waiting for downloads to complete..."
while pgrep -x "aria2c" > /dev/null; do
    echo "🔽 Downloads still in progress..."
    sleep 5
done

echo "✅ All models downloaded successfully!"

echo "Checking and copying workflow..."
mkdir -p "$WORKFLOW_DIR"

# Ensure the file exists in the current directory before moving it
cd /

SOURCE_DIR="/comfyui-qwen-template/workflows"

# Ensure destination directory exists
mkdir -p "$WORKFLOW_DIR"

# Loop over each file in the source directory
for file in "$SOURCE_DIR"/*; do
    # Skip if it's not a file
    [[ -f "$file" ]] || continue

    dest_file="$WORKFLOW_DIR/$(basename "$file")"

    if [[ -e "$dest_file" ]]; then
        echo "File already exists in destination. Deleting: $file"
        rm -f "$file"
    else
        echo "Moving: $file to $WORKFLOW_DIR"
        mv "$file" "$WORKFLOW_DIR"
    fi
done

# Workspace as main working directory
echo "cd $NETWORK_VOLUME" >> ~/.bashrc


echo "Updating default preview method..."
CONFIG_PATH="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Manager"
CONFIG_FILE="$CONFIG_PATH/config.ini"

# Ensure the directory exists
mkdir -p "$CONFIG_PATH"

# Create the config file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating config.ini..."
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
    echo "config.ini already exists. Updating preview_method..."
    sed -i 's/^preview_method = .*/preview_method = auto/' "$CONFIG_FILE"
fi
echo "Config file setup complete!"
echo "Default preview method updated to 'auto'"

URL="http://127.0.0.1:8188"
echo "Starting ComfyUI"

# Build ComfyUI command with optional flags
COMFYUI_CMD="python3 $NETWORK_VOLUME/ComfyUI/main.py --listen --use-sage-attention"

if [ "$USE_EXTRA_MODEL_PATHS" == "true" ]; then
  COMFYUI_CMD="$COMFYUI_CMD --extra-model-paths-config /comfyui-qwen-template/src/extra_model_paths.yaml"
fi

if [ -n "$DEFAULT_WORKFLOW" ] && [ -f "$DEFAULT_WORKFLOW" ]; then
  # Set as the default workflow that loads when ComfyUI opens
  DEFAULT_GRAPH_DIR="$NETWORK_VOLUME/ComfyUI/web/assets"
  mkdir -p "$DEFAULT_GRAPH_DIR"
  cp "$DEFAULT_WORKFLOW" "$DEFAULT_GRAPH_DIR/defaultGraph.json"
  echo "✅ Default workflow set: $DEFAULT_WORKFLOW"
fi

nohup $COMFYUI_CMD > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &
until curl --silent --fail "$URL" --output /dev/null; do
  echo "🔄  ComfyUI Starting Up... You can view the startup logs here: $NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log"
  sleep 2
done
echo "🚀 ComfyUI is ready"
sleep infinity

