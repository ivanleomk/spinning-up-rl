#!/bin/bash
# Train SFT model and upload to HuggingFace
# Usage: ./train.sh 100
# Run from within the prime-rl directory

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

SIZE=${1:?Usage: ./train.sh <size>}
SESSION_NAME="sft-train-n${SIZE}"

# Capture the current working directory (should be prime-rl)
WORK_DIR="${PWD}"

# Fix for Ghostty terminal compatibility with tmux
export TERM=xterm-256color
export WANDB_ENTITY=ivanleomk

# Validate size matches available datasets (from create_dataset.py SFT_SIZES)
VALID_SIZES=(100 500 2500 5000 10000)
if [[ ! " ${VALID_SIZES[*]} " =~ " ${SIZE} " ]]; then
    echo "Error: Invalid size '${SIZE}'. Valid sizes: ${VALID_SIZES[*]}"
    exit 1
fi

# Load .env file if it exists (check both current dir and parent)
if [[ -f ".env" ]]; then
    log_info "Loading .env file..."
    set -a
    source .env
    set +a
elif [[ -f "../.env" ]]; then
    log_info "Loading ../.env file..."
    set -a
    source ../.env
    set +a
else
    log_warn "No .env file found, skipping..."
fi

# Login to HuggingFace if HF_TOKEN is set
if [[ -n "${HF_TOKEN:-}" ]]; then
    log_info "Logging into HuggingFace..."
    hf auth login --token "$HF_TOKEN"
else
    log_warn "HF_TOKEN not set, skipping HuggingFace login..."
fi

# Add after SIZE validation
EPOCHS=1
BATCH_SIZE=32

# Calculate max_steps: (n_examples * epochs) / batch_size
MAX_STEPS=$(( (SIZE * EPOCHS) / BATCH_SIZE ))

# Ensure at least 1 step for tiny datasets
if [ ${MAX_STEPS} -lt 1 ]; then
    MAX_STEPS=1
fi

# Config
MODEL="Qwen/Qwen3-0.6B"
DATASET="ivanleomk/reverse-chinese-poetry-${SIZE}"
HF_REPO="ivanleomk/chinese-reverse-sft-n${SIZE}"

echo "Training on ${DATASET}..."
echo "Dataset size: ${SIZE}, Epochs: ${EPOCHS}, Max steps: ${MAX_STEPS}"

# Create config
mkdir -p ./configs/ablation
cat > ./configs/ablation/sft_n${SIZE}.toml << EOF
max_steps = ${MAX_STEPS}

[ckpt]

[model]
name = "${MODEL}"

[data]
name = "${DATASET}"
seq_len = 4096
batch_size = 32

[optim]
lr = 2e-5
EOF

# Create training script to run in tmux
TRAIN_SCRIPT=$(mktemp)
cat > "${TRAIN_SCRIPT}" << SCRIPT
#!/bin/bash
set -euo pipefail

cd "${WORK_DIR}"
echo "Training on ${DATASET}..."
uv run sft @ ./configs/ablation/sft_n${SIZE}.toml \\
    --wandb.project reverse-text-sft \\
    --wandb.name chinese-reverse-sft-n${SIZE}

echo "Uploading to ${HF_REPO}..."
hf upload ${HF_REPO} ./outputs/weights/step_${MAX_STEPS}

echo "✓ Done: https://huggingface.co/${HF_REPO}"
echo "Press Enter to close this session..."
read
SCRIPT

chmod +x "${TRAIN_SCRIPT}"

bash ${TRAIN_SCRIPT}

# Kill existing session if it exists
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

# Start tmux session
echo "Starting tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "bash ${TRAIN_SCRIPT}; rm ${TRAIN_SCRIPT}"

echo "Training started in tmux session '${SESSION_NAME}'"
echo "Attach with: TERM=xterm-256color tmux attach -t ${SESSION_NAME}"
