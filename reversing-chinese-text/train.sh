#!/bin/bash
# Train SFT model and upload to HuggingFace
# Usage: ./train.sh 100

set -euo pipefail

SIZE=${1:?Usage: ./train.sh <size>}
SESSION_NAME="sft-train-n${SIZE}"

# Fix for Ghostty terminal compatibility with tmux
export TERM=xterm-256color
export WANDB_ENTITY=ivanleomk

# Validate size matches available datasets (from create_dataset.py SFT_SIZES)
VALID_SIZES=(100 500 2500 5000 10000)
if [[ ! " ${VALID_SIZES[*]} " =~ " ${SIZE} " ]]; then
    echo "Error: Invalid size '${SIZE}'. Valid sizes: ${VALID_SIZES[*]}"
    exit 1
fi

# Config
MODEL="Qwen/Qwen3-0.6B"
DATASET="ivanleomk/reverse-chinese-poetry-${SIZE}"
MAX_STEPS=10
HF_REPO="ivanleomk/chinese-reverse-sft-n${SIZE}"

echo "Training on ${DATASET}..."

# Create config
mkdir -p configs/ablation
cat > configs/ablation/sft_n${SIZE}.toml << EOF
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
cat > "${TRAIN_SCRIPT}" << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT

cat >> "${TRAIN_SCRIPT}" << SCRIPT
cd "${PWD}"
echo "Training on ${DATASET}..."
uv run sft @ configs/ablation/sft_n${SIZE}.toml \\
    --wandb.project reverse-text-sft \\
    --wandb.name chinese-reverse-sft-n${SIZE}

echo "Uploading to ${HF_REPO}..."
uv run huggingface-cli upload ${HF_REPO} outputs/weights/step_${MAX_STEPS}

echo "✓ Done: https://huggingface.co/${HF_REPO}"
echo "Press Enter to close this session..."
read
SCRIPT

chmod +x "${TRAIN_SCRIPT}"

# Kill existing session if it exists
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

# Start tmux session
echo "Starting tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "bash ${TRAIN_SCRIPT}; rm ${TRAIN_SCRIPT}"

echo "Training started in tmux session '${SESSION_NAME}'"
echo "Attach with: TERM=xterm-256color tmux attach -t ${SESSION_NAME}"