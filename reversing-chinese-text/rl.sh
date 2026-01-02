#!/bin/bash
# Train RL model on top of SFT checkpoint and upload to HuggingFace
# Usage: ./rl.sh 100
# Run from within the prime-rl directory

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

SIZE=${1:?Usage: ./rl.sh <size>}
SESSION_NAME="rl-train-n${SIZE}"

# Capture the current working directory (should be prime-rl)
WORK_DIR="${PWD}"

# Fix for Ghostty terminal compatibility with tmux
export TERM=xterm-256color
export WANDB_ENTITY=ivanleomk

# Validate size matches available SFT models
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

# RL Config
MAX_STEPS=20
SEQ_LEN=2048
BATCH_SIZE=128
ROLLOUTS_PER_EXAMPLE=16
MAX_TOKENS=128
LR="3e-6"

# Model paths
SFT_MODEL="ivanleomk/chinese-reverse-sft-n${SIZE}"
HF_REPO="ivanleomk/chinese-reverse-rl-n${SIZE}"

# Environment (your verifiers environment)
ENV_ID="reverse-chinese"

echo "=============================================="
echo "RL Training on SFT model: ${SFT_MODEL}"
echo "Environment: ${ENV_ID}"
echo "Max steps: ${MAX_STEPS}"
echo "=============================================="

# Create config directory
mkdir -p configs/rl
CONFIG_FILE="configs/rl/rl_n${SIZE}.toml"

cat > "${CONFIG_FILE}" << EOF
max_steps = ${MAX_STEPS}
seq_len = ${SEQ_LEN}

[model]
name = "${SFT_MODEL}"

[wandb]
project = "reverse-text-rl"
name = "chinese-reverse-rl-n${SIZE}"

[orchestrator]
batch_size = ${BATCH_SIZE}
rollouts_per_example = ${ROLLOUTS_PER_EXAMPLE}

[orchestrator.sampling]
max_tokens = ${MAX_TOKENS}

[[orchestrator.env]]
id = "${ENV_ID}"

[ckpt]

[trainer.optim]
lr = ${LR}

[inference]
EOF

log_info "Config saved to: ${CONFIG_FILE}"
echo ""
cat "${CONFIG_FILE}"
echo ""

# Create training script to run in tmux
TRAIN_SCRIPT=$(mktemp)
cat > "${TRAIN_SCRIPT}" << SCRIPT
#!/bin/bash
set -euo pipefail

cd "${WORK_DIR}"
echo "Starting RL training..."
uv run rl @ ./${CONFIG_FILE}

echo "Uploading to ${HF_REPO}..."
hf upload ${HF_REPO} ./outputs/weights/step_$((MAX_STEPS))

echo "✓ Done: https://huggingface.co/${HF_REPO}"
echo "Press Enter to close this session..."
read
SCRIPT

chmod +x "${TRAIN_SCRIPT}"

bash ${TRAIN_SCRIPT}

# # Kill existing session if it exists
# tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

# # Start tmux session
# echo "Starting tmux session: ${SESSION_NAME}"
# tmux new-session -d -s "${SESSION_NAME}" "bash ${TRAIN_SCRIPT}; rm ${TRAIN_SCRIPT}"

# echo "Training started in tmux session '${SESSION_NAME}'"
# echo "Attach with: TERM=xterm-256color tmux attach -t ${SESSION_NAME}"
