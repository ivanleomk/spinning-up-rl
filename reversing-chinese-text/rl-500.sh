#!/bin/bash
# Train RL on SFT-500 with configurable parameters
# Usage: ./rl-500.sh [steps] [rollouts] [lr]
# Example: ./rl-500.sh 100 32 1e-6

set -euo pipefail

MAX_STEPS=${1:-20}
ROLLOUTS=${2:-16}
LR=${3:-"3e-6"}

RUN_NAME="rl-500-s${MAX_STEPS}-r${ROLLOUTS}-lr${LR}"

export TERM=xterm-256color
export WANDB_ENTITY=ivanleomk

# Load .env
[[ -f ".env" ]] && source .env
[[ -n "${HF_TOKEN:-}" ]] && hf auth login --token "$HF_TOKEN"

echo "=============================================="
echo "Run: ${RUN_NAME}"
echo "Steps: ${MAX_STEPS} | Rollouts: ${ROLLOUTS} | LR: ${LR}"
echo "=============================================="

mkdir -p configs/rl/ablations
CONFIG="configs/rl/ablations/${RUN_NAME}.toml"

cat > "${CONFIG}" << EOF
max_steps = ${MAX_STEPS}
seq_len = 2048

[model]
name = "ivanleomk/chinese-reverse-sft-n500"

[wandb]
project = "reverse-text-rl-ablations"
name = "${RUN_NAME}"

[orchestrator]
batch_size = 128
rollouts_per_example = ${ROLLOUTS}

[orchestrator.sampling]
max_tokens = 128

[[orchestrator.env]]
id = "reverse-chinese"

[ckpt]

[trainer.optim]
lr = ${LR}

[inference]
EOF

uv run rl @ ./${CONFIG}

hf upload "ivanleomk/${RUN_NAME}" ./outputs/weights/step_${MAX_STEPS}
echo "✓ Done: https://huggingface.co/ivanleomk/${RUN_NAME}"