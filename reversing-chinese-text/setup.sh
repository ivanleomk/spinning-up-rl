#!/bin/bash
#
# Setup script for Prime-RL SFT training and evaluation
#
# Usage:
#   ./setup.sh
#
# This script installs:
#   - Prime-RL (cloned from GitHub)
#   - uv (Python package manager)
#   - flash-attn (for fast attention)
#   - vllm (for inference)
#   - wandb (for experiment tracking)
#   - verifiers (for evaluation)

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

# -------------------------------------------------------------------
# 0. Load environment variables
# -------------------------------------------------------------------
if [[ -f ".env" ]]; then
    log_info "Loading .env file..."
    set -a
    source .env
    set +a
else
    log_warn "No .env file found, skipping..."
fi

# -------------------------------------------------------------------
# 1. Install base packages
# -------------------------------------------------------------------
log_info "Installing base packages..."
sudo apt update && sudo apt install -y build-essential curl git tmux htop nvtop

# -------------------------------------------------------------------
# 2. Install uv (if not already installed)
# -------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
else
    log_info "uv already installed"
fi

# Source uv environment
[[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"

# -------------------------------------------------------------------
# 3. Clone Prime-RL
# -------------------------------------------------------------------
if [[ ! -d "prime-rl" ]]; then
    log_info "Cloning prime-rl..."
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git
else
    log_info "prime-rl already exists, pulling latest..."
    cd prime-rl && git pull && cd ..
fi

cd prime-rl

# -------------------------------------------------------------------
# 4. Sync Prime-RL dependencies
# -------------------------------------------------------------------
log_info "Syncing prime-rl dependencies..."
uv sync --all-extras

# -------------------------------------------------------------------
# 5. Install flash-attn
# -------------------------------------------------------------------
log_info "Installing flash-attn..."
uv pip install flash-attn --no-build-isolation

# -------------------------------------------------------------------
# 6. Install vllm (for inference/evaluation)
# -------------------------------------------------------------------
log_info "Installing vllm..."
uv pip install vllm

# -------------------------------------------------------------------
# 7. Install wandb (for experiment tracking)
# -------------------------------------------------------------------
log_info "Installing wandb..."
uv pip install wandb

# -------------------------------------------------------------------
# 8. Install prime CLI tool
# -------------------------------------------------------------------
log_info "Installing prime CLI..."
uv tool install prime

# -------------------------------------------------------------------
# 9. Install the chinese-text-reverse environment (verifiers)
# -------------------------------------------------------------------
log_info "Installing chinese-text-reverse environment..."
prime env install ivanleomk/chinese-text-reverse

# -------------------------------------------------------------------
# 10. Verify installation
# -------------------------------------------------------------------
log_info "Verifying installation..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Check flash-attn
uv run python -c "import flash_attn; print(f'flash-attn: OK')" || log_warn "flash-attn import failed"

# Check vllm
uv run python -c "import vllm; print(f'vllm: OK')" || log_warn "vllm import failed"

# Check verifiers environment
uv run python -c "import chinese_text_reverse; print(f'chinese-text-reverse env: OK')" || log_warn "chinese-text-reverse import failed"


# -------------------------------------------------------------------
# 11. Auto-login to services (if tokens are set)
# -------------------------------------------------------------------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    log_info "Logging into wandb..."
    uv run wandb login "$WANDB_API_KEY"
else
    log_warn "WANDB_API_KEY not set, skipping wandb login"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    log_info "Logging into HuggingFace..."
    uv run huggingface-cli login --token "$HF_TOKEN"
else
    log_warn "HF_TOKEN not set, skipping HuggingFace login"
fi

echo ""
log_info "=============================================="
log_info "Setup complete!"
log_info "=============================================="
echo ""
echo "Next steps:"
echo "  1. Login to wandb:     uv run wandb login"
echo "  2. Login to HuggingFace: uv run huggingface-cli login"
