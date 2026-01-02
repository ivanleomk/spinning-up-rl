#!/bin/bash
# Benchmark a HuggingFace model using verifiers + vLLM
# Usage: ./eval.sh <model_name>
# Example: ./eval.sh ivanleomk/chinese-reverse-sft-n100

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

MODEL=${1:?Usage: ./eval.sh <model_name>}
ENV_ID="reverse-chinese"
SAMPLES=1000
VLLM_PORT=8000
MAX_SEQ_LEN=2048

echo "=============================================="
echo "Benchmarking: ${MODEL}"
echo "Environment: ${ENV_ID}"
echo "Samples: ${SAMPLES}"
echo "=============================================="

# Start vLLM server in background
log_info "Starting vLLM server..."
vllm serve "${MODEL}" \
    --port ${VLLM_PORT} \
    --max-model-len ${MAX_SEQ_LEN} &
VLLM_PID=$!

# Cleanup function to kill vLLM on exit
cleanup() {
    log_info "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for vLLM to be ready
log_info "Waiting for vLLM server to start..."
while ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    echo "  Waiting for vLLM..."
    sleep 5
done
log_info "vLLM server ready!"

# Run evaluation
log_info "Running evaluation..."

uv run vf-eval "${ENV_ID}" \
    -m "${MODEL}" \
    -b "http://localhost:${VLLM_PORT}/v1" \
    -s \
    -r 1 \
    -n "${SAMPLES}"

echo ""
log_info "Evaluation complete!"
