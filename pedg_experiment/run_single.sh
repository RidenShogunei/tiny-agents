#!/bin/bash
# Run PEDG experiment
# Usage: bash run_single.sh --mode smoke --gpu 1

set -e

MODE="${1:-smoke}"
GPU="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== PEDG Experiment Runner ==="
echo "Mode: $MODE, GPU: $GPU"

# Check GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Check vLLM availability
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM already running on port 8000"
else
    echo "Starting vLLM with Qwen3.5-9B on GPU $GPU..."
    
    # Stop any existing vllm on port 8000
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    
    # Start vLLM
    CUDA_VISIBLE_DEVICES=$GPU python -m vllm serve \
        /home/jinxu/.cache/tiny-agents/models/Qwen/Qwen3.5-9B \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --model-name Qwen3.5-9B \
        > /tmp/vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
    
    # Wait for vLLM to be ready
    echo "Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "vLLM is ready!"
            break
        fi
        sleep 5
        echo "  ... waiting ($i/60)"
    done
fi

# Run experiment
cd /home/jinxu/hermes-agent/pedg_experiment

# Create cache dir
mkdir -p results/cache

# Run based on mode
if [ "$MODE" == "smoke" ]; then
    echo "Running smoke test..."
    python main_experiment.py --mode smoke --gpu $GPU
elif [ "$MODE" == "full" ]; then
    echo "Running full experiment..."
    python main_experiment.py --mode full --gpu $GPU
elif [ "$MODE" == "analyze" ]; then
    echo "Running analysis..."
    python main_experiment.py --mode analyze
elif [ "$MODE" == "report" ]; then
    echo "Generating reports..."
    python main_experiment.py --mode report
fi

echo "=== Done ==="
