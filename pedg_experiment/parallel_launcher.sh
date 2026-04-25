#!/bin/bash
# Parallel launcher for PEDG full experiment
# Splits 10 task families across 3 GPUs for ~3x speedup

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../venv/bin/python"

# Partition: GPU1 -> families 0-3, GPU2 -> families 4-6, GPU3 -> families 7-9
# Each process gets 3-4 families with 200 episodes each = 1400-1600 episodes per GPU
# Total: 10 families x 200 eps = 2000 episodes per aligned/conflict = 4000 total episodes
# Wait, full experiment is 10 families x 200 eps x 8 cells x 2 (aligned/conflict) = 32,000?
# Let's recalculate: 10 families x 200 eps x 4 cells (2 info_struct x 2 oversight) = 8000 per condition
# Total: 10 families x 200 eps x 8 cells = 16,000 episodes

echo "=============================================="
echo "PEDG Parallel Experiment Launcher"
echo "=============================================="
echo "GPU1 -> Families 0-3 (4 families)"
echo "GPU2 -> Families 4-6 (3 families)"  
echo "GPU3 -> Families 7-9 (3 families)"
echo "=============================================="

LOG_DIR="/tmp/pedg_parallel"
mkdir -p "$LOG_DIR"

# Clean old processes
pkill -f "main_experiment.*--gpu" 2>/dev/null || true
sleep 2

# Launch 3 parallel processes
# Each uses a different family subset via TASK_FAMILIES env var split
# The experiment uses: families = min(families, len(TASK_FAMILIES))
# Family splitting by passing different 'families' arg won't work (always first N)
# Instead, we use a custom approach: each process generates all tasks but only runs a subset

# Create a temporary task generation file for each process
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

for i in 1 2 3; do
    GPU=$i
    if [ $i -eq 1 ]; then
        FAMILIES_SPLIT="0-3"
    elif [ $i -eq 2 ]; then
        FAMILIES_SPLIT="4-6"
    else
        FAMILIES_SPLIT="7-9"
    fi
    
    # Create a custom run script for this GPU that patches family range
    cat > "$TEMP_DIR/run_gpu${GPU}.py" << PYSCRIPT
import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path("$SCRIPT_DIR")
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "$GPU"

# Monkey-patch generate_all_tasks to only yield certain families
import envs.task_families as tf_module
_orig_generate = tf_module.generate_all_tasks

def patched_generate(seed, families, episodes_per_family):
    all_tasks = _orig_generate(seed=seed, families=families, episodes_per_family=episodes_per_family)
    # Split by family range
    family_start, family_end = $FAMILIES_SPLIT
    filtered = [t for t in all_tasks if family_start <= t.family_idx < family_end]
    print(f"GPU $GPU: Running families {family_start}-{family_end-1} ({len(filtered)} tasks)")
    return filtered

tf_module.generate_all_tasks = patched_generate

# Now run the experiment
from main_experiment import run_full
import yaml

config_path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

results, df = run_full(config, gpu_id=$GPU)
print(f"GPU $GPU COMPLETE: {len(results)} episodes")
PYSCRIPT

    echo "Starting GPU $GPU (families $FAMILIES_SPLIT)..."
    $VENV "$TEMP_DIR/run_gpu${GPU}.py" > "$LOG_DIR/gpu${GPU}.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All 3 processes started. Logs: $LOG_DIR/gpu{1,2,3}.log"
echo "Watch progress: tail -f $LOG_DIR/gpu*.log"
echo ""
echo "To check status: ps aux | grep main_experiment | grep -v grep"
echo "To stop all: pkill -f 'run_gpu.*\.py'"