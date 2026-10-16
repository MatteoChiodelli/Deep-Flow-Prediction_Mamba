#!/bin/bash
# Position sweep: 6 runs, each one adds a single self-attention block to the baseline,
# moving it from out1 to out5 (configs A_out1 ... A_out5 in DfpNet.py), seed fixed to 42.
#
# Usage (from the train/ folder, with the dml env active):
#   ./run_attention_position.sh [run_name]
# Example:
#   nohup ./run_attention_position.sh posSweep > sweep_attention.log 2>&1 &

set -o pipefail

SEED=42
RUN_NAME=${1:-posSweep}
STAGES=(out1 out2 out2b out3 out4 out5)

cd "$(dirname "$0")"
mkdir -p logs

for stage in "${STAGES[@]}"; do
    model="A_${stage}"
    log="logs/${model}_${RUN_NAME}_s${SEED}.log"
    echo "=== $(date '+%F %T') | Training ${model} (Attention after ${stage}) | seed ${SEED} ==="
    if python -u runTrain.py "$model" "$RUN_NAME" "$SEED" 2>&1 | tee "$log"; then
        echo "=== ${model} done ==="
    else
        echo "!!! ${model} failed, see ${log} - continuing with the next stage"
    fi
done

echo "=== Sweep finished $(date '+%F %T') ==="
