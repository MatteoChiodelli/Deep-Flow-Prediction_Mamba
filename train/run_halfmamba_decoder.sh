#!/bin/bash
# halfMamba (out1, out2, out2b) + V-Mamba blocks in the high-resolution decoder stages (configs HMdec_* in DfpNet.py):
#   HMdec_d2          + dout2 (64x64, mirror of out1)
#   HMdec_d2_d2b      + dout2, dout2b (32x32)
#   HMdec_d2_d2b_d3   + dout2, dout2b, dout3 (16x16): full mirror of the encoder
# Waits for the attention sweep (run_attention_position.sh) to finish before starting, so the GPU is not shared.
#
# Usage (from the train/ folder):
#   nohup ./run_halfmamba_decoder.sh [run_name] > sweep_halfmamba_decoder.log 2>&1 &

set -o pipefail

SEED=42
RUN_NAME=${1:-decSweep}
MODELS=(HMdec_d2 HMdec_d2_d2b HMdec_d2_d2b_d3)
PYTHON=~/conda/envs/dml/bin/python   # explicit, works even without "conda activate"

cd "$(dirname "$0")"
mkdir -p logs

while pgrep -f run_attention_position.sh > /dev/null; do
    echo "$(date '+%F %T') | attention sweep still running, waiting..."
    sleep 300
done

for model in "${MODELS[@]}"; do
    log="logs/${model}_${RUN_NAME}_s${SEED}.log"
    echo "=== $(date '+%F %T') | Training ${model} | seed ${SEED} ==="
    if "$PYTHON" -u runTrain.py "$model" "$RUN_NAME" "$SEED" 2>&1 | tee "$log"; then
        echo "=== ${model} done ==="
    else
        echo "!!! ${model} failed, see ${log} - continuing with the next model"
    fi
done

echo "=== Sweep finished $(date '+%F %T') ==="
