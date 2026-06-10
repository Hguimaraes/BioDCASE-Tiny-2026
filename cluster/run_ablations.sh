#!/bin/bash
# --
# Run the ablation configs sequentially in the CURRENT session (no sbatch).
# Intended for an interactive GPU allocation, e.g.:
#   salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=03:00:00
#   bash cluster/run_ablations.sh
#
# Each run writes a self-contained record to experiments/runs/<id>/ and a log
# to cluster/logs/. Commit experiments/runs/ and pull back to update the
# dashboard locally.

set -uo pipefail   # NOT -e: one failed run must not abort the whole sweep

# --- environment (edit the dataset path for your cluster) -------------------
export BIODCASE_DATA_ROOT="${BIODCASE_DATA_ROOT:-$SLURM_TMPDIR/BioDCASE2026_TinyML_Development_Dataset}"
export BIODCASE_SKIP_DEPLOYMENT=1     # no docker/ESP-IDF on the node
export BIODCASE_SKIP_QUANTIZATION=1   # no ai-edge; quantize locally
export MPLBACKEND=Agg                 # headless plots (no PyQt6)

source .venv/bin/activate
mkdir -p cluster/logs

# --- sweep definition (edit these two lists) --------------------------------
# all configs in experiments/configs/, or list them explicitly
CONFIGS=(experiments/configs/*.yaml)
SEEDS=(1 2 3)                         # set to (42) for a quick single-seed pass

# ----------------------------------------------------------------------------
echo "git: $(git rev-parse --short HEAD) | data: $BIODCASE_DATA_ROOT"
nvidia-smi || true

declare -a OK=() FAIL=()
for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="$(basename "${cfg%.yaml}")_s${seed}"
    log="cluster/logs/${tag}.log"
    echo ""
    echo "==================================================================="
    echo "RUN: $cfg | seed $seed  ->  $log"
    echo "==================================================================="
    if BIODCASE_CONFIG="$cfg" BIODCASE_SEED="$seed" \
         python biodcase2026_tiny_ml_pytorch.py 2>&1 | tee "$log"; then
      OK+=("$tag")
    else
      FAIL+=("$tag")
      echo "*** FAILED: $tag (see $log)"
    fi
  done
done

echo ""
echo "=== sweep done: ${#OK[@]} ok, ${#FAIL[@]} failed ==="
[ ${#FAIL[@]} -gt 0 ] && printf '  FAILED: %s\n' "${FAIL[@]}"
echo "run records in experiments/runs/ - commit and pull back to update the dashboard"
