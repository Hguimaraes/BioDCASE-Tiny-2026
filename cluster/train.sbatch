#!/bin/bash
# --
# SLURM job template for BioDCASE-Tiny 2026 training runs
#
# usage (from repo root on the cluster):
#   sbatch cluster/train.sbatch
#
# adjust account / partition / resources to your cluster allocation.

#SBATCH --job-name=biodcase-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x_%j.out
# #SBATCH --account=<your-account>

set -euo pipefail

mkdir -p cluster/logs

# environment - adjust to your cluster setup (module names differ per site)
# module load python/3.12
source .venv/bin/activate

# dataset location on the cluster (overrides config.yaml datamodule.dataset.root_path)
export BIODCASE_DATA_ROOT="${BIODCASE_DATA_ROOT:-$HOME/datasets/biodcase2026_tinyML/BioDCASE2026_TinyML_Development_Dataset}"

# no docker on cluster nodes -> training + quantization + host eval only
export BIODCASE_SKIP_DEPLOYMENT=1

echo "job: ${SLURM_JOB_ID:-local} | host: $(hostname) | git: $(git rev-parse --short HEAD)"
nvidia-smi || true

python biodcase2026_tiny_ml_pytorch.py

echo "done - run records in experiments/runs/ (commit them and pull back locally)"
