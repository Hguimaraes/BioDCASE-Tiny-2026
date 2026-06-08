# BioDCASE TinyML Experiments

This branch uses a stage-gated experiment workflow. Code is implemented and CPU-smoke-tested locally; GPU training runs manually on the remote server.

## Branch

Development branch:

```bash
git switch feature/biome-tinyml
```

## Baseline Preflight

Run this first on every machine:

```bash
python3 -m experiments.run_baseline --mode preflight
```

The command writes a run record under `output/experiments/` and appends to `output/experiments/summary.csv`.

## Local Dependencies

Create a local environment before running inference or dataset evaluation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_pytorch.txt
```

If the `sklearn` package name fails during installation, install `scikit-learn` and rerun the missing packages explicitly.

## Baseline Inference Smoke Test

After installing the PyTorch requirements, run the packaged pretrained model on the submission demo WAV files:

```bash
python3 -m experiments.run_baseline --mode submission-smoke
```

The default config skips embedded flashing/deployment because this workflow currently assumes Docker-only access, not a physical ESP32 board.

## Dataset Evaluation

Point the run at the downloaded BioDCASE dataset with either an environment variable:

```bash
BIODCASE_DATASET_ROOT=/path/to/BioDCASE_2026_Task_3_TinyML_Dataset_V1 \
python3 -m experiments.run_baseline --mode dataset-eval
```

or an explicit CLI override:

```bash
python3 -m experiments.run_baseline \
  --mode dataset-eval \
  --dataset-root /path/to/BioDCASE_2026_Task_3_TinyML_Dataset_V1
```

## Experiment Sequence

1. Reproduce pretrained baseline metrics.
2. Train a modulation-spectrum-input student.
3. Train a spectrogram student with modulation side-channel conditioning.
4. Add Perch 2.0 logit distillation.
5. Add student-only augmentation while keeping teacher inputs unaugmented.

All experiments should produce a YAML run record and a CSV summary row.

## MSS 2D Input Experiment

This experiment keeps the full BioME-style modulation spectrum map instead of
averaging over the acoustic-frequency and modulation-frequency axes. The v2
preprocessing drops the modulation DC bin, clips extreme values, and applies
per-sample z-score normalization.

```text
waveform -> STFT power -> amplitude envelope -> FFT over time -> [1, 513, 150]
```

Fast local checks:

```bash
python3 -m experiments.run_mss2d \
  --mode preflight \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML

python3 -m experiments.run_mss2d \
  --mode feature-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML

python3 -m experiments.run_mss2d \
  --mode model-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

The current feature-smoke result is `[1, 513, 150]`. The current MSS-only
TinyCNN has 62,923 trainable parameters.

Remote training command:

```bash
BIODCASE_DATASET_ROOT=/path/to/biodcase2026_tinyML \
python3 -m experiments.run_mss2d --mode train
```

Training uses `cache_mss2d_v2`, so it will not mix with the baseline mel cache
or the first MSS cache.
Model checkpoints and run records are written under ignored `output/`.

The trainer saves the final model and best checkpoints under the run's model
directory:

```text
output/experiments/mss2d_input/<timestamp>_train/models/MSS2DTinyCNN.pth
output/experiments/mss2d_input/<timestamp>_train/models/MSS2DTinyCNN_best_accuracy.pth
output/experiments/mss2d_input/<timestamp>_train/models/MSS2DTinyCNN_best_loss.pth
output/experiments/mss2d_input/<timestamp>_train/models/checkpoint_summary.yaml
```

Each training run writes a single feedback log:

```text
output/experiments/mss2d_input/<timestamp>_train/train.log
```

Send back that `train.log` plus the sibling `run.yaml` when reporting remote
results. The log includes the command, git commit, host/GPU info, dataset/cache
settings, model config, epoch metrics, final test metrics, and traceback if the
run fails.

## Local HTML Report

Generate a local report from the CSV summary:

```bash
python3 -m experiments.report
```

The report is written to `output/experiments/report.html`. The `output/` directory is ignored, so this report is for local inspection and is not committed to GitHub.

## Remote Training Handoff

Use the same branch and config on the GPU server:

```bash
git fetch origin
git switch feature/biome-tinyml
git pull
python3 -m experiments.run_baseline --mode preflight
```

Set `BIODCASE_DATASET_ROOT` on the remote machine before running dataset evaluation or training. Training artifacts should stay under `output/` unless an experiment config sets a different ignored output root.
