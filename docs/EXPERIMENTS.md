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

## MSS 2D V3 Regularized CNN

This experiment reuses `cache_mss2d_v2` and changes only the student model and
training recipe:

```text
MSS v2 features -> wider CNN -> BatchNorm -> Dropout -> AdamW -> early stopping
```

Remote training command:

```bash
python3 -m experiments.run_mss2d \
  --config experiments/configs/mss2d_v3_regularized.yaml \
  --mode train \
  --dataset-root /path/to/biodcase2026_tinyML
```

The v3 model has about 250k trainable parameters. It evaluates the
`best_accuracy` checkpoint rather than the final epoch checkpoint. The immediate
goal is to beat the MSS2D-v2 best validation accuracy of `0.5046`; the next
larger goal is the pretrained baseline accuracy of about `0.5628`.

## Mel + MSS Side-Channel

This experiment keeps the baseline mel feature as the main branch and adds the
v2 modulation-spectrum map as a side-channel branch.

```text
audio -> mel [1, 40, 133] -> mel CNN ----\
                                          concat -> classifier
audio -> MSS [1, 513, 150] -> MSS CNN ---/
```

The cache stores one flat vector per sample:

```text
mel size: 5,320
MSS size: 76,950
combined feature size: 82,270
cache id: cache_mel_mss_v1
```

Fast local checks:

```bash
python3 -m experiments.run_mel_mss \
  --mode feature-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML

python3 -m experiments.run_mel_mss \
  --mode model-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

Remote training command:

```bash
python3 -m experiments.run_mel_mss \
  --mode train \
  --dataset-root /path/to/biodcase2026_tinyML
```

The first side-channel model has 109,707 trainable parameters and evaluates the
`best_accuracy` checkpoint. Send back the `train.log`, `run.yaml`, and
`models/checkpoint_summary.yaml` from the run directory.

## Perch 2.0 Evaluation Scaffold

Perch is not a deployable TinyML student for the ESP board, but it is useful as
a research baseline and possible teacher. This branch keeps Perch work isolated
from the PyTorch/TinyML environment because the Perch stack pulls JAX,
TensorFlow, Apache Beam, and Perch-Hoplite dependencies.

Create a separate environment:

```bash
python3.12 -m venv .venv-perch
source .venv-perch/bin/activate
pip install --upgrade pip
pip install -r requirements_perch.txt
```

On Compute Canada, the package index may only expose patched NumPy wheels such
as `2.1.1+computecanada` and not the upstream `2.0.x` wheels required by
Perch-Hoplite metadata. In that case, use the cluster requirements file and
install Perch-Hoplite without dependency resolution after the rest of the stack
is installed:

```bash
python3.12 -m venv .venv-perch
source .venv-perch/bin/activate
pip install --upgrade pip
pip install -r requirements_perch_cluster.txt
pip install --no-deps git+https://github.com/google-research/perch-hoplite.git@v0.1.1
```

The local Perch repository declares Python `<3.12`, so Python 3.12 is a smoke
test rather than a guaranteed supported path. Run preflight first:

```bash
python3 -m experiments.run_perch \
  --mode preflight \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

Check that the BioDCASE labels have a direct mapping to Perch species labels:

```bash
python3 -m experiments.run_perch \
  --mode label-map \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

If imports pass, run a lightweight Perch-Hoplite preset smoke check:

```bash
python3 -m experiments.run_perch \
  --mode hoplite-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

The planned adaptation is to extract the Perch logits for the 10 bird species
using scientific-name mapping. `Background` is not a species logit, so we will
evaluate it as a complement/open-set score against the target bird classes
before deciding whether direct Perch output is a fair baseline or whether we
should train a small classifier on Perch embeddings.

Local frozen-Perch embedding head run:

```bash
source .venv-perch/bin/activate
python3 -m experiments.run_perch \
  --mode train-head \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML \
  --num-epochs 10
```

The teacher head is trained on frozen Perch embeddings and writes:

```text
models/perch_embedding_head_best.keras
models/perch_embedding_head_final.keras
teacher_soft_labels.npz
```

The `.npz` file contains clean-audio teacher logits and probabilities for the
train and validation splits. Use the best checkpoint for distillation.

For a CPU smoke run, cap each class per split:

```bash
python3 -m experiments.run_perch \
  --mode train-head \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML \
  --num-epochs 10 \
  --max-files-per-class 2
```

This trains only a small dense classifier head on cached frozen Perch
embeddings. It does not fine-tune Perch and is not intended for deployment.

To regenerate soft labels from an existing best teacher head:

```bash
python3 -m experiments.run_perch \
  --mode export-soft-labels \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML \
  --teacher-model-path output/experiments/perch2_eval/<run>_train-head/models/perch_embedding_head_best.keras
```

## Perch Logit Distillation Into Mel + MSS

The first teacher-student experiment keeps the deployable Mel+MSS student
architecture unchanged and changes only the training objective. The student
still receives the flat `[mel, MSS]` feature vector and exports the same
classification path for TinyML deployment.

```text
student_loss =
  1.0 * CE(student_logits, hard_label)
  + 0.5 * KL(student_logits / T, teacher_logits / T) * T^2

T = 2.0
teacher = frozen Perch embedding head exported as teacher_soft_labels.npz
```

The teacher archive is generated by `experiments.run_perch` and is not committed
because it lives under ignored `output/`. Copy the archive to the remote server
or point the run at a local copy with `PERCH_TEACHER_SOFT_LABELS`.

Fast checks:

```bash
PERCH_TEACHER_SOFT_LABELS=output/experiments/perch2_eval/20260608_073215_train-head/teacher_soft_labels.npz \
python3 -m experiments.run_mel_mss_distill \
  --mode preflight \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML

PERCH_TEACHER_SOFT_LABELS=output/experiments/perch2_eval/20260608_073215_train-head/teacher_soft_labels.npz \
python3 -m experiments.run_mel_mss_distill \
  --mode model-smoke \
  --dataset-root /home/hguimaraes/datasets/biodcase2026_tinyML
```

Remote training command:

```bash
PERCH_TEACHER_SOFT_LABELS=/path/to/teacher_soft_labels.npz \
python3 -m experiments.run_mel_mss_distill \
  --mode train \
  --dataset-root /path/to/biodcase2026_tinyML
```

The runner aligns teacher logits to student samples by stable relative dataset
keys such as `Train/Common Chaffinch/file.wav`, so the dataset root may differ
between local and remote machines. The feedback log adds per-epoch hard CE,
soft KL, and total distillation loss metrics.

Experiment ideas to revisit after this run:

1. Sweep `soft_loss_weight` over `0.25`, `0.5`, and `1.0` while keeping
   temperature at `2.0`.
2. Sweep temperature over `1.5`, `2.0`, and `4.0` using the best soft-loss
   weight.
3. Add a tiny SED-style mel branch: preserve time frames, predict framewise
   logits, and aggregate with learned attention instead of global average
   pooling.
4. Fuse MSS as a side-channel into the SED head, either as a global context
   vector injected into each time frame or as a small aligned temporal branch.
5. Add Perch embedding MSE distillation from cached Perch embeddings. This
   branch is training-only and should be discarded before TFLite/ESP export.
6. Try the notebook's stop-gradient split only after the simpler KL and
   embedding-MSE variants are measured, because a very small student may need
   classification gradients in the encoder.

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
