# BioDCASE-Tiny 2026 — Experiment Plan

Status: agreed 2026-06-09

## 0. Decisions

- **Objective: accuracy-first.** Maximize ACC / macro-AUC while staying
  at or below the baseline envelope (≤111 KB tflite, ≤~330 ms model
  time, ≤~200 KB peak RAM). Compression (Track E) trims at the end.
- **Teacher: Perch** (local `.venv-perch` already exists). BirdNET or an
  ensemble only if Perch distillation underdelivers.
- **Custom features: research-first.** Evaluate on host only; write the
  embedded C kernel only if a custom feature beats tuned-mel by a clear
  margin (≥ +3 pts ACC). Ship baseline mel otherwise.
- **Old `feature/*` branches: keep for now, delete after** each fresh
  track implementation lands (they were authored by a previous agent and
  are not to be built upon).

## 1. Where we stand

Baseline (reproduced locally on validation set, 550 samples):

| Model | ACC | ROC AUC | Size | MACs | ESP32 model time |
|---|---|---|---|---|---|
| Baseline `.pth` (float) | 0.5628 | 0.8931 | 393 KB | 23.3 M | — |
| Baseline `.tflite` (int8) | 0.5701 | 0.8938 | 111 KB | 23.3 M | ~330 ms |

Task: 11-class (10 birds + background), 3 s clips @ 24 kHz, ranked on
ACC, ROC AUC, `.tflite` size, on-device times (feature / model / total),
and peak RAM (baseline: 202 KB).

Note on existing `feature/*` branches (Perch distillation, Mel+MSS, SED,
MobileNetV3, BioME): these were produced by a previous, weaker coding
agent. We do **not** build on them. The ideas they gesture at (foundation
-model distillation, slim architectures, alternative features) are folded
into the tracks below as fresh, clean implementations; the old branches
can be deleted once we've confirmed nothing in them is worth salvaging.

## 2. Lessons from the 2025 technical reports (papers/)

1. **Dhar (SlimCNN / MobileGRU)** — full compression pipeline:
   teacher→student logit distillation (KL + CE, α=0.5, T=3.0, cosine LR),
   magnitude pruning (polynomial 0→35% sparsity), then QAT (int8, small
   LR, light augmentation). 151→13 KB with ~2.5 pt AP loss. Distillation
   occasionally *improved* the student (regularization). Dynamic,
   SNR-aware augmentation (SpecMixup overlay + Gaussian noise, p=0.3,
   intensity gated on an SNR estimate) was central; they disabled feature
   caching to augment per-epoch.
2. **Martin** — domain-informed feature tuning + architectural slimming:
   restrict mel band to the species' range, more mel bins (64), fewer
   filters (24/32), global pooling instead of Flatten. Halved on-device
   compute (27.9→15.2 ms) *and* improved AP (0.958→0.986).
3. **Walter (organizer)** — tiny CNNs all plateau ~90% AP regardless of
   width within limits; aggressively cheap features (16 mels, win 2048,
   hop 1024) cut feature time to 1.6 ms and model to 7 KB. Class
   imbalance was the main failure mode.
4. **Oguamanam (TMU)** — feature extraction can dominate the total
   on-device budget; super-cheap features (spectral flux stats + linear
   SVM = 24 B model) were surprisingly competitive *for the binary task*;
   gammatone filterbanks gave better noise contrast than mel.
5. **Espitia** — audio-domain augmentation (±2 semitone pitch shift +
   white noise) lifted accuracy 90.4→94% on a slimmed MobileNet.

Caveats: all 2025 reports are *binary* detection; 2026 is 11-class, so
expect smaller headroom from ultra-tiny models and bigger gains from
better training signal (augmentation, distillation).

## 3. Experiment tracks

Each track = one feature branch. All code must run on CPU and GPU
(device autodetect, no hardcoded `cuda:0`), locally and on the cluster.

### Track A — Training recipe on the baseline arch (cheap, do first)
Branch: `feature/training-recipe`
- Dynamic augmentation (bypass feature cache for train split):
  SpecAugment-style time/freq masking, mixup/SpecMixup across classes,
  Gaussian noise; optionally waveform-level pitch shift ±2 semitones.
- Cosine LR schedule + warmup, label smoothing, class-balanced sampling,
  best-checkpoint selection on val AUC (exists), more epochs.
- Goal: isolate how much of the gap is training signal vs architecture.
- Expected: +5–10 pts ACC at zero deployment cost. This becomes the new
  floor for every other track.

### Track B — Slim architectures (depthwise-separable)
Branch: `feature/slim-cnn`
- Replace baseline's dense 3×3 convs with MobileNet-style
  depthwise-separable blocks; global average pooling head (no Flatten).
- Width/depth sweep targeting 3 operating points: ~30 KB, ~60 KB,
  ~110 KB (= baseline size) tflite.
- Verify every op is TFLite-Micro/esp-nn friendly (int8 DW-conv is well
  supported and fast on ESP32-S3).

### Track C1 — Perch v2 logit distillation (primary, deployable)
Branch: `feature/perch-distillation`
- **Teacher = frozen Perch v2 encoder + an 11-class head trained on our
  labels.** We build the head on our data rather than read Perch's native
  species logits: it avoids eBird code-mapping mistakes and handles the
  **Background** class natively (Perch has no urban-noise class). Coverage
  check confirmed all 10 competition species exist in Perch v2's
  `inat2024` taxonomy, but we still train our own head.
- Pipeline (in `experiments/perch/`):
  1. `export_embeddings.py` (runs in `.venv-perch`): frozen Perch v2
     (`perch_v2_cpu`, 32 kHz / 5 s window, 1536-d embeddings) over every
     clip, saved keyed by wav stem for cross-env alignment. One-time cost.
  2. `train_teacher_head.py`: linear/MLP head on cached embeddings;
     reports the teacher's own val ACC/AUC = **reference ceiling**; exports
     raw soft logits per clip.
  3. Distill into the tiny CNN student: `α·KL(T) + (1−α)·CE`, sweep
     α ∈ {0.3, 0.5, 0.7}, T ∈ {2, 3, 4}. Student stays a plain,
     fully-deployable CNN; teacher logits are aligned by stem.
- Compose with Track A augmentation (teacher logits from clean audio,
  student sees augmented input — consistency distillation).

### Track C2 — MSAB-FiLM student (BioME-inspired, research-first)
Branch: `feature/msab-film`
- Inject Modulation Spectrogram Average Bands (MSAB) into the student CNN
  via FiLM conditioning (`x' = γ⊙x + β`, with `(γ,β)` from the MSAB context
  vector), porting the BioME idea from Transformer layers to conv blocks.
- MSAB is a single global vector per 3 s clip, so the on-device overhead
  is bounded and computed once — but it is still an extra DSP kernel
  (FFT-along-time + band averaging) on the ESP32. Inference-path FiLM is
  the chosen design; the embedded C kernel only gets written if the
  host-side gain clears **+3 pts ACC** over C1.
- Distill from the same Perch teacher, so C2 isolates the FiLM/MSAB
  contribution on top of C1.

### Track D — Feature extraction budget
Branch: `feature/feature-tuning`
- Stay within the provided mel pipeline (deployable for free); sweep
  n_mels ∈ {32, 40, 64}, stride ∈ {512, 768, 1024}, band limits.
  Feature time is only 3.1 ms vs 330 ms model time, so the win here is
  *input-size reduction* → fewer model MACs, not feature time itself.
- Custom features (e.g. modulation spectrum, PCEN): treat as
  research-only unless we commit to writing the embedded C kernel
  (README warns this is non-trivial). Decide explicitly (see open
  questions).

### Track E — Compression: QAT, pruning, structured slimming
Branch: `feature/qat-compression`
- Current pipeline is post-training quantization. Add QAT (torchao or
  litert-torch is already in the venv) for the final candidate.
- Magnitude pruning 30–50% before QAT (Dhar recipe) — helps size only if
  we also shrink layers; on ESP32 sparse weights don't speed up dense
  kernels, so prefer *structured* slimming guided by pruning saliency.
- Low-rank/channel reduction of the 32→64→128 progression.

### Track F — Temporal pooling / SED heads
Branch: `feature/sed-pooling-head`
- Attention pooling or per-frame logits + max/mean pooling instead of
  GAP; cheap (1×1 convs) and known to help on weak 3 s labels.

## 4. Process & infrastructure

- **Branch per feature**, PR into `main` only after a win is confirmed on
  the standard report; keep `main` = reproducible best-known pipeline.
- **Standard report per run** (extend `model_evaluation.py` /
  experiments runner): ACC, macro AUC, per-class confusion, tflite size,
  MACs, params; store config + git SHA + seed in the run folder.
- **Repro**: fixed seeds, 3 seeds for any result we act on (550-sample
  val set ⇒ ±2–3 pts ACC noise between seeds is plausible).
- **Cluster**: Compute Canada (requirements file already on
  `feature/perch-eval`). Jobs must be config-driven (no interactive
  steps), checkpoint/resume-safe, and device-agnostic:
  `device = 'cuda' if torch.cuda.is_available() else 'cpu'` everywhere,
  `pin_memory`/`num_workers` from config.
- **Deployability gate**: every candidate must convert to int8 tflite and
  pass `submission/submission_test.py`; profile on the Korvo-2 when
  available before locking a submission.

## 5. Suggested order

1. Track A (training recipe) — biggest expected gain per effort.
2. Track B (slim arch) on top of A.
3. Track C (distillation) on top of A+B — likely the top-end result.
4. Track E (QAT) on the final candidate.
5. Tracks D/F in parallel as capacity allows.

## 6. Workflow: local ↔ SLURM cluster

The cluster is SLURM-based; the dataset is already staged there. Code
moves by `git pull`, results come back as committed run records.

1. Develop on a `feature/*` branch locally; smoke-test on CPU
   (`skip_deployment_flag: True`, a few epochs).
2. On the cluster: `git pull`, grab an interactive allocation
   (`salloc --gres=gpu:1 ...`), set/stage `BIODCASE_DATA_ROOT`, then
   `bash cluster/run_ablations.sh` to sweep `experiments/configs/*.yaml`
   across seeds in one session (no sbatch).
3. Every run writes a self-contained record to
   `experiments/runs/<run_id>/` — full resolved config, git SHA + branch,
   seed, host/GPU, per-epoch `metrics.csv`, final metrics, artifact
   sizes. These folders are small and tracked in git: commit them on the
   cluster, pull them back locally.
4. Locally: `python experiments/build_dashboard.py` regenerates
   `docs/experiment_plan.html` (this plan + sortable results table).
5. Environment overrides (so the same config works everywhere):
   `BIODCASE_DATA_ROOT` (dataset path), `BIODCASE_SKIP_DEPLOYMENT=1`
   (no docker/ESP-IDF on cluster nodes).

## 7. Discussion log

- **2026-06-09** — Plan agreed (accuracy-first, Perch teacher,
  research-first custom features). Baseline reproduced locally:
  float 0.5628 ACC / 0.8931 AUC; int8 0.5701 ACC / 0.8938 AUC.
  Track A (`feature/training-recipe`) implemented and smoke-tested:
  feature-space dynamic augmentation (time/freq masking, Gaussian noise,
  mixup), label smoothing, cosine LR with warmup, best-checkpoint
  selection on val macro-AUC, full run logging.
- **2026-06-09 (cont.)** — Track C design finalized after reading the
  BioME paper. Teacher = frozen Perch v2 (`perch_v2_cpu`, 1536-d
  embeddings, runs locally on CPU) + 11-class head trained on our labels.
  Confirmed all 10 species are in Perch v2's taxonomy; Background needs
  our head. Split into C1 (deployable Perch logit distillation) and C2
  (MSAB-FiLM research, inference-path FiLM, +3 pt promotion bar). Built
  `experiments/perch/export_embeddings.py` and `train_teacher_head.py`;
  embedding export validated end-to-end.
- **2026-06-10** — First full C1 run on the cluster (H100 MIG, 120 ep,
  seed 42): val 0.5537 ACC / 0.8841 AUC — **at baseline, marginally below**
  (baseline pth 0.5628 / 0.8931). Distillation produced no lift in this
  configuration. Problem: baseline→C1 changed many knobs at once (recipe +
  distillation), so effects are confounded. Launching the recipe×distill
  2×2 ablation (`experiments/make_ablation_configs.py` → `abl_recipe_only`,
  `abl_distill_no_aug`, `abl_c1_full`) to attribute the effect; added
  `BIODCASE_SEED`/`BIODCASE_RUN_NAME` env overrides for seed sweeps.
  Hypotheses: (a) Track A augmentation over-regularizes the 97k-param CNN;
  (b) distillation needs α/T tuning; (c) the student is too small to absorb
  Perch through mel features (→ motivates Track B slimmer-but-better arch
  and/or feature distillation).
- **2026-06-10 (cont.)** — Ablation 2×2 done (3 seeds each). Verdict:
  **distillation works, augmentation hurts.** Best = pure Perch distillation,
  no augmentation: 0.5974 ACC / 0.9099 AUC (vs baseline 0.5628 / 0.8931;
  +3.5 ACC / +1.7 AUC). Adding Track A augmentation to distillation drops it
  to 0.5647 / 0.8961 — confirms hypothesis (a): teacher soft targets already
  regularize, heavy aug over-regularizes the tiny CNN. New working baseline
  = `abl_distill_no_aug`. Next: (1) α/T sweep on the no-aug base (cheap);
  (2) Track B slimmer-but-stronger student (the teacher gap is still huge,
  0.60 vs 0.89, so architecture is now the main lever). Augmentation, if
  revisited, must be much gentler.
- **2026-06-10 (cont.)** — Track B started (`feature/slim-cnn`). Added
  `SlimCNN`: MobileNet-style depthwise-separable student on the same mel
  input (1×40×133), keeping it drop-in deployable. Default 35k params /
  5.9M MACs (vs baseline 97k / 23.3M; int8 ~74 KB vs 111 KB) — 4× fewer
  MACs, so on-device model time should drop sharply too. Configs:
  `experiments/configs/trackB/{slim_default,slim_wide,slim_deep}.yaml`
  (+ Perch distillation, no aug). Smoke-tested end-to-end incl. int8.
  Cluster run pending; question is whether the DS arch distills better than
  the dense baseline at equal/lower budget.
- **2026-06-10 (cont.)** — Track B result (seed 42): DS arch is an
  efficiency win, not an accuracy one. slim_default 0.5719/0.8991 @76KB
  (−2.5 ACC for −32% size), slim_wide 0.5865/0.9025 @109KB (≈ dense within
  noise), slim_deep 0.5483 (too much mel downsampling). Student plateaus
  ~0.57–0.60 *regardless of architecture* while teacher is 0.89 → the limit
  is the input representation, not the model. Motivates C2.
- **2026-06-10 (cont.)** — Track C2 built (`feature/slim-cnn`): ported MSAB
  (modulation-spectrum average bands) + FiLM from BioME/speechprotolab.
  `ModulationSpectrum` (biodcase_tiny/feature_extraction) → 258-d per-clip
  vector, cached by stem (`experiments/features/export_msab.py`). `FiLM2d` +
  `SlimCNNFiLM` (MSAB standardized + projected to 32-d, FiLM after each DS
  block; 66k params / 5.9M MACs). MSAB ctx threaded through the distillation
  train + FiLM val/test loops. tflite export deferred (research-first; needs
  a 2-input/on-device MSAB kernel only if it wins). Matched pair to isolate
  the effect: `experiments/configs/C2/{c2_film,c2_nofilm}.yaml` (same base
  arch + same Perch distillation, FiLM on/off). Smoke-tested end-to-end.
