# BioDCASE-Tiny 2026 — Experiment Plan & Results

Status: agreed 2026-06-09 · last updated 2026-06-16

## Current status (TL;DR)

- **Best model so far: Track D2 — `Baseline` CNN + PCEN-mel front-end + Perch
  logit **and embedding** distillation → 0.6952 ACC / 0.9402 AUC** (3 seeds),
  up from the provided baseline 0.5628 / 0.8931 (**+13.2 ACC**) and from
  PCEN-logit-distill 0.6497 / 0.9296 (**+4.5 ACC / +1.1 AUC**). Same deployable
  97k-param / 23 M-MAC Baseline (float tflite matches the `.pth` exactly) — only
  the *training signal* changes: the student also regresses Perch's full 1536-d
  embedding (not just the 11 logits).
- **Confirmed at 3 seeds (2026-06-16) and the gain is attributed.** A 3-condition
  ablation isolates the two new levers:
  - **Embedding distillation is the driver** — +2.25 ACC / +0.88 AUC over the
    EMA-only arm (embed 0.6916 vs EMA-only 0.6691).
  - **Weight EMA is ~neutral** once embedding distillation is on: +0.36 ACC /
    −0.05 AUC (full 0.6952 vs embed-no-EMA 0.6916), inside seed noise and it
    *adds* variance (±0.012 vs ±0.006). Keep it as a free small ACC bump or drop
    it for simplicity — the embedding loss is what matters.
- **Two levers move the needle, nothing else does:**
  1. **The front-end** — PCEN-mel ≫ log-mel (+5.2 ACC). This was the real
     bottleneck; the student had plateaued on the input representation.
  2. **The teacher** — Perch logit distillation (+3.5 ACC over plain CE, and
     it lifted *every* model we tried, including a generic EfficientNet +10.7).
- **Architecture changes don't help.** Slim depthwise CNN, MSAB-FiLM, GRU/
  BiGRU recurrence, and an ImageNet EfficientNet backbone all matched-or-lost
  vs the plain Baseline. The model was never the limiting factor.
- **What's next (critical path):** prove the PCEN gain **survives int8 on the
  device** (fixed-point PCEN), then a small distill α/T sweep + compression for
  the final submission. See [What's next](#what-s-next).

## Experiment scoreboard

Validation set (549 samples). "Δ ACC" is vs the *prior* best (PCEN-distill
0.6497). Every row that we acted on is 3 seeds unless noted.

| Experiment | Idea | ACC / AUC | Δ ACC | Verdict |
|---|---|---|---|---|
| Provided baseline | log-mel CNN, no distill | 0.5628 / 0.8931 | −8.7 | starting point |
| Track A: augmentation | SpecAugment + mixup + noise | 0.5647 / 0.8961 | −8.5 | ❌ over-regularizes the tiny CNN |
| Track C1: Perch distillation | KL+CE from Perch (log-mel) | 0.5974 / 0.9099 | −5.2 | ✅ +3.5 over baseline |
| Track B: slim DS-CNN | depthwise-separable, ~76 KB | 0.5865 / 0.9025 | −6.3 | ◐ efficiency only, not accuracy |
| Track C2: MSAB-FiLM | BioME modulation feats + FiLM | 0.5865 / 0.9005 | −6.3 | ❌ = param-matched plain CNN |
| Track D: PCEN-mel (40) | Perch-like front-end | 0.6497 / 0.9296 | prior best | ⭐ +5.2 over log-mel (3 seeds) |
| **Track D2: embed distill + EMA** | regress Perch's 1536-d emb + weight EMA | **0.6952 / 0.9402** | **+4.5** | ⭐ **project best (3 seeds), +4.5 / +1.1** |
| Track D2 abl: embed, no EMA | embedding distill only | 0.6916 / 0.9407 | +4.2 | ✅ ties the bundle (EMA ~neutral) |
| Track D2 abl: EMA only, no embed | weight EMA only | 0.6691 / 0.9319 | +1.9 | ◐ embed is the real lever |
| Track D: PCEN 80-mel | double the mel resolution | 0.610 / 0.907 | −3.9 | ❌ worse (2 seeds) |
| Track F: uni-GRU | recurrence over time | 0.6333 / 0.9178 | −1.6 | ❌ |
| Track F: bi-GRU | bidirectional recurrence | 0.6157 / 0.9183 | −3.4 | ❌ |
| EfficientNet, frozen probe | ImageNet feats + MLP | 0.5738 / 0.8859 | −7.6 | probe only |
| EfficientNet, CE fine-tune | unfrozen, plain CE | 0.5483 / 0.8787 | −10.1 | ❌ overfits |
| EfficientNet, Perch-KD | unfrozen, distilled | 0.6557 / 0.9182 | +0.6* | ❌ ≈ACC but ↓AUC, ~4× MACs |

\* single seed, within the Baseline's 3-seed spread and selected on best-ACC
epoch (optimistic); its AUC is clearly below Baseline → no robust gain.

The three Track D2 rows are the 2026-06-16 ablation (3 seeds each, all on the
PCEN-Baseline + logit distillation): full bundle (embed + EMA), embed-only, and
EMA-only. Embedding distillation contributes +2.25 ACC / +0.88 AUC (embed-only −
EMA-only); EMA contributes +0.36 ACC / −0.05 AUC on top of embed → neutral.

## What's next

In priority order. The first item is the real blocker for a submission; the
rest are incremental.

0. ✅ **Track D2 confirmed at 3 seeds (2026-06-16) — new project best, 0.6952 /
   0.9402.** Done. The embed-vs-EMA ablation attributes the gain: embedding
   distillation is the driver, EMA is ~neutral. New working base for everything
   below = `pcen_embed.yaml` (embedding distillation; EMA optional). Remaining
   D2 sweeps still worth a cheap pass: β (embedding-loss weight) and the
   logit-adjust/weighted CE arm for rare-class macro-AUC.
1. **Fixed-point PCEN deployment check (critical path).** PCEN currently runs
   as float torch at train/eval time. The whole +5.2 win only counts if it
   survives **int8 quantization + the on-device feature kernel**. `perch` ships
   a streaming `fixed_pcen` reference; port it, quantize the PCEN-Baseline, and
   confirm ACC/AUC hold. Until this passes, "best" is host-only.
2. **Distillation α/T micro-sweep** on the PCEN-Baseline (α ∈ {0.3,0.5,0.7},
   T ∈ {2,3,4}). Distillation is the dominant lever, so this is the most likely
   place left to find free accuracy. Cheap (no recaching).
3. **Compression for submission (Track E).** QAT and/or structured slimming of
   the PCEN-Baseline to lock size/latency/RAM within the envelope. Needed for
   the deliverable regardless of #2.
4. **Optional — attention-pooling head.** The one untested Track F idea
   (per-frame logits + max/mean or attention pooling, 1×1 convs, no recurrence).
   Low expected value after GRU failed, but cheap if #1–#3 leave time.

**Parked / rejected — do not revisit:** heavy augmentation, MSAB-FiLM, 80-mel,
GRU/BiGRU recurrence, EfficientNet backbone. All matched-or-lost vs Baseline.

## Where we stand (numbers)

Task: **11-class** (10 birds + background), 3 s clips @ 24 kHz. Ranked on ACC,
macro ROC-AUC, `.tflite` size, on-device times (feature / model / total), and
peak RAM (baseline 202 KB). Envelope target: ≤111 KB tflite, ≤~330 ms model
time, ≤~200 KB RAM.

| Model | ACC | ROC AUC | Size | MACs | ESP32 model time |
|---|---|---|---|---|---|
| Provided baseline `.tflite` (int8) | 0.5701 | 0.8938 | 111 KB | 23.3 M | ~330 ms |
| **Best: PCEN + logit+embed distill `.pth` (float)** | **0.6952** | **0.9402** | 393 KB | 23.3 M | — (int8 pending) |

Same architecture as the baseline — the gains are all front-end + teacher, so
size/MACs are unchanged. The remaining work is proving the int8/on-device path.

## Key lessons (distilled)

1. **Distillation > architecture.** Perch logit distillation lifted every
   model we tried; swapping backbones did not. The teacher is the workhorse.
2. **The plateau was the front-end, not the model.** Every architecture stalled
   at ~0.57–0.60 on log-mel; PCEN-mel broke the plateau with the *same* model.
3. **The tiny CNN over-regularizes easily.** Heavy augmentation hurt, and every
   "add capacity" idea (FiLM, GRU, bigger backbone) overfit the 2.2k-clip train
   set. Soft teacher targets are the right amount of regularization.
4. **Published ideas didn't transfer.** MSAB-FiLM (BioME) and MobileGRU (Dhar)
   both worked in their original settings but not on our tiny-CNN / PCEN /
   11-class task. Test, don't assume.

## Decisions

- **Objective: accuracy-first** within the baseline envelope (≤111 KB tflite,
  ≤~330 ms model time, ≤~200 KB RAM). Compression (Track E) trims at the end.
- **Teacher: Perch** (local `.venv-perch`). BirdNET / ensemble only if Perch
  underdelivers (it did not).
- **Custom features: research-first** — host-only eval; write the embedded C
  kernel only when a feature beats tuned-mel by a clear margin. PCEN cleared
  this bar (+5.2) and has a real fixed-point path, so it gets promoted.
- **Old `feature/*` branches:** authored by a previous, weaker agent; not built
  upon. Fresh implementations only.

## Lessons from the 2025 technical reports (papers/)

1. **Dhar (MobileGRU)** — full compression pipeline: teacher→student logit
   distillation (KL + CE, α=0.5, T=3.0, cosine LR), magnitude pruning
   (polynomial 0→35%), then int8 QAT. Distillation occasionally *improved* the
   student (regularization). SNR-aware dynamic augmentation was central. (We
   adopted the distillation recipe; the GRU did not transfer — see Track F.)
2. **Martin** — domain-informed feature tuning + slimming: restrict mel band to
   the species range, more mel bins (64), fewer filters, global pooling instead
   of Flatten. Halved on-device compute *and* improved AP.
3. **Walter (organizer)** — tiny CNNs plateau ~90% AP regardless of width;
   ultra-cheap features (16 mels) cut feature time to 1.6 ms. Class imbalance
   was the main failure mode.
4. **Oguamanam (TMU)** — feature extraction can dominate the on-device budget;
   gammatone filterbanks gave better noise contrast than mel.
5. **Espitia** — audio-domain augmentation (pitch shift + noise) lifted a
   slimmed MobileNet.

Caveat: all 2025 reports are *binary* detection; 2026 is 11-class, so expect
smaller headroom from ultra-tiny models and bigger gains from training signal.

## Experiment tracks

Each track = one feature branch. All code runs on CPU and GPU (device
autodetect), locally and on the cluster.

### Track A — Training recipe ❌ (augmentation hurts)
`feature/training-recipe` — SpecAugment time/freq masking, mixup, Gaussian
noise, cosine LR + warmup, label smoothing, best-checkpoint on val AUC. The
recipe scaffolding is kept (cosine/label-smoothing/checkpointing), but
**augmentation over-regularizes the tiny CNN** and is off in the best config.

### Track B — Slim architectures ◐ (efficiency, not accuracy)
`feature/slim-cnn` — `SlimCNN` depthwise-separable student on the mel input.
35k params / 5.9M MACs / ~76 KB int8 (4× fewer MACs than baseline) at ≈ the
same accuracy. A good *efficiency* operating point, not an accuracy win.

### Track C1 — Perch v2 logit distillation ✅ (core win)
`feature/perch-distillation` — Teacher = frozen Perch v2 encoder + an 11-class
head trained on our labels (handles Background natively, avoids eBird mapping
errors). Pipeline in `experiments/perch/`: `export_embeddings.py` (in
`.venv-perch`) → `train_teacher_head.py` (teacher ceiling 0.8925) → distill
into the tiny CNN (`α·KL(T) + (1−α)·CE`, α=0.5, T=3.0). Teacher logits aligned
by wav stem. The backbone of every strong result.

### Track C2 — MSAB-FiLM ❌ (no transfer)
`feature/slim-cnn` — Modulation Spectrogram Average Bands injected via FiLM
(BioME idea ported to conv blocks). Matched-pair test showed the gain was
capacity, not the mechanism: a param-matched plain CNN ties it. Below the +3 pt
bar → no on-device MSAB kernel.

### Track D — Feature extraction front-end ⭐ (project best)
`feature/pcen-frontend` — **PCEN-mel is the biggest single win.** Same framing
as the int log-mel (24 kHz, 40 mel, win 4096 / hop 512 → (1,40,133)); only the
compression changes (log → PCEN, Perch's bio params `smoothing_coef=0.145,
gain=0.8, bias=10, root=4`, 50 Hz low edge). `0.5974 → 0.6497 ACC` over 3
non-overlapping seeds. Ported to torch in
`biodcase_tiny/feature_extraction/pcen_mel.py`; `feature_type: pcen_mel` switch.
PCEN has a fixed-point embedded path (`perch fixed_pcen`) → promotable.
Resolution bump to 80 mel **hurt** (0.610); 40 mel stays default.

### Track D2 — Embedding distillation + training-signal bundle ⭐ (project best, 3 seeds)
**Confirmed: 0.6952 / 0.9402 (3 seeds) — new project best, +4.5 ACC / +1.1 AUC over
PCEN-logit-distill (0.6497 / 0.9296).** The float tflite matches the `.pth` exactly,
so the deployable model is unchanged (97k params / 23 M MACs) — the gain is pure
training signal. The 3-condition ablation attributes it: **embedding distillation is
the driver** (+2.25 ACC / +0.88 AUC over EMA-only), **weight EMA is ~neutral** once
embedding distillation is on (+0.36 ACC / −0.05 AUC, within seed noise and adds
variance). New working base = `pcen_embed.yaml`; EMA optional.

Built on the PCEN-Baseline best. The student is far below the teacher (0.6497 vs
0.8925, a 24-pt gap) and the train set is tiny (2.2k clips), so the highest-value
moves squeeze more *training signal* out of the teacher we already have — all keep
the deployed model byte-for-byte identical (only training changes). Implemented:
1. **Embedding (feature) distillation** — regress Perch's 1536-d embedding from the
   student's 128-d GAP descriptor via a **train-only** `Linear(128→1536)` head
   (`Baseline.forward_with_features`; head discarded at inference, verified the
   exported tflite graph is unchanged). Loss = MSE + (1−cos) on standardized
   embeddings; well-scaled on real PCEN data (~0.84 / ~0.96 at init, order of KD+CE).
2. **Weight EMA** — eval + best-checkpoint on the averaged weights (no BatchNorm in
   Baseline → parameter EMA suffices). Deployed as-is.
3. **Class-imbalance handling on the CE term** — `logit_adjust` (Menon et al.,
   τ·log prior) or inverse-freq `weighted`; aimed at macro-AUC's rare bird classes.
4. **α/T probe** leaning harder on the strong teacher (α 0.7, T 4).

Wiring: `distillation.embed`, `ema`, `class_balance` recipe blocks (all default off,
existing configs unaffected). Configs via `experiments/make_embed_distill_configs.py`
→ `pcen_embed.yaml` (headline: embed + EMA) + `configs/embed_sweep/*` (ablations).
**Data prereq:** the teacher embeddings `perch_v2_cpu/Train.npz` (gitignored) must be
present — copy or regenerate via `export_embeddings.py`. Not yet run; needs 3-seed
eval vs the 0.6497 best.

### Track E — Compression: QAT, pruning, structured slimming ⏳ (todo)
`feature/qat-compression` — Add QAT (torchao / litert-torch in the venv) for
the final candidate. Prefer *structured* slimming (ESP32 gains nothing from
sparse weights) guided by pruning saliency; low-rank/channel reduction of the
32→64→128 progression. Run on the PCEN-Baseline winner.

### Track F — Temporal pooling / SED heads ❌ (recurrence) / ⏳ (attention)
`feature/crnn-recurrence` — **Recurrence rejected.** `BaselineGRU` (identical
conv stack, head swapped to freq-pool → GRU → temporal-pool) hurt both uni
(0.6333) and bi (0.6157); more recurrence → worse. Still untried: attention
pooling / per-frame logits (1×1 convs, no recurrence).

## Process & infrastructure

- **Branch per feature**, PR into `main` only after a win is confirmed; keep
  `main` = reproducible best-known pipeline.
- **Standard run record** per run: ACC, macro AUC, tflite size, MACs, params,
  resolved config, git SHA, seed, host — written to `experiments/runs/<id>/`,
  small and git-tracked.
- **Repro**: fixed seeds; 3 seeds for any result we act on (549-sample val ⇒
  ±2–3 pts ACC seed noise).
- **Deployability gate**: every candidate converts to int8 tflite and passes
  `submission/submission_test.py`; profile on the Korvo-2 before locking.

## Workflow: local ↔ SLURM cluster

The cluster is SLURM-based; the dataset is staged there. Code moves by
`git pull`, results come back as committed run records.

1. Develop on a `feature/*` branch locally; smoke-test on CPU
   (`skip_deployment_flag: True`, a few epochs).
2. On the cluster: `git pull`, `salloc --gres=gpu:1 ...`, set
   `BIODCASE_DATA_ROOT`, then `bash cluster/run_ablations.sh
   'experiments/configs/*.yaml' '1 2 3'` (no sbatch).
3. Each run writes `experiments/runs/<id>/` (config + git SHA + seed + host +
   per-epoch `metrics.csv` + final metrics + artifact sizes). Commit on the
   cluster, pull back locally.
4. Locally: `python experiments/build_dashboard.py` regenerates
   `docs/experiment_plan.html` (this plan + the per-seed results table).
5. Env overrides: `BIODCASE_DATA_ROOT`, `BIODCASE_CONFIG`, `BIODCASE_SEED`,
   `BIODCASE_RUN_NAME`, `BIODCASE_SKIP_DEPLOYMENT=1`,
   `BIODCASE_SKIP_QUANTIZATION=1`, `MPLBACKEND=Agg`.

## Discussion log

- **2026-06-09** — Plan agreed (accuracy-first, Perch teacher, research-first
  custom features). Baseline reproduced locally: float 0.5628 ACC / 0.8931 AUC;
  int8 0.5701 / 0.8938. Track A (`feature/training-recipe`) implemented and
  smoke-tested: feature-space dynamic augmentation, label smoothing, cosine LR
  with warmup, best-checkpoint on val macro-AUC, full run logging.
- **2026-06-09 (cont.)** — Track C design finalized after reading the BioME
  paper. Teacher = frozen Perch v2 (`perch_v2_cpu`, 1536-d, CPU) + 11-class head
  on our labels. All 10 species in Perch v2's taxonomy; Background needs our
  head. Split into C1 (deployable logit distillation) and C2 (MSAB-FiLM
  research, +3 pt promotion bar). Built `export_embeddings.py` and
  `train_teacher_head.py`; export validated end-to-end.
- **2026-06-10** — First full C1 run (H100 MIG, 120 ep, seed 42): 0.5537 /
  0.8841 — **at baseline, marginally below**. Distillation produced no lift in
  this config. Problem: baseline→C1 changed recipe + distillation at once.
  Launching a recipe×distill 2×2 ablation to attribute the effect; added
  `BIODCASE_SEED`/`BIODCASE_RUN_NAME` overrides.
- **2026-06-10 (cont.)** — Ablation 2×2 done (3 seeds). Verdict: **distillation
  works, augmentation hurts.** Best = pure distillation, no aug: 0.5974 / 0.9099
  (+3.5 / +1.7 over baseline). Adding augmentation drops it to 0.5647 / 0.8961 —
  teacher soft targets already regularize, heavy aug over-regularizes the tiny
  CNN. New working base = `abl_distill_no_aug`.
- **2026-06-10 (cont.)** — Track B (`feature/slim-cnn`): `SlimCNN`
  depthwise-separable student, 35k params / 5.9M MACs / ~74 KB int8. Result
  (seed 42): efficiency win, not accuracy. slim_default 0.5719 @76KB, slim_wide
  0.5865 @109KB (≈ dense), slim_deep 0.5483. Student plateaus ~0.57–0.60
  *regardless of architecture* while teacher is 0.89 → the limit is the input
  representation, not the model. Motivates C2 and (later) Track D.
- **2026-06-10 (cont.)** — Track C2 built: MSAB (modulation-spectrum) + FiLM
  ported from BioME. 258-d per-clip vector cached by stem; `SlimCNNFiLM` (66k).
  Matched pair (FiLM on/off, same base + distillation) to isolate the effect.
- **2026-06-11** — C2 result (seed 42): MSAB-FiLM does **not** beat a
  param-matched plain CNN. c2-film 0.5865 (66k) vs slim_wide 0.5865 (62k, no
  FiLM) — gain is capacity, not the mechanism. Below +3 pt bar → no on-device
  MSAB kernel. Negative transfer of the BioME idea to this setting.
- **2026-06-12** — Track D (`feature/pcen-frontend`): **PCEN front-end is a
  decisive win and the new project best.** Hypothesis (user's): the Track B
  plateau is the *input representation*, so match the teacher's compression.
  Swapped int log-mel for torch **PCEN-mel** at identical framing (Perch's bio
  params, 50 Hz low edge). 3 seeds: **0.6497/0.9296 vs log-mel 0.5974/0.9099 =
  +5.2 ACC / +2.0 AUC**, non-overlapping. Not teacher-alignment (teacher logits
  are fixed) — PCEN is simply a better front-end. Has a fixed-point embedded
  path (`fixed_pcen`) → promotable.
- **2026-06-12 (cont.)** — Track D 80-mel: doubling mel resolution **hurts**
  (0.610 vs 0.6497, 2 seeds). The tiny Baseline can't exploit the extra rows;
  40 mel stays default.
- **2026-06-12 (cont.)** — Track F (`feature/crnn-recurrence`): **recurrence
  rejected, tested two ways.** `BaselineGRU` (identical conv stack, head swapped
  to freq-pool → GRU → temporal-pool; `bidirectional` flag). 3 seeds each:
  Baseline 0.6497 → uni-GRU 0.6333 (−1.6) → bi-GRU 0.6157 (−3.4). Monotonic;
  bidirectional (the fair offline test) is worst → recurrence is the wrong
  inductive bias, params overfit. Baseline (conv + GAP) stays best.
- **2026-06-12 (cont.)** — EfficientNet probe + fine-tune (`experiments/timm`,
  `test_efficientnet_ln` ImageNet-init, PCEN input, 1 seed): frozen probe
  0.5738, CE fine-tune 0.5483, **Perch-KD fine-tune 0.6557 / 0.9182**.
  Takeaways: (1) **distillation dominates** — KD lifted the same model +10.7 ACC
  over CE. (2) CE fine-tune fell *below* the frozen probe (unfreezing overfits;
  KD's soft targets regularize). (3) No robust gain over Baseline (ACC within
  its seed spread, AUC clearly lower) at ~4× MACs → not a submission candidate.
  Confirms the wins are the PCEN front-end + Perch distillation, not the model.
- **2026-06-15** — Track D2 (`feature/embed-distill`): implemented the
  training-signal bundle on the PCEN-Baseline best — **embedding (feature)
  distillation** (train-only `Linear(128→1536)` head regressing Perch's 1536-d
  embedding; tflite graph verified unchanged), **weight EMA**, **logit-adjusted /
  class-weighted CE**, and an α/T probe. All gated behind new recipe blocks
  (default off → existing pipeline untouched). End-to-end + regression tested on
  CPU; hint loss well-scaled on real PCEN features. Configs generated
  (`pcen_embed.yaml` + `embed_sweep/`); awaiting the teacher-embedding dump on the
  cluster and a 3-seed eval vs 0.6497. Rationale: distillation is the dominant
  lever and the student trails the teacher by 24 pts, so richer per-clip teacher
  supervision (1536 dims vs 11 logits) is the most likely free accuracy.
- **2026-06-16** — Track D2 **first result (s1, local CPU): 0.6812 / 0.9394** —
  the best single-seed number so far. Same seed, the jump over PCEN-distill-s1
  (0.6430 / 0.9275) is **+3.8 ACC / +1.2 AUC**, and it clears the 3-seed best
  mean (0.6497). Config = `pcen_embed.yaml` (embedding distillation weight 1.0 +
  weight EMA, decay 0.999; class_balance off; α 0.5, T 3.0). The exported float
  tflite reproduces the `.pth` exactly (0.6812 / 0.9394) → the train-only
  projection head leaves the deployed 97k-param model untouched; the win is
  entirely in the training signal, as the project thesis predicts. Caveat: **one
  seed** — s2/s3 + the `embed_sweep/` ablations are queued to (a) confirm and (b)
  separate the embedding-distillation contribution from EMA.
- **2026-06-16 (cont.)** — Track D2 **confirmed at 3 seeds + gain attributed →
  new project best.** Ran a 3-condition × 3-seed ablation on the PCEN-Baseline +
  logit distillation: (a) full bundle `pcen-embed` (embed + EMA) **0.6952 ±
  0.0122 / 0.9402 ± 0.0008**, (b) `embed-noema` (embed only) 0.6916 ± 0.0058 /
  0.9407 ± 0.0008, (c) `ema-only` (EMA, no embed) 0.6691 ± 0.0092 / 0.9319 ±
  0.0031. Verdict: **embedding distillation is the lever** (+2.25 ACC / +0.88 AUC,
  embed-only − EMA-only), **EMA is ~neutral** once embed is on (+0.36 ACC / −0.05
  AUC, full − embed-only; inside seed noise and it widens the spread). Full bundle
  is +4.5 ACC / +1.1 AUC over the prior PCEN-distill best (0.6497 / 0.9296) and
  the deployed model is byte-for-byte unchanged. Promotes Track D2 to project best;
  `pcen_embed.yaml` becomes the working base (EMA kept as a free, optional ACC
  bump). Remaining D2 sweeps (β embedding-loss weight, logit-adjust/weighted CE)
  are cheap follow-ups; the critical path is now the fixed-point PCEN int8 check.
