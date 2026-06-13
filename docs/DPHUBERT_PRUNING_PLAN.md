# Plan: DPHuBERT-style joint distillation + structured pruning to a 120k student

Branch: `feature/dphubert-pruning` · drafted 2026-06-12 · status: **core implemented & validated**

Goal: a student of **~120k parameters** obtained by *learned* structured pruning
(channel selection) instead of hand-picked width, distilling from **Perch 2.0**.
Builds on the Track G EfficientNet-B3 + layer-to-layer work (`feature/effnetb3-distill`).

## Implementation status (2026-06-12)

Built and smoke-validated in `experiments/pruning/`:
- `hardconcrete.py` — FLOP/DPHuBERT Hard-Concrete L0 gate (differentiable `l0_norm`).
- `gated_effnet.py` — EfficientNet-style net with gates on **MBConv expand
  channels only** (block I/O fixed → no residual coupling). Differentiable
  `get_num_params()`; `to_pruned()` rebuilds a compact gate-free model. Verified
  `max|gated_eval − pruned| = 0.0` (exact, by folding the eval soft-mask scale
  into the SE-reduce + project input weights).
- `train_dphubert.py` — 3 stages (distill+prune / prune / final-distill) on the
  **40-mel PCEN** student; logit KD from Perch + optional single `spatial_embedding`
  hint (interpolated to the student grid). Lagrangian per DPHuBERT: weights at
  `lr`, gates at `+reg_lr`, multipliers λ1/λ2 at `−reg_lr` (gradient ascent).
- **Validation run** (base width 0.35 → 560k, target 120k, 50 stage-1 epochs):
  expected sparsity climbed 0.006→0.766 tracking the 0.786 target, ~params
  560k→131k; `prune()` → **134,543 params**, functional right after pruning
  (acc 0.45, not random) → mechanism confirmed. (Accuracy not meaningful yet:
  only 50 stage-1 + 1 final epoch.)

Remaining: full-length runs (≈100 stage-1 + 60 final epochs), `--feature-distill on`
(needs the `export_spatial.py` teacher dump), and eval vs the hand-slimmed 120k /
Baseline. To hit exactly 120k, run the full schedule or nudge `reg_lr`/target.

---

## 1. What DPHuBERT does (the method we're porting)

DPHuBERT ([Peng et al., Interspeech 2023](https://arxiv.org/abs/2305.17651)) is
**joint distillation + structured pruning** in three stages:

1. **Joint distill + prune** (`distill.py`): the student is a **copy of the
   teacher** with learnable **Hard-Concrete (L0) gates** on structured units
   (conv channels, attention heads, FFN intermediate). Train with
   `loss = distill_loss + loss_reg`, where:
   - `distill_loss` is **layer-to-layer** L1 + cosine (their `cos_type=log_sig`
     is exactly the `-logsigmoid(cos)` we already use in `train_effnetb3_l2l.py`).
   - `loss_reg` is a **size-targeting Lagrangian** ([lightning.py](../../DPHuBERT/lightning.py)):
     ```
     expected_sparsity = 1 - student.get_num_params() / original_num_params   # differentiable in the gates
     loss_reg = λ1·(expected − target) + λ2·(expected − target)²              # λ learned by gradient ascent
     ```
     `get_num_params()` sums each layer's params as a function of its gates'
     expected L0, so the optimizer is pushed to an **exact target size**; the
     target sparsity warms up linearly from 0.
2. **Prune** (`prune.py`): physically drop the zeroed channels/heads → a smaller,
   dense model.
3. **Final distill** (`final_distill.py`): distill the pruned model again (no
   gates) to recover accuracy.

The **core contribution we want** is stage 1: *learned, size-targeted structured
pruning driven by distillation*. We already have the distillation half.

---

## 2. The blocker that reshapes everything: Perch is a black box

DPHuBERT requires three kinds of access to the teacher that **we do not have**:

| DPHuBERT needs | Our situation with Perch 2.0 |
|---|---|
| Teacher **weights** to init the student as a copy | Perch is a **TF SavedModel** — weights not accessible, can't copy/init from it |
| Insert **gates into the teacher's layers** | Can't edit a frozen SavedModel graph |
| **Per-layer** hidden states for layer-to-layer distill | SavedModel exposes only `embedding`, `spatial_embedding` (one map), `logits`, `spectrogram` |

So we **cannot prune Perch's own copy**. Whatever we prune must be a network
**we build and control**. Two ways to honor "teacher = Perch 2.0":

### Variant B — direct (literal "teacher = Perch")
Prunable model = our own EfficientNet-B3 (ImageNet-init via timm, or scratch) with
channel gates. Distill **directly from Perch**: logit KD + the single
`spatial_embedding` hint (the L2L we built). Lagrangian prunes to 120k. Final distill.
- ✅ Literal interpretation; reuses everything in `feature/effnetb3-distill`.
- ❌ Loses DPHuBERT's two pillars: **no teacher-weight init** (start from ImageNet/scratch)
  and **one hint layer** only (Perch exposes a single spatial map).

### Variant A — interpose our own B3 teacher (recommended)
First train a **wide** EfficientNet-B3 *we control* on PCEN-Perch with Perch
distillation (logit + spatial hint) → call it **B3★**. Then DPHuBERT-prune a copy
of B3★ down to 120k, distilling **layer-to-layer from B3★** (which we own → *all*
intermediate stage maps available, and the 120k student **starts as a copy of
B3★**).
- ✅ Recovers both DPHuBERT pillars: **teacher-copy init** + **true multi-layer
  hints**. Two-hop: Perch → B3★ → pruned-120k.
- ❌ Extra stage (train B3★ first); B3★ only as good as our Perch distillation.

**Recommendation:** scaffold for **Variant B first** (smallest delta from current
code, answers "does learned pruning beat hand-slimming?"), keep **Variant A** as
the upgrade if B3★ is a meaningfully better teacher than raw Perch-with-one-hint.

---

## 3. The hard part: structured channel pruning for EfficientNet

DPHuBERT's gates target HuBERT units (conv channels, attention heads, FFN interm).
EfficientNet has **no attention** — we prune **channels**, which is trickier
because of residual coupling. Port the FLOP `HardConcrete` module (small,
self-contained — [hardconcrete.py](../../DPHuBERT/wav2vec2/hardconcrete.py)) and
attach gates to:

- **MBConv expand** channels (the 1×1 expand output = the "interm" analog) — the
  cheapest, highest-leverage place to prune; depthwise + SE follow the gate.
- **MBConv project / block-output** channels — **coupled** within a stage by the
  residual add: blocks 2..n in a stage add to the block input, so their output
  channel dims must stay equal. → blocks in the same stage **share one output
  gate** (group-coupled pruning), exactly the residual-handling DPHuBERT does for
  the transformer. This is the main correctness risk.
- **Stem** out-channels and **head** (final 1×1) channels.
- (Optional) whole-block gate (`hard_concrete_for_layer`) to drop entire MBConvs,
  like DPHuBERT's `ffnlayer`/`attlayer` units — lets depth shrink too.

Then implement a **differentiable `get_num_params()`**: each conv's param count =
`f(active_in, active_out, k, groups)` with `active = gate.l0_norm()`; sum over the
net (mirrors `wav2vec2/model.py:get_num_params` and the per-component
`get_num_params` methods). `original_num_params` = the unpruned count.
**Target:** `target_sparsity = 1 − 120_000 / original_num_params` (so the prunable
base must start > 120k; e.g. start from `channel_multiplier≈0.25` B3 ≈ 590k and
prune ~80%, giving the gates real room to select channels).

A physical `prune()` (mirror `prune.py` + `pruning_utils.py`) rebuilds a dense
timm-style EfficientNet from the surviving channels, so the final 120k model has
no gates and runs as a normal CNN.

---

## 4. What we reuse vs. build new

**Reuse (already on `feature/effnetb3-distill`):**
- PCEN-Perch front-end (`pcen_mel_perch`) + `cache_pcen_perch`.
- `EffNetB3Slim` / the slim-B3 builder, `forward_with_spatial`.
- The L1+cosine hint loss + teacher dumps (`export_spatial.py`, teacher logits).
- The standalone training-loop style (`train_effnetb3_l2l.py`).

**Build new:**
1. `pruning/hardconcrete.py` — port FLOP's `HardConcrete` (≈70 lines, drop-in).
2. `pruning/prunable_effnet.py` — wrap a timm EfficientNet-B3 with channel gates +
   group-coupled stage gates + differentiable `get_num_params()` + `prune()`.
3. `pruning/lagrangian.py` — the size Lagrangian (λ1, λ2, target-sparsity warmup),
   lifted from `lightning.py:training_step`.
4. `experiments/timm/dphubert_prune.py` — stage-1 joint distill+prune loop
   (distill loss from §2 + Lagrangian), separate optimizer group for the gates +
   λ at `reg_learning_rate`.
5. `prune` + `final_distill` steps (can be one script with `--stage`).
6. Config `experiments/configs/dphubert_120k.yaml` (target_params, base width,
   sparsity warmup, reg_lr, distill weights).

---

## 5. Honest expectations & open decisions

- **Prior:** our recurring finding is *distillation dominates architecture*; a
  hand-slimmed B3 already ≈ Baseline. So the **accuracy** upside of learned
  pruning is uncertain. The real value is **exact, deployable size control** (hit
  120k while letting L0 choose *which* channels) — potentially a better 120k than
  hand-picked width, and a principled compression story for the report.
- **Decide: which base to prune — MACs vs. params.** 120k *params* on the
  **Perch-resolution** input (128×500) is still ~tens of M MACs (input dominates
  MACs) → research model, **not deployable**. If the goal is a *deployable* 120k
  model, prune the **small-input** B3 (40×133, ~5M MACs, `pcen_effnetb3.yaml`)
  and accept only the single Perch hint (no resolution match). **Pick the goal
  first** — layer-to-layer research (Perch-res) vs. deployable compression
  (small-input). They point at different bases.
- **Risk:** residual/stage channel coupling is the classic structured-pruning
  footgun; get the grouping wrong and `prune()` produces a broken graph. Budget
  time for a tiny end-to-end prune test on a 2-stage toy net first.
- **Single-hint limitation** persists under Variant B (Perch gives one map);
  Variant A removes it but adds the B3★ stage.

## 6. Milestones

1. Port `HardConcrete`; unit-test `l0_norm` + sampling. *(small)*
2. Wrap a timm B3 with expand-channel gates only; verify differentiable
   `get_num_params()` tracks a hand-pruned count. *(core risk)*
3. Add stage-coupled output gates + `prune()`; toy end-to-end prune test. *(core risk)*
4. Stage-1 loop: distill (reuse loss) + Lagrangian to target 120k; confirm the
   expected size converges to target. *(integration)*
5. `prune()` → `final_distill` → evaluate vs. hand-slimmed 120k and Baseline.
6. If promising, Variant A (B3★ teacher + multi-layer hints).

**Estimate:** milestones 1–4 are the bulk (the prunable-EfficientNet + Lagrangian
is the real engineering, ~most of the effort); 5–6 reuse existing training/eval.
