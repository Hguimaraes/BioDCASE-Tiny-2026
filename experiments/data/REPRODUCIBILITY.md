# Reproducing the extra training data (BioDCASE-Tiny 2026)

Single source of truth for regenerating the extra teacher-labeled samples used to
augment training: **Xeno-canto birds** (the 10 target species) + **TAU urban
background** (the Background class). Everything below is deterministic given the
pinned sources, seeds, params, and library versions.

> Honest caveat up front: Xeno-canto is a **live, mutable** database, so the
> download is **not** reproducible from the query alone — it is pinned by the
> committed id manifest (`xc_download_manifest.csv`). Individual recordings can
> still be removed by their authors over time, so the **definitive** artifact is
> the archived processed output (clips/embeddings), backed up off-disk. TAU is an
> immutable Zenodo record and is fully reproducible.

## Sources (pinned)

| Source | Pin | Notes |
|---|---|---|
| Xeno-canto | `experiments/data/xc_download_manifest.csv` (7,996 recordings; 7,750 used) | each row has XC `id` + `file` URL → re-download by id, NOT by re-query |
| TAU Urban Acoustic Scenes 2022 Mobile, development | Zenodo record **6337421** (audio.1–16.zip + meta), immutable + MD5 | DCASE 2022 Task 1 dev set; 230,350 × 1 s segments |

## Environments

Two venvs (pinned in `requirements.txt`): `.venv` (torch: download/slice/sample/
filter) and `.venv-perch` (Perch/TF: `export_embeddings.py` only). Key versions:
`librosa 0.11.0`, `soundfile 0.14.0`, `perch-hoplite 0.1.1`, `torch 2.12.0`,
`numpy 2.4.6` (.venv) / `2.0.2` (.venv-perch), `xenocanto-api 0.3.1`.

## Pipeline (run in order)

Paths assume dataset root `/home/hguimaraes/datasets`. Species table:
`experiments/data/species.py`. All randomness is seeded; file lists are sorted
before any seeded shuffle, so selection is filesystem-independent.

### A. Xeno-canto download  (→ raw recordings)
```bash
.venv-perch/bin/python experiments/data/download_xeno_canto.py \
    --quality ">C" --length 3-120 --max-per-species 800 \
    --out /home/hguimaraes/datasets/extra/xc/raw
```
Quality A+B (`>C`), 3–120 s, ≤800/species → ~7,933 recordings. **To reproduce
exactly**, fetch the ids in `xc_download_manifest.csv` (the live query may now
return a different set).

### B. Slice XC → 3 s clips  (deterministic; extract-all)
```bash
.venv/bin/python experiments/data/slice_audio.py \
    --in-dir  /home/hguimaraes/datasets/extra/xc/raw \
    --out-dir /home/hguimaraes/datasets/extra/xc/clips \
    --hop-sec 2.5            # defaults: clip 3 s, 24 kHz mono, light VAD (min-active 0.1), no cap, seed 0
```
→ **123,958 clips** (every VAD-passing window from every recording).

### C. Perch embeddings  (the expensive pass; resumable)
```bash
.venv-perch/bin/python experiments/perch/export_embeddings.py \
    --data-root /home/hguimaraes/datasets/extra/xc --split clips \
    --out /home/hguimaraes/datasets/extra/xc/embeddings --preset perch_v2_cpu --shard-size 2000
```
→ `…/extra/xc/embeddings/perch_v2_cpu/clips.npz`. Frozen Perch ⇒ deterministic.

### D. Agreement + confidence filter  (pick τ)
```bash
.venv/bin/python experiments/data/filter_clips.py predict \
    --emb  /home/hguimaraes/datasets/extra/xc/embeddings/perch_v2_cpu/clips.npz \
    --head experiments/perch/embeddings/perch_v2_cpu/teacher_mlp/teacher_head.pt \
    --out  /home/hguimaraes/datasets/extra/xc/predictions.npz
.venv/bin/python experiments/data/filter_clips.py select \
    --pred /home/hguimaraes/datasets/extra/xc/predictions.npz \
    --criterion agree_conf --threshold 0.8 \
    --clips-root /home/hguimaraes/datasets/extra/xc/clips \
    --out /home/hguimaraes/datasets/extra/xc/kept_clips.txt
```
**τ = 0.8, criterion `agree_conf` → exactly 86,303 kept bird clips** (confirmed:
matches `kept_clips.txt`). Deterministic given `predictions.npz` + τ.

### E. TAU download  (Zenodo 6337421, files 1–16 + meta)
```bash
DEST=/media/hguimaraes/Expansion/datasets/TAU-urban-acoustic-scenes-2022-mobile
BASE="https://zenodo.org/records/6337421/files/TAU-urban-acoustic-scenes-2022-mobile-development.audio"
for i in $(seq 1 16); do
  wget -c -O "$DEST/...audio.${i}.zip" "${BASE}.${i}.zip?download=1"
done   # then unzip -n into /home/hguimaraes/datasets/extra/tau-2022
```

### F. TAU → 3 s background clips  (deterministic; seeded)
```bash
.venv/bin/python experiments/data/make_background_from_tau.py \
    --in-dir  /home/hguimaraes/datasets/extra/tau-2022/TAU-urban-acoustic-scenes-2022-mobile-development/audio \
    --out-dir /home/hguimaraes/datasets/extra/xc/clips/Background \
    --n 10000           # defaults: n-seg 3, seed 0
```
Concatenates 3 consecutive 1 s segments; round-robin over **14,400 unique
recordings** → **10,000 background clips**, each from a distinct recording.
Verified identical across runs at fixed seed.

## Pinned hyperparameters

| Step | Params |
|---|---|
| XC query | `q:">C"`, `len:3-120`, ≤800/species |
| XC slice | clip 3 s, hop 2.5 s, 24 kHz mono, light VAD `min-active 0.1` `floor-factor 0.15`, extract-all, seed 0 |
| Perch | preset `perch_v2_cpu`, mean-pool over frames/channels |
| Filter | criterion `agree_conf`, **τ = 0.8** |
| TAU bg | `n-seg 3`, n 10000, seed 0 |

## Final artifacts

| Artifact | Count | Location |
|---|---|---|
| XC kept bird clips | 86,303 | `extra/xc/clips/<species>/` + `kept_clips.txt` |
| TAU background clips | 10,000 | `extra/xc/clips/Background/` |
| **Extra 11-class pool** | **96,303** | `extra/xc/clips/` |

## Determinism notes
- File lists are **sorted before seeded shuffle** (glob order is filesystem-dependent) — fixed 2026-06-18.
- XC slice is order-independent (extract-all: every window kept).
- Pin library versions (`requirements.txt`): librosa resampling & Perch inference are version-sensitive.
- For submission, also **archive the processed clips/embeddings off-disk** (XC mutability insurance); TAU needs no such backup.
