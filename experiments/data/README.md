# External audio collection (teacher-labeled distillation data)

Expands the tiny 2.2k-clip train set with extra in-domain audio for the 10
target species, to be **labeled by the Perch teacher** (not trusted by source
label) and used as additional distillation data — the highest-ceiling lever for
this small-data regime.

## Layout

Lives alongside the original dataset, under `…/datasets/extra/<source>/`:

```
/home/hguimaraes/datasets/
  biodcase2026_tinyML/…/{Train,Validation}/<class>/   # original dataset
  extra/
    xc/
      raw/   <class>/<Genus_species>/XC*.{mp3,wav,flac}   # downloads (mixed formats)
      clips/ <class>/XC<id>_<startms>.wav                 # 3 s / 24 kHz / mono, teacher-ready
```

`clips/` is a class-subfolder root → consumable as a "split" by
`experiments/perch/export_embeddings.py --data-root …/extra/xc --split clips`.

## Pipeline

1. **Download** (`download_xeno_canto.py`) — per-species Xeno-canto queries by
   genus + epithet. Needs `xenocanto-api` + `python-dotenv` and an API key in
   `.env` (`XENO_CANTO_API_KEY=…`, git-ignored).
   ```
   python experiments/data/download_xeno_canto.py \
       --quality ">C" --length 3-120 --max-per-species 800 \
       --out /home/hguimaraes/datasets/extra/xc/raw
   ```
   - `q:">C"` = quality A+B (needed: A-only starves Mallard/Blue Tit/Tawny Owl).
   - `len:3-120` keeps files sliceable but bounded (focal recordings run minutes).
   - Skips already-downloaded recordings on re-run.

2. **Slice** (`slice_audio.py`) — mp3/wav → uniform 3 s, 24 kHz mono wav with
   energy-VAD gating (drops the long silences in focal recordings) and a
   per-class cap for balance. `hop < clip` gives overlap so scarce classes reach
   target.
   ```
   python experiments/data/slice_audio.py \
       --in-dir /home/hguimaraes/datasets/extra/xc/raw \
       --out-dir /home/hguimaraes/datasets/extra/xc/clips \
       --target-per-class 5000 --hop-sec 1.5
   ```
   Classes printed as "under target" → re-run those with a smaller `--hop-sec`
   (e.g. 1.0) to add overlap.

3. **Label + distill** (next) — run Perch over `clips/` to get soft logits (and
   embeddings), then add to the distillation training set (Track D2 recipe).

## Availability (A+B, 3–120 s, measured 2026-06)

~28.6k recordings → ~394k non-overlapping 3 s clips. Every class clears 5k;
Mallard is the floor (896 recordings → ~7k raw clips, needs mild overlap after
VAD). Per-class counts: see the metadata previews (`--metadata-only`).

## Notes / caveats

- **Domain shift:** Xeno-canto is *focal* (single bird, close mic); the task is
  *passive soundscape*. Hence teacher-labeling + treating species folders as a
  prior only. Consider Perch-confidence filtering before adding to training.
- **Background class** is not collected here (not a species) — source separately.
- `species.py` holds the canonical species table, shared with the (planned)
  iNaturalist collector so layouts/labels stay consistent.
- Requirements: `pip install -r experiments/data/requirements.txt`.
