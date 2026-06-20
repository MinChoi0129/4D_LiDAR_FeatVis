# 4D-LiDAR FeatVis

Standalone visualizer for **4D-Utonia** per-point features and **MOS** labels on
LiDAR scans. Copy this folder to a machine with a display and run.

## The feature dump

`tools/extract_features_4d.py` produces an **accumulated 4D dump**, `(N, 8+K)`:

| cols | meaning |
|---|---|
| `0:3` | `x, y, z` — every sweep ego-motion-aligned into the **current frame** |
| `3:6` | `pca_r, pca_g, pca_b` — the encoder's high-dim feature → 3 PCA comps → RGB |
| `6` | `time` — signed seconds (`0` = current frame; past `<0`, future `>0`) |
| `7` | `gt_label` — MOS ground truth (`-1` ignore / `0` static / `1` moving), valid on the current frame only |
| `8:8+K` | `feat_0..` — top-K PCA features (`K=--feat-dims`, default 32) for the `similarity` mode |

It is **accumulated** on purpose: a single frame's features look flat, but stacking
the temporal window (all sweeps share one coordinate frame) is what makes the 4D
signal visible — moving objects smear into trails while static structure stays sharp.

## Install & run

```bash
pip install -r requirements.txt

# richest feature view: PCA-RGB (default)
python visualize.py --npy examples/feat_seq08_4017.npy --mode rgb

# the 4D signal: colour by sweep time (past=blue ... now=white ... future=red)
python visualize.py --npy examples/feat_seq08_4017.npy --mode time

# ground-truth moving objects (green) on the current frame
python visualize.py --npy examples/feat_seq08_4017.npy --mode label

# "are the features good?": cosine similarity to the moving prototype (bright=similar)
python visualize.py --npy examples/feat_seq08_4017.npy --mode similarity --ref moving
# ...or shift+click your own reference point (e.g. one car) and see what lights up
python visualize.py --npy examples/feat_seq08_4017.npy --mode similarity --pick

# headless (no display): save a top-down BEV PNG instead of a 3D window
python visualize.py --npy examples/feat_seq08_4017.npy --mode time --bev out.png
```

A sample dump ships in `examples/` (SemanticKITTI seq08 frame 4017), so it runs
out of the box. Example BEV renders are in `renders/`.

## Two npy schemas (modes validate their columns)

There are two kinds of dump. **Modes refuse to run on the wrong one** instead of
silently mis-reading a float feature column as labels (a continuous PCA column is
*not* a class id):

- **A — feature dump** `(N, 8)` (above): use `rgb` / `feature` / `time` / `label`.
- **B — prediction dump** `(N, ≥5)` from a finetuned MOS model, with **integer**
  `gt` and `pred` columns: use `label` (gt) / `confusion` (gt vs pred).

| mode | npy columns | colors | use for |
|---|---|---|---|
| `rgb` | `3,4,5` (PCA-RGB float) | feature cols → RGB | **4D feature** (default, richest) |
| `feature` | `3` (scalar float) | viridis heatmap | one feature axis |
| `time` | `6` (seconds) | coolwarm (past→future) | **the 4D / motion signal** |
| `label` | `7` (int class) | moving=green, static=gray, ignore=dim | GT (or a prediction dump's gt) |
| `confusion` | `gt`,`pred` (int) | TP green / FP blue / FN red / TN gray | MOS error analysis (schema B) |
| `similarity` | `8:` (feat block) | turbo (bright=similar) | **judging feature quality** (see below) |

Column indices are overridable (`--color-cols`, `--feat-col`, `--time-col`,
`--label-col`, `--gt-col`, `--pred-col`, `--feat-start`).

## How to read it

- **rgb**: *similar color = similar learned representation*. Look for the scene
  carving into ground / buildings / vegetation / vehicles **without labels**, and
  especially whether **moving objects differ in color from static ones**. Absolute
  hue is arbitrary; relative color is what matters. (PCA of an SSL feature is often
  geometry-dominated — distance/height — so don't over-read a single axis.)
- **time**: in the accumulated cloud, static structure overlaps across sweeps (mixed
  color) while a moving object leaves a blue→red streak. Zoom / `--range-crop` to a
  moving vehicle to see the trail clearly.
- **label**: green = moving GT on the current frame; past/future sweeps are unlabeled
  (dim). The honest sanity check for "where are the moving objects".
- **confusion**: needs a prediction dump (schema B); green=TP, blue=FP, red=FN.

### How do I judge if the features are *good*? — `similarity` mode

You **cannot** tell from `rgb`: PCA→3 keeps only ~15% of the 768-dim feature, so
colour is a weak proxy, and a single object showing many colours is *expected*
(per-point features encode local part/geometry, and the top PCA axes are geometry-
dominated — height/range). Object-colour uniformity is **not** a quality signal.

Instead, use `similarity`: it computes the **cosine similarity of every point's
feature to a reference** and maps it to a heatmap (bright = similar). A *good*
feature makes a whole object — and other instances of the same thing — light up
together.

```bash
# pick a car by shift+click; if features are good, that car + other cars glow
python visualize.py --mode similarity --pick
# reference = the mean feature of GT moving points -> "how moving-like is each point?"
python visualize.py --mode similarity --ref moving
# reference = a specific point index or xyz
python visualize.py --mode similarity --ref-idx 12345
python visualize.py --mode similarity --ref-xyz 8.0 -3.0 0.0
```

For the 4D goal specifically: `--ref moving` shows whether moving objects are
*distinctive in feature space*. If moving and static cars light up the same, the
SSL leaned on geometry, not motion — a real, useful negative result. The decisive
quality metric is still the downstream **MOS moving-IoU** (scratch vs pretrained).

## Options (config.yaml or CLI)

`--mode {rgb|feature|time|label|confusion|similarity}`, `--npy`,
`--color-cols a b c`, `--gamma`, `--feat-col`, `--norm {percentile|minmax|sigmoid}`,
`--cmap`, `--time-col`, `--time-cmap`, `--label-col`, `--gt-col`, `--pred-col`,
`--ref {moving|static}` / `--ref-idx N` / `--ref-xyz X Y Z` / `--pick`,
`--feat-start`, `--sim-cmap`, `--range-crop RMIN RMAX` / `--no-range-crop`,
`--flatten-2d`, `--point-size`, `--bev out.png`.

## Producing more dumps (on the training machine)

```bash
# accumulated 4D feature dump for any frame (uses the pretrain weights)
python tools/extract_features_4d.py \
    --ckpt exp/utonia4d/pretrain/model/model_last.pth \
    --seq 8 --frame 4017 --out feat_seq08_4017.npy
# --teacher      : use the EMA-teacher encoder
# --t0-only      : legacy single-frame dump (no accumulation)
# --hypercolumn  : concat ALL encoder scales per point -> no 0.8m voxel 'blocks'
# --feat-dims N  : also store top-N PCA feats (cols 8:8+N) for similarity mode (default 32; 0=off)
```

### Why does `rgb` look blocky? (resolution)

The pretrain encoder is `enc_mode` (encoder only, no decoder), so its features live
at the **coarse bottleneck grid (~0.8 m)**. The default dump broadcasts that one
coarse feature to every point inside the 0.8 m voxel, so you see ~0.8 m colored
cubes -- e.g. one frame has 224k points but only ~9k distinct features. This is
*honest*: it is the resolution at which the self-distillation actually operates.

Pass `--hypercolumn` to instead concat every encoder scale (0.05 m fine ... 0.8 m
coarse) per point -> features become per-point unique (no blocks), at the cost of
mixing in lower, less-SSL-supervised levels. Use it for a smooth zoomed-in 3D view;
use the default to see the true representation resolution.

For `confusion`, export a per-point `[x, y, z, gt, pred]` npy from the finetuned MOS
model and point `--gt-col` / `--pred-col` at the gt/pred columns.
