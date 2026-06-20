"""Per-point coloring modes for 4D-LiDAR feature / MOS visualization.

All functions take the loaded (N, C) npy array and return (N, 3) RGB in [0, 1].

Two npy SCHEMAS (do not mix them up -- modes validate their columns and raise
rather than silently mis-reading a feature column as labels):

  A) FEATURE dump  (tools/extract_features_4d.py), (N, 8):
       0:3 = xyz | 3:6 = PCA-RGB (float) | 6 = time(s) | 7 = GT label (-1/0/1)
       -> use rgb / feature / time / label
  B) PREDICTION dump (a trained MOS model), (N, >=5):
       0:3 = xyz | gt = int class col | pred = int class col
       -> use label (gt) / confusion (gt, pred)

Modes
-----
rgb        : columns [3,4,5] (floats) used directly as RGB -- the PCA-RGB feature
             view; the *richest* look at a multi-dim self-supervised feature.
feature    : a single scalar column -> viridis heatmap (one feature axis).
time       : the per-point time column (signed seconds) -> diverging colormap
             (past=blue ... present=white ... future=red). Reveals the 4D motion
             trails in an accumulated multi-sweep dump.
label      : an INTEGER class column -> discrete MOS colors (moving=green,
             static=gray, ignore=dim). Default col 7 (the feature dump's GT).
confusion  : a gt + a pred INTEGER column -> TP/FP/FN/TN colors. Needs a
             prediction dump (schema B); a feature dump has no pred column.
similarity : cosine similarity of every point's FEATURE (cols 8:, the feat block)
             to a reference -> colormap (bright = similar). The honest "are the
             features good?" view -- a good feature lights up a whole object and
             its siblings together. Reference = a clicked point, an (x,y,z), or a
             'moving'/'static' GT prototype. Needs a dump made with --feat-dims>0.
"""
from __future__ import annotations

import numpy as np

def _get_cmap(name):
    """Fetch a matplotlib colormap across versions (cm.get_cmap removed in mpl>=3.9)."""
    try:
        import matplotlib
        return matplotlib.colormaps[name]          # mpl >= 3.6
    except Exception:
        try:
            from matplotlib import cm
            return cm.get_cmap(name)               # mpl < 3.9 fallback
        except Exception:
            return None                            # matplotlib unavailable

# MOS class -> RGB. In an accumulated 4D dump the past/future sweeps are unlabeled
# (-1) and make up ~83% of points, so ignore is a DIM backdrop (not bright purple)
# -- that way static (gray) and the rare moving (bright green) actually pop.
MOS_COLORS = {
    -1: (0.14, 0.09, 0.18),  # ignore / unlabeled : dim plum (recedes)
    0:  (0.45, 0.45, 0.48),  # static             : gray
    1:  (0.10, 0.95, 0.25),  # moving             : bright green
}


def normalize(v: np.ndarray, mode: str = "percentile") -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-v))
    if mode == "minmax":
        lo, hi = float(v.min()), float(v.max())
    else:  # percentile (robust to outliers) — default
        lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    return np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def color_rgb(data: np.ndarray, cols=(3, 4, 5), gamma: float = 1.0) -> np.ndarray:
    """Use 3 feature columns directly as RGB (the PCA-RGB feature view)."""
    idx = list(cols)
    assert data.shape[1] > max(idx), f"need columns {idx}, npy has {data.shape[1]}"
    rgb = np.clip(data[:, idx].astype(np.float32), 0.0, 1.0)
    if gamma != 1.0:
        rgb = np.power(rgb, gamma)
    return rgb


def color_feature(data: np.ndarray, col: int = 3, norm: str = "percentile",
                  cmap: str = "viridis") -> np.ndarray:
    """Single scalar column -> colormap heatmap (default viridis)."""
    s = normalize(data[:, col], norm)
    cm = _get_cmap(cmap)
    if cm is None:
        return np.stack([s, s, s], axis=1)
    return np.asarray(cm(s))[:, :3]


def _require_class_column(data: np.ndarray, col: int, mode: str) -> np.ndarray:
    """Return col as integer class ids, or raise a clear error if it's a feature col.

    Guards against the silent-garbage failure: a feature dump's PCA columns are
    floats in [0,1]; rounding them as if they were class ids paints ~50% of the
    cloud 'moving'. We refuse instead of lying.
    """
    if data.shape[1] <= col:
        raise ValueError(
            f"mode '{mode}' needs an integer class column at col {col}, but the npy "
            f"has only {data.shape[1]} columns ({data.shape}). "
            f"A feature dump has no label/pred columns -- use mode rgb/feature/time."
        )
    v = data[:, col]
    is_int = bool(np.allclose(v, np.round(v)))
    n_uniq = int(len(np.unique(np.round(v, 6))))
    if not is_int or n_uniq > 64:
        flag = "non-integer " if not is_int else ""
        which = "--label-col" if mode == "label" else "--gt-col / --pred-col"
        raise ValueError(
            f"mode '{mode}' expects INTEGER class ids at col {col}, but col {col} has "
            f"{n_uniq} distinct {flag}values in [{v.min():.3f}, {v.max():.3f}] -- "
            f"that is a continuous FEATURE column, not labels.\n"
            f"  This looks like a feature dump. Either use mode rgb/feature/time, or "
            f"point {which} at the real class column.\n"
            f"  (4D feature dumps store the GT label at col 7; a confusion view needs a "
            f"separate MOS-prediction dump with both gt and pred columns.)"
        )
    return np.rint(v).astype(int)


def color_label(data: np.ndarray, col: int = 7) -> np.ndarray:
    """Integer MOS class column -> discrete colors (moving=green, static=gray, ignore=purple)."""
    lab = _require_class_column(data, col, "label")
    out = np.full((len(lab), 3), 0.3, dtype=np.float32)
    for k, c in MOS_COLORS.items():
        out[lab == k] = c
    return out


def color_time(data: np.ndarray, col: int = 6, cmap: str = "coolwarm") -> np.ndarray:
    """Per-point time column (signed seconds) -> diverging colormap.

    Centered at t=0 (the current frame) so past sweeps and future sweeps fall on
    opposite ends of the colormap. In an accumulated dump this paints the motion
    trail of every moving object from past -> present.
    """
    assert data.shape[1] > col, f"need column {col} (time), npy has {data.shape[1]}"
    t = data[:, col].astype(np.float32)
    span = float(np.abs(t).max()) + 1e-9          # symmetric around 0
    s = np.clip(0.5 + 0.5 * t / span, 0.0, 1.0)    # past->0, t0->0.5, future->1
    cm = _get_cmap(cmap)
    if cm is None:
        return np.stack([s, s, s], axis=1)
    return np.asarray(cm(s))[:, :3]


def color_confusion(data: np.ndarray, gt_col: int = 7, pred_col: int = 8,
                    moving_class: int = 1) -> np.ndarray:
    """gt + pred columns -> confusion colors (TP green, FP blue, FN red, TN gray).

    Needs a MOS-prediction dump that carries BOTH a gt and a pred class column;
    a feature dump has neither, so this raises rather than fabricating a result.
    """
    gt = _require_class_column(data, gt_col, "confusion")
    pr = _require_class_column(data, pred_col, "confusion")
    out = np.full((len(gt), 3), 0.30, dtype=np.float32)              # TN: gray
    out[gt == -1] = (0.55, 0.20, 0.65)                              # ignore: purple
    tp = (gt == moving_class) & (pr == moving_class)
    fp = (gt != moving_class) & (gt != -1) & (pr == moving_class)
    fn = (gt == moving_class) & (pr != moving_class)
    out[tp] = (0.10, 0.85, 0.25)                                    # green
    out[fp] = (0.15, 0.35, 0.95)                                    # blue
    out[fn] = (0.95, 0.15, 0.15)                                    # red
    return out


def _features(data: np.ndarray, feat_start: int = 8) -> np.ndarray:
    """The reduced per-point feature block (cols feat_start:), or a clear error."""
    if data.shape[1] <= feat_start:
        raise ValueError(
            f"similarity mode needs a feature block at col {feat_start}+, but the npy has "
            f"only {data.shape[1]} columns. Re-extract WITH features:\n"
            f"  python tools/extract_features_4d.py --feat-dims 32 --ckpt ... --out X.npy\n"
            f"(PCA-RGB alone is 3 dims -- too few to compare points.)"
        )
    return data[:, feat_start:].astype(np.float32)


def reference_vector(data: np.ndarray, spec, feat_start: int = 8,
                     label_col: int = 7) -> np.ndarray:
    """Build the reference feature to compare against.

    spec may be: an int point index; an (x,y,z) -> nearest point; or 'moving' /
    'static' -> the mean feature of that GT class (a class prototype).
    """
    F = _features(data, feat_start)
    if isinstance(spec, (int, np.integer)):
        return F[int(spec)]
    if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 3:
        d = np.linalg.norm(data[:, :3] - np.asarray(spec, np.float32), axis=1)
        return F[int(d.argmin())]
    if spec in ("moving", "static"):
        cls = 1 if spec == "moving" else 0
        lab = np.rint(data[:, label_col]).astype(int)
        m = lab == cls
        if not m.any():
            raise ValueError(f"no '{spec}' (class {cls}) points in col {label_col} -- "
                             f"cannot build a prototype on this frame.")
        return F[m].mean(0)
    raise ValueError(f"bad reference spec: {spec!r} (use int / (x,y,z) / 'moving' / 'static')")


def color_similarity(data: np.ndarray, ref_vec: np.ndarray, feat_start: int = 8,
                     cmap: str = "inferno", norm: str = "percentile",
                     whiten: bool = True) -> np.ndarray:
    """Cosine similarity of each point's feature to ref_vec -> colormap (bright=similar).

    This is the honest "are the features good?" view: a *good* feature makes a whole
    object (and other instances of it) light up together. PCA-RGB cannot show this
    because it keeps only ~15% of the feature variance.

    whiten: the feature block is variance-ordered, so the top 1-2 PCA axes (geometry:
    range/height) otherwise dominate the dot product and make EVERYTHING look similar.
    Standardizing each axis to unit variance lets all axes contribute equally, so the
    similarity reflects the learned pattern rather than raw geometry. Strongly
    recommended; pass whiten=False to see the raw (geometry-dominated) behaviour.
    """
    F = _features(data, feat_start).astype(np.float32)
    ref = np.asarray(ref_vec, dtype=np.float32).reshape(-1)
    if whiten:
        mu = F.mean(0)
        sd = F.std(0) + 1e-6
        F = (F - mu) / sd
        ref = (ref - mu) / sd
    fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    sim = fn @ (ref / (np.linalg.norm(ref) + 1e-9))     # cosine in [-1, 1]
    s = normalize(sim, norm)
    cm = _get_cmap(cmap)
    if cm is None:
        return np.stack([s, s, s], axis=1)
    return np.asarray(cm(s))[:, :3]


def build_colors(data: np.ndarray, cfg) -> np.ndarray:
    mode = cfg["mode"]
    if mode == "rgb":
        return color_rgb(data, cfg["color_cols"], cfg.get("gamma", 1.0))
    if mode == "feature":
        return color_feature(data, cfg["feat_col"], cfg.get("norm", "percentile"),
                             cfg.get("cmap", "viridis"))
    if mode == "label":
        return color_label(data, cfg["label_col"])
    if mode == "confusion":
        return color_confusion(data, cfg["gt_col"], cfg["pred_col"])
    if mode == "time":
        return color_time(data, cfg.get("time_col", 6), cfg.get("time_cmap", "coolwarm"))
    if mode == "similarity":
        # ref is resolved in visualize.main (it may need an interactive pick), then
        # stashed in cfg["_ref_vec"]; fall back to the 'moving' prototype.
        ref = cfg.get("_ref_vec")
        if ref is None:
            ref = reference_vector(data, cfg.get("ref", "moving"),
                                   cfg.get("feat_start", 8), cfg.get("label_col", 7))
        return color_similarity(data, ref, cfg.get("feat_start", 8),
                                cfg.get("sim_cmap", "inferno"), cfg.get("norm", "percentile"),
                                cfg.get("whiten", True))
    raise ValueError(f"unknown mode '{mode}' (rgb|feature|label|confusion|time|similarity)")
