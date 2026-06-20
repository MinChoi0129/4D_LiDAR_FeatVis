"""npy loading + simple scene ops for the 4D-LiDAR feature visualizer.

Expected npy layout (N, C), C >= 3:
    cols 0:3 = xyz
    cols 3:  = features / labels (mode-dependent; see coloring.py)
Our 4D feature dump is (N, 6): [x, y, z, pca_r, pca_g, pca_b].
"""
from __future__ import annotations

import numpy as np


def load_npy(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"expected (N, >=3) array, got {data.shape} from {path}")
    return data.astype(np.float32)


def crop_range(data: np.ndarray, rmin: float, rmax: float) -> np.ndarray:
    """Keep points whose horizontal (xy) distance is within [rmin, rmax] meters."""
    dist = np.linalg.norm(data[:, :2], axis=1)
    keep = (dist >= rmin) & (dist <= rmax)
    return data[keep]


def flatten_z(xyz: np.ndarray) -> np.ndarray:
    """Project to the ground plane (BEV-like) by zeroing z."""
    out = xyz.copy()
    out[:, 2] = 0.0
    return out
