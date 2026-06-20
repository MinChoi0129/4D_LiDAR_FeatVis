#!/usr/bin/env python
"""4D-LiDAR feature / MOS visualizer (standalone, Open3D).

Reads an (N, C) npy (cols 0:3 = xyz) and renders it colored by one of several
modes (see coloring.py). Built for the 4D-Utonia PCA-RGB feature dumps
([x,y,z,pca_r,pca_g,pca_b]) and for MOS prediction/confusion dumps.

Usage
-----
    # interactive 3D (needs a display)
    python visualize.py --npy examples/feat_seq08_4017.npy --mode rgb

    # headless: save a top-down (BEV) PNG instead
    python visualize.py --npy examples/feat_seq08_4017.npy --mode rgb --bev out.png

Config defaults live in config.yaml; any CLI flag overrides it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import coloring
from loader import crop_range, flatten_z, load_npy

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    ap = argparse.ArgumentParser(description="4D-LiDAR feature/MOS visualizer")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--npy", help="path to the (N,C) npy")
    ap.add_argument("--mode", choices=["rgb", "feature", "label", "confusion", "time"])
    ap.add_argument("--time-col", type=int, help="time mode: signed-seconds column (default 6)")
    ap.add_argument("--time-cmap", help="time mode colormap (default coolwarm)")
    ap.add_argument("--color-cols", type=int, nargs=3, help="rgb mode: 3 feature columns")
    ap.add_argument("--gamma", type=float, help="rgb mode: gamma (<1 brightens)")
    ap.add_argument("--feat-col", type=int, help="feature mode: scalar column")
    ap.add_argument("--norm", choices=["percentile", "minmax", "sigmoid"])
    ap.add_argument("--cmap", help="feature mode colormap (default viridis)")
    ap.add_argument("--label-col", type=int)
    ap.add_argument("--gt-col", type=int)
    ap.add_argument("--pred-col", type=int)
    ap.add_argument("--range-crop", type=float, nargs=2, metavar=("RMIN", "RMAX"))
    ap.add_argument("--no-range-crop", action="store_true")
    ap.add_argument("--flatten-2d", action="store_true", help="squash z (BEV-like)")
    ap.add_argument("--point-size", type=float)
    ap.add_argument("--bev", help="save a top-down PNG to this path instead of a 3D window")
    return ap.parse_args()


def merge_cfg(args):
    cfg = dict(
        npy_file=os.path.join(HERE, "examples", "feat_seq08_4017.npy"),
        mode="rgb", color_cols=[3, 4, 5], gamma=1.0,
        feat_col=3, norm="percentile", cmap="viridis", time_col=6, time_cmap="coolwarm",
        label_col=7, gt_col=7, pred_col=8,
        range_crop=[0.0, 60.0], flatten_2d=False,
        point_size=2.0, background=[0.05, 0.05, 0.08],
    )
    cfg.update(load_config(args.config))
    # CLI overrides
    o = {
        "npy_file": args.npy, "mode": args.mode, "color_cols": args.color_cols,
        "gamma": args.gamma, "feat_col": args.feat_col, "norm": args.norm,
        "cmap": args.cmap, "label_col": args.label_col, "gt_col": args.gt_col,
        "pred_col": args.pred_col, "point_size": args.point_size,
        "time_col": args.time_col, "time_cmap": args.time_cmap,
    }
    cfg.update({k: v for k, v in o.items() if v is not None})
    if args.range_crop is not None:
        cfg["range_crop"] = args.range_crop
    if args.no_range_crop:
        cfg["range_crop"] = None
    if args.flatten_2d:
        cfg["flatten_2d"] = True
    cfg["bev"] = args.bev or cfg.get("bev")
    return cfg


def print_legend(mode):
    if mode in ("label", "pred"):
        print("legend: moving=green, static=gray, ignore/unlabeled=dim")
    elif mode == "confusion":
        print("legend: TP=green, FP=blue, FN=red, TN=gray, ignore=dim")
    elif mode == "rgb":
        print("color = PCA(features)->RGB: similar color == similar learned representation")
    elif mode == "feature":
        print("color = viridis(scalar): dark purple=low ... yellow=high (one feature axis)")
    elif mode == "time":
        print("color = sweep time: past=blue ... now=white ... future=red (4D motion trails)")


def save_bev(xyz, colors, out, point_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    order = np.argsort(xyz[:, 2])  # draw low points first
    ax.scatter(xyz[order, 0], xyz[order, 1], c=np.clip(colors[order], 0, 1),
               s=point_size, edgecolors="none")
    ax.set_aspect("equal"); ax.set_facecolor("black")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, facecolor="black")
    print(f"saved BEV image -> {out}")


def show_o3d(xyz, colors, cfg):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="4D-LiDAR FeatVis", width=1440, height=900)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0))
    opt = vis.get_render_option()
    opt.point_size = float(cfg["point_size"])
    opt.background_color = np.asarray(cfg["background"], dtype=np.float64)
    print("[controls] drag=rotate, scroll=zoom, shift+drag=pan, q/esc=quit")
    vis.run()
    vis.destroy_window()


def main():
    cfg = merge_cfg(parse_args())
    data = load_npy(cfg["npy_file"])  # noqa: E501
    print(f"loaded {data.shape} from {cfg['npy_file']} | mode={cfg['mode']}")
    if cfg["range_crop"]:
        data = crop_range(data, cfg["range_crop"][0], cfg["range_crop"][1])
        print(f"range-cropped to {cfg['range_crop']} m -> {data.shape[0]} pts")
    colors = coloring.build_colors(data, cfg)
    xyz = data[:, :3].copy()
    if cfg["flatten_2d"]:
        xyz = flatten_z(xyz)
    print_legend(cfg["mode"])
    if cfg.get("bev"):
        save_bev(xyz, colors, cfg["bev"], cfg["point_size"])
    else:
        show_o3d(xyz, colors, cfg)


if __name__ == "__main__":
    main()
