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
    ap.add_argument("--mode", choices=["rgb", "feature", "label", "confusion", "time", "similarity"])
    ap.add_argument("--time-col", type=int, help="time mode: signed-seconds column (default 6)")
    ap.add_argument("--time-cmap", help="time mode colormap (default coolwarm)")
    # similarity mode
    ap.add_argument("--ref", choices=["moving", "static"],
                    help="similarity: reference = the mean feature of this GT class (default moving)")
    ap.add_argument("--ref-idx", type=int, help="similarity: reference = this point index")
    ap.add_argument("--ref-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="similarity: reference = the point nearest this xyz")
    ap.add_argument("--pick", action="store_true",
                    help="similarity: shift+click a reference point in a picker window first")
    ap.add_argument("--feat-start", type=int, help="similarity: first feature column (default 8)")
    ap.add_argument("--sim-cmap", help="similarity colormap (default turbo)")
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
        ref="moving", feat_start=8, sim_cmap="turbo",
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
        "ref": args.ref, "ref_idx": args.ref_idx, "ref_xyz": args.ref_xyz,
        "feat_start": args.feat_start, "sim_cmap": args.sim_cmap,
    }
    cfg.update({k: v for k, v in o.items() if v is not None})
    cfg["pick"] = args.pick
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
    elif mode == "similarity":
        print("color = cosine similarity to the reference feature: bright=similar, dark=different")


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


def pick_reference_idx(data, cfg):
    """Open an Open3D editing window; return the index of the first shift+clicked point."""
    import open3d as o3d
    base = coloring.color_rgb(data, cfg.get("color_cols", [3, 4, 5]), 1.0) \
        if data.shape[1] > 5 else np.full((len(data), 3), 0.6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(base, 0, 1).astype(np.float64))
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="PICK a reference point: shift+click, then press Q")
    vis.add_geometry(pcd)
    print("[pick] shift+click ONE point (e.g. on a car), then press Q to confirm")
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if not picked:
        raise SystemExit("no point picked -- aborting")
    print(f"[pick] using point index {picked[0]}")
    return picked[0]


def resolve_reference(data, cfg):
    """Reference feature for similarity mode: clicked point > --ref-idx > --ref-xyz > class prototype."""
    if cfg.get("pick"):
        spec = pick_reference_idx(data, cfg)
    elif cfg.get("ref_idx") is not None:
        spec = int(cfg["ref_idx"])
    elif cfg.get("ref_xyz") is not None:
        spec = list(cfg["ref_xyz"])
    else:
        spec = cfg.get("ref", "moving")
    print(f"[similarity] reference = {spec if not isinstance(spec, int) else f'point #{spec}'}")
    return coloring.reference_vector(data, spec, cfg.get("feat_start", 8), cfg.get("label_col", 7))


def main():
    cfg = merge_cfg(parse_args())
    data = load_npy(cfg["npy_file"])  # noqa: E501
    print(f"loaded {data.shape} from {cfg['npy_file']} | mode={cfg['mode']}")
    if cfg["range_crop"]:
        data = crop_range(data, cfg["range_crop"][0], cfg["range_crop"][1])
        print(f"range-cropped to {cfg['range_crop']} m -> {data.shape[0]} pts")
    if cfg["mode"] == "similarity":
        cfg["_ref_vec"] = resolve_reference(data, cfg)
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
