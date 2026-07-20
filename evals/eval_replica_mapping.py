#!/usr/bin/env python3
"""
Evaluate Replica mapping quality.

Metrics:
  Depth AbsRel / RMSE / delta1  — raycasting GT mesh (no EGL required)
  Chamfer / Accuracy / F-score  — point cloud vs GT mesh (no EGL required)
  PSNR / SSIM / LPIPS           — render map.ply via Open3D OffscreenRenderer
                                   (requires libEGL.so.1; skipped gracefully if absent)

Usage:
    python eval_replica_mapping.py \
        --out_dir   logs/office0_run1_w20_nested-giant \
        --scene_dir data/Replica/office0 \
        --cam_params data/Replica/cam_params.json
"""

import argparse
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation


# ── optional LPIPS ────────────────────────────────────────────────────────────
try:
    import torch
    import lpips as lpips_lib
    _lpips_fn = lpips_lib.LPIPS(net="alex").eval()
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def _to_lpips_tensor(img: np.ndarray) -> "torch.Tensor":
    """HxWx3 uint8 → 1x3xHxW float in [-1, 1] (LPIPS input convention)."""
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


# ── trajectory ────────────────────────────────────────────────────────────────

def load_tum_trajectory(path: Path, fps: float = 30.0) -> dict[int, np.ndarray]:
    """Read a TUM trajectory keyed by frame index: round(timestamp * fps).

    Replica timestamps are synthesised as frame_idx / fps, so this recovers
    the index that matches the frameNNNNNN.jpg / depthNNNNNN.png filenames.
    """
    poses = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            ts = float(p[0])
            t = np.array([float(x) for x in p[1:4]])
            q = np.array([float(x) for x in p[4:8]])
            R = Rotation.from_quat(q).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            poses[round(ts * fps)] = c2w
    return poses


# ── depth metrics (raycasting — no EGL) ───────────────────────────────────────

def raycast_depth(
    scene: "o3d.t.geometry.RaycastingScene",
    c2w: np.ndarray,
    W: int, H: int,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Raycast GT mesh → depth map (H, W) float32. 0 = no hit."""
    import open3d.core as o3c
    w2c = np.linalg.inv(c2w)
    intrinsic = o3d.core.Tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3c.float64
    )
    extrinsic = o3d.core.Tensor(w2c, dtype=o3c.float64)
    rays = scene.create_rays_pinhole(intrinsic, extrinsic, W, H)
    result = scene.cast_rays(rays)
    depth = result["t_hit"].numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Standard monocular depth metrics (AbsRel, RMSE, δ<1.25) over valid pixels.

    The prediction is median-scaled to GT first — monocular scale is
    arbitrary, so unscaled errors would mostly measure the scale offset.
    """
    valid = (pred > 0) & (gt > 0) & np.isfinite(pred) & np.isfinite(gt)
    if valid.sum() == 0:
        return {"absrel": None, "rmse": None, "delta1": None}
    p, g = pred[valid], gt[valid]
    scale = float(np.median(g / p))
    p_s = p * scale
    absrel = float(np.mean(np.abs(p_s - g) / g))
    rmse   = float(np.sqrt(np.mean((p_s - g) ** 2)))
    delta1 = float(np.mean(np.maximum(p_s / g, g / p_s) < 1.25))
    return {"absrel": absrel, "rmse": rmse, "delta1": delta1}


# ── mesh loading ─────────────────────────────────────────────────────────────

def load_mesh_as_o3d_legacy(mesh_path: Path) -> "o3d.geometry.TriangleMesh":
    """Load a mesh via trimesh (handles quads/n-gons) and return Open3D legacy mesh."""
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices  = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces,    dtype=np.int32))
    return legacy


def load_mesh_as_o3d_tensor(mesh_path: Path) -> "o3d.t.geometry.TriangleMesh":
    """Load a mesh via trimesh and return Open3D tensor mesh (for raycasting)."""
    return o3d.t.geometry.TriangleMesh.from_legacy(load_mesh_as_o3d_legacy(mesh_path))


# ── trajectory alignment (Sim3) ───────────────────────────────────────────────

def compute_sim3_alignment(est_tum: Path, gt_tum: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Sim3 alignment that maps the estimated trajectory onto the GT.
    Returns (scale, R 3x3, t 3) such that: p_gt ≈ scale * R @ p_est + t
    """
    from evo.core import sync
    from evo.tools import file_interface

    traj_ref = file_interface.read_tum_trajectory_file(str(gt_tum))
    traj_est = file_interface.read_tum_trajectory_file(str(est_tum))
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    _, r, t, s = traj_est.align(traj_ref, correct_scale=True, return_parameters=True)
    return float(s), r, t


def apply_sim3(pcd: o3d.geometry.PointCloud, s: float, R: np.ndarray, t: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a new point cloud with points transformed by s * R @ p + t."""
    pts = np.asarray(pcd.points)
    aligned_pts = s * (R @ pts.T).T + t
    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(aligned_pts)
    if pcd.has_colors():
        aligned.colors = pcd.colors
    return aligned


# ── Chamfer metrics (no EGL) ──────────────────────────────────────────────────

def chamfer_metrics(
    recon_pcd: o3d.geometry.PointCloud,
    mesh_path: Path,
    n_samples: int = 200_000,
    threshold: float = 0.05,
) -> dict:
    """Point-cloud reconstruction metrics against the GT mesh.

    accuracy = mean recon→GT distance, completeness = mean GT→recon distance,
    chamfer = their average; precision/recall/F-score count points within
    `threshold` metres (5 cm — the common indoor-reconstruction cutoff).
    GT points are sampled uniformly from the mesh surface.
    """
    mesh = load_mesh_as_o3d_legacy(mesh_path)
    gt_pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    d_r2g = np.asarray(recon_pcd.compute_point_cloud_distance(gt_pcd))
    d_g2r = np.asarray(gt_pcd.compute_point_cloud_distance(recon_pcd))

    accuracy     = float(d_r2g.mean())
    completeness = float(d_g2r.mean())
    precision    = float((d_r2g < threshold).mean())
    recall       = float((d_g2r < threshold).mean())
    denom = precision + recall
    f_score = float(2 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "accuracy": accuracy, "completeness": completeness,
        "chamfer": (accuracy + completeness) / 2,
        "precision": precision, "recall": recall, "f_score": f_score,
    }


# ── rendering metrics (EGL required) ──────────────────────────────────────────

def try_rendering_metrics(
    map_path: Path,
    scene_dir: Path,
    poses: dict[int, np.ndarray],
    W: int, H: int,
    fx: float, fy: float, cx: float, cy: float,
    point_size: float,
) -> dict:
    """Returns PSNR/SSIM/LPIPS averaged over keyframes, or empty dict on failure."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([0., 0., 0., 1.]))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = "defaultUnlit"
    mat.point_size = point_size
    pcd = o3d.io.read_point_cloud(str(map_path))
    renderer.scene.add_geometry("map", pcd, mat)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    from skimage.metrics import structural_similarity

    for frame_idx, c2w in sorted(poses.items()):
        gt_rgb_path = scene_dir / "results" / f"frame{frame_idx:06d}.jpg"
        if not gt_rgb_path.exists():
            continue
        gt_rgb = cv2.cvtColor(cv2.imread(str(gt_rgb_path)), cv2.COLOR_BGR2RGB)

        w2c = np.linalg.inv(c2w)
        renderer.setup_camera(intrinsic, w2c)
        rendered = np.asarray(renderer.render_to_image())

        mse = np.mean((rendered.astype(np.float32) - gt_rgb.astype(np.float32)) ** 2)
        psnr_vals.append(float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse))))
        ssim_vals.append(float(structural_similarity(rendered, gt_rgb, channel_axis=2, data_range=255)))
        if HAS_LPIPS:
            with torch.no_grad():
                lpips_vals.append(float(_lpips_fn(
                    _to_lpips_tensor(rendered), _to_lpips_tensor(gt_rgb)).item()))

    return {
        "psnr":  round(float(np.mean(psnr_vals)),  3) if psnr_vals  else None,
        "ssim":  round(float(np.mean(ssim_vals)),  4) if ssim_vals  else None,
        "lpips": round(float(np.mean(lpips_vals)), 4) if lpips_vals else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """Score one run's mapping quality; writes <out_dir>/mapping_metrics.json.

    Depth and Chamfer metrics need only raycasting (no EGL); the rendering
    metrics (PSNR/SSIM/LPIPS) need an OffscreenRenderer and are skipped
    gracefully when EGL is unavailable.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--scene_dir",  required=True)
    parser.add_argument("--cam_params", required=True)
    parser.add_argument("--fps",         type=float, default=30.0)
    parser.add_argument("--depth_scale", type=float, default=6553.5)
    parser.add_argument("--point_size",  type=float, default=3.0)
    parser.add_argument("--no_chamfer",  action="store_true")
    parser.add_argument("--no_render",   action="store_true",
                        help="Skip PSNR/SSIM/LPIPS (useful if EGL unavailable)")
    args = parser.parse_args()

    out_dir   = Path(args.out_dir)
    scene_dir = Path(args.scene_dir)
    mesh_path = scene_dir.parent / f"{scene_dir.name}_mesh.ply"

    with open(args.cam_params) as f:
        cam = json.load(f)["camera"]
    W, H = int(cam["w"]), int(cam["h"])
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    traj_path = out_dir / "trajectory_tum.txt"
    map_path  = out_dir / "map.ply"

    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not map_path.exists():
        raise FileNotFoundError(f"map.ply not found — re-run without --skip_ply: {map_path}")

    poses = load_tum_trajectory(traj_path, args.fps)
    print(f"Evaluating {len(poses)} keyframes in {scene_dir.name}")

    results: dict = {}

    # ── depth via raycasting (no EGL) ─────────────────────────────────────────
    if mesh_path.exists():
        print("Raycasting GT mesh for depth metrics...")
        t_mesh = load_mesh_as_o3d_tensor(mesh_path)
        raycast_scene = o3d.t.geometry.RaycastingScene()
        raycast_scene.add_triangles(t_mesh)

        absrel_vals, rmse_vals, delta1_vals = [], [], []
        for frame_idx, c2w in sorted(poses.items()):
            gt_depth = raycast_depth(raycast_scene, c2w, W, H, fx, fy, cx, cy)
            da3_depth_path = scene_dir / "results" / f"depth{frame_idx:06d}.png"
            if not da3_depth_path.exists():
                continue
            sensor_depth = cv2.imread(str(da3_depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            sensor_depth /= args.depth_scale
            dm = depth_metrics(sensor_depth, gt_depth)
            if dm["absrel"] is not None:
                absrel_vals.append(dm["absrel"])
                rmse_vals.append(dm["rmse"])
                delta1_vals.append(dm["delta1"])

        results["depth_absrel"] = round(float(np.mean(absrel_vals)), 4) if absrel_vals else None
        results["depth_rmse"]   = round(float(np.mean(rmse_vals)),   4) if rmse_vals   else None
        results["depth_delta1"] = round(float(np.mean(delta1_vals)), 4) if delta1_vals else None
    else:
        print(f"[warn] Mesh not found: {mesh_path} — skipping depth metrics")

    # ── Chamfer (no EGL) ──────────────────────────────────────────────────────
    if not args.no_chamfer and mesh_path.exists():
        print("Computing Chamfer distance (with Sim3 alignment)...")
        pcd = o3d.io.read_point_cloud(str(map_path))
        gt_tum_path = scene_dir / "gt_tum.txt"
        try:
            s, R, t = compute_sim3_alignment(traj_path, gt_tum_path)
            print(f"  Sim3 alignment: scale={s:.4f}")
            pcd_aligned = apply_sim3(pcd, s, R, t)
        except Exception as e:
            print(f"  [warn] Sim3 alignment failed ({e}), using raw point cloud")
            pcd_aligned = pcd
        cm = chamfer_metrics(pcd_aligned, mesh_path)
        results.update({k: round(v, 5) for k, v in cm.items()})

    # ── rendering metrics (EGL) ───────────────────────────────────────────────
    if not args.no_render:
        print("Attempting rendering metrics (PSNR/SSIM/LPIPS)...")
        try:
            rm = try_rendering_metrics(map_path, scene_dir, poses, W, H, fx, fy, cx, cy, args.point_size)
            results.update(rm)
            print("  Rendering succeeded.")
        except Exception:
            print("  [warn] Rendering failed (EGL unavailable?). Skipping PSNR/SSIM/LPIPS.")
            traceback.print_exc()
            results.update({"psnr": None, "ssim": None, "lpips": None})
    else:
        results.update({"psnr": None, "ssim": None, "lpips": None})

    # ── print + save ──────────────────────────────────────────────────────────
    print("\n── Mapping Metrics ──────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<20} {v}")
    print("─────────────────────────────────────────────────────")

    out_path = out_dir / "mapping_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
