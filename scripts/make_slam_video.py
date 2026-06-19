"""
Render a side-by-side SLAM video: original camera image | reconstructed map.

Left panel  : the original input frames (a directory of frame_<seq>_<ts_ns>.jpg,
              e.g. from scripts/extract_bag_images.py or the bridge spool).
Right panel : the finished point map (map.ply), rendered either as a
              first-person flythrough from the camera's own pose (default) or
              top-down with the trajectory.

The map and trajectory share the SLAM world frame, so no ground truth /
alignment is needed — this visualizes the SLAM output as-is.

Views (--view):
  first_person : project the global point cloud through the optimized camera
                 pose at each frame (poses slerp-interpolated between
                 keyframes). At keyframes this roughly matches the input image.
                 Real intrinsics were not saved, so a pinhole is synthesized
                 from --hfov.
  topdown      : top-down scatter of the map with the trajectory revealed up to
                 the current time and the current camera position marked.

Pure OpenCV + matplotlib (no Open3D / no GPU).

    python scripts/make_slam_video.py \\
        --run_dir outputs/ros_run --image_dir /tmp/drone1_frames \\
        --out plots/slam_video.mp4 --view first_person --hfov 70
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── data loading ──────────────────────────────────────────────────────────────

def read_ply_xyzrgb(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the binary_little_endian xyz+rgb PLY written by SLAMResult.save_ply."""
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rgb = np.column_stack([data["r"], data["g"], data["b"]]).astype(np.uint8)
    return xyz, rgb


def read_tum_poses(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, translations Nx3, quats Nx4 [x,y,z,w]) from TUM."""
    d = np.loadtxt(path, comments="#")
    return d[:, 0].astype(np.float64), d[:, 1:4].astype(np.float64), d[:, 4:8]


def list_frames(image_dir: str) -> tuple[list[Path], np.ndarray]:
    """Sorted frame_<seq>_<ts_ns>.jpg paths and their timestamps (seconds)."""
    paths = sorted(Path(image_dir).glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg in {image_dir}")
    ts = np.array([int(p.stem.split("_")[2]) / 1e9 for p in paths])
    return paths, ts


# ── pose interpolation ──────────────────────────────────────────────────────

class PoseTrack:
    """cam-to-world pose at any time, slerp/lerp-interpolated between keyframes."""

    def __init__(self, ts: np.ndarray, trans: np.ndarray, quats: np.ndarray):
        from scipy.spatial.transform import Rotation, Slerp
        self.ts = ts
        self.trans = trans
        self.rots = Rotation.from_quat(quats)
        self.slerp = Slerp(ts, self.rots)

    def c2w(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        j = int(np.clip(np.searchsorted(self.ts, t), 1, len(self.ts) - 1))
        t0, t1 = self.ts[j - 1], self.ts[j]
        w = (t - t0) / (t1 - t0 + 1e-9)
        tr = (1 - w) * self.trans[j - 1] + w * self.trans[j]
        R = self.slerp(t).as_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = tr
        return M


# ── first-person rendering ────────────────────────────────────────────────────

def render_first_person(xyz: np.ndarray, bgr: np.ndarray, c2w: np.ndarray,
                        W: int, H: int, fx: float, fy: float,
                        splat: int, znear: float) -> np.ndarray:
    """Project the world point cloud through camera pose c2w (OpenCV convention)."""
    w2c = np.linalg.inv(c2w)
    pcam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = pcam[:, 2]
    front = z > znear
    u = fx * pcam[:, 0] / z + W / 2.0
    v = fy * pcam[:, 1] / z + H / 2.0
    m = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Painter's algorithm: draw far points first so near points overwrite.
    order = np.argsort(z[m])[::-1]
    uu = u[m][order].astype(np.int32)
    vv = v[m][order].astype(np.int32)
    cc = bgr[m][order]

    canvas = np.zeros((H, W, 3), np.uint8)
    rng = range(-splat, splat + 1)
    for dy in rng:
        for dx in rng:
            vy, ux = vv + dy, uu + dx
            ok = (vy >= 0) & (vy < H) & (ux >= 0) & (ux < W)
            canvas[vy[ok], ux[ok]] = cc[ok]
    return canvas


# ── top-down rendering ────────────────────────────────────────────────────────

def render_topdown_base(xyz, rgb, plane, size, point_size):
    """Rasterize the top-down map once; return BGR base + world->pixel fn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    a, b = (sorted(np.argsort(xyz.max(0) - xyz.min(0))[-2:])
            if plane == "auto" else axis[plane])
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(xyz[:, a], xyz[:, b], c=rgb / 255.0, s=point_size, edgecolors="none")
    ax.set_aspect("equal")
    lo, hi = xyz[:, [a, b]].min(0), xyz[:, [a, b]].max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    h = int(fig.bbox.height)
    base = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR).copy()
    trans = ax.transData.transform

    def to_px(world_xy):
        disp = trans(world_xy[:, [a, b]])
        disp[:, 1] = h - disp[:, 1]
        return disp.astype(np.int32)

    plt.close(fig)
    return base, to_px


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side SLAM video (image | reconstructed map)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_dir", default="outputs/ros_run")
    ap.add_argument("--map", default=None, help="Override path to map.ply")
    ap.add_argument("--traj", default=None, help="Override path to trajectory_tum.txt")
    ap.add_argument("--image_dir", required=True,
                    help="Dir of frame_<seq>_<ts_ns>.jpg input images")
    ap.add_argument("--out", default="plots/slam_video.mp4")
    ap.add_argument("--view", default="first_person",
                    choices=["first_person", "topdown"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth input frame")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--panel", type=int, default=480, help="Panel height in pixels")
    ap.add_argument("--max_points", type=int, default=800_000,
                    help="Subsample the map to at most this many points")
    # first-person knobs
    ap.add_argument("--hfov", type=float, default=120.0,
                    help="Horizontal field of view in degrees (synthesized pinhole)")
    ap.add_argument("--splat", type=int, default=2,
                    help="Point radius in pixels (0 = single pixel)")
    ap.add_argument("--znear", type=float, default=1e-3)
    # top-down knobs
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--plane", default="auto", choices=["auto", "xy", "xz", "yz"])
    args = ap.parse_args()

    rd = Path(args.run_dir)
    map_path = args.map or str(rd / "map.ply")
    traj_path = args.traj or str(rd / "trajectory_tum.txt")

    xyz, rgb = read_ply_xyzrgb(map_path)
    if len(xyz) > args.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    bgr_pts = rgb[:, ::-1].copy()  # RGB -> BGR for OpenCV canvas
    t_ts, t_xyz, t_quat = read_tum_poses(traj_path)
    paths, img_ts = list_frames(args.image_dir)
    if args.stride > 1:
        paths, img_ts = paths[::args.stride], img_ts[::args.stride]
    if args.max_frames:
        paths, img_ts = paths[:args.max_frames], img_ts[:args.max_frames]
    print(f"map {len(xyz)} pts | traj {len(t_xyz)} poses | "
          f"{len(paths)} frames | view={args.view}")

    # Right-panel geometry (match the input image aspect ratio).
    probe = cv2.imread(str(paths[0]))
    aspect = probe.shape[1] / probe.shape[0]
    rh = args.panel
    rw = int(round(rh * aspect))

    if args.view == "first_person":
        track = PoseTrack(t_ts, t_xyz, t_quat)
        fx = (rw / 2.0) / np.tan(np.radians(args.hfov) / 2.0)
        fy = fx  # square pixels
    else:
        base, to_px = render_topdown_base(xyz, rgb, args.plane, args.panel,
                                          args.point_size)
        traj_px = to_px(t_xyz)
        cv2.polylines(base, [traj_px.reshape(-1, 1, 2)], False, (90, 90, 90), 1,
                      cv2.LINE_AA)
        rh = base.shape[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for i, (p, t) in enumerate(zip(paths, img_ts)):
        left = cv2.imread(str(p))
        if left is None:
            continue
        left = cv2.resize(left, (int(left.shape[1] * rh / left.shape[0]), rh))

        if args.view == "first_person":
            right = render_first_person(xyz, bgr_pts, track.c2w(t), rw, rh,
                                        fx, fy, args.splat, args.znear)
            label = "reconstructed map (first-person)"
        else:
            right = base.copy()
            k = int(np.searchsorted(t_ts, t))
            if k >= 2:
                cv2.polylines(right, [traj_px[:k].reshape(-1, 1, 2)], False,
                              (0, 255, 255), 2, cv2.LINE_AA)
            cx, cy = traj_px[min(k, len(traj_px) - 1)]
            cv2.circle(right, (int(cx), int(cy)), 6, (0, 0, 255), -1, cv2.LINE_AA)
            label = "reconstructed map (top-down)"

        cv2.putText(left, f"input  t={t:.2f}s  #{i}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(right, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.hconcat([left, right])
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(frame)

    if writer is None:
        raise RuntimeError("No frames were written (no readable images?)")
    writer.release()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
