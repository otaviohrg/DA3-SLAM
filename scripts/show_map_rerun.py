"""
Show a saved DA3-SLAM map in Rerun, with the TUM trajectory as a rainbow path.

The path runs red (first pose) -> violet (last pose), so direction and revisits
are readable at a glance.  Every `--arrow_every`-th pose also gets an arrow
along the camera's viewing direction, and a "pose" timeline animates the
current camera along the path.

Usage:
    python scripts/show_map_rerun.py ~/lab_map_11                 # spawn a viewer
    python scripts/show_map_rerun.py ~/lab_map_11 --viewer connect --addr rerun+http://127.0.0.1:9876/proxy
    python scripts/show_map_rerun.py ~/lab_map_11 --viewer save --save lab11.rrd
    python scripts/show_map_rerun.py --ply map.ply --tum trajectory_tum.txt

Aerial PNG with a transparent background (same map + rainbow path, no Rerun):
    python scripts/show_map_rerun.py ~/lab_map_11 --image lab11.png               # oblique, 40° tilt
    python scripts/show_map_rerun.py ~/lab_map_11 --image lab11_plan.png --tilt 0 # straight down
    python scripts/show_map_rerun.py ~/lab_map_11 --image lab11.png --azimuth 180 --max_height 0.8
"""

import argparse
from pathlib import Path

import numpy as np

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}


def load_ply(path: Path, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Memory-map a binary little-endian PLY and randomly keep <= max_points.

    (visualize_map.load_ply unpacks row by row, far too slow for 10M+ points.)
    """
    with open(path, "rb") as f:
        n, props, fmt = 0, [], None
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, ty, name = line.split()
                props.append((name, "<" + _PLY_TYPES[ty]))
            if line == "end_header":
                break
        offset = f.tell()
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: only binary_little_endian PLY is supported (got {fmt})")

    vertices = np.memmap(path, dtype=np.dtype(props), mode="r", offset=offset, shape=(n,))
    if n > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
        vertices = vertices[keep]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    names = vertices.dtype.names
    colors = (np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
              if {"red", "green", "blue"} <= set(names) else None)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], (colors[finite] if colors is not None else None)


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(timestamps, (N,3) positions, (N,4) quaternions x y z w)."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def rainbow(n: int) -> np.ndarray:
    """(n, 3) uint8 colours, hue 0 (red) -> 0.83 (violet)."""
    h = np.linspace(0.0, 0.83, max(n, 1)) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    v, p, q, t = np.ones_like(f), np.zeros_like(f), 1.0 - f, f
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N,4) x y z w -> (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def start_rerun(rr, mode: str, addr: str | None, save: str | None) -> None:
    if mode == "spawn":
        rr.spawn()
    elif mode == "connect":
        fn = getattr(rr, "connect_grpc", None) or getattr(rr, "connect")
        fn(addr) if addr else fn()
    elif mode == "serve":
        fn = getattr(rr, "serve_web", None) or getattr(rr, "serve")
        fn()
    elif mode == "save":
        rr.save(save or "map.rrd")
    else:
        raise ValueError(f"unknown viewer mode: {mode!r}")


# ── offline image (aerial view, transparent background) ──────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def aerial_basis(positions: np.ndarray, rotations: np.ndarray, tilt_deg: float, azimuth_deg: float):
    """Orthographic aerial camera: returns (up, right, screen_up, view_dir).

    `up` is the normal of the trajectory's best-fit plane (a ground robot moves
    in one), signed to agree with the cameras' own up (-y in OpenCV).  If the
    path is not planar enough to trust, the mean camera up is used directly.
    Azimuth 0 lays the path's longest extent horizontally (landscape image).
    """
    cam_up = _normalize(-rotations[:, :, 1].mean(axis=0))
    centred = positions - positions.mean(axis=0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    planar = len(sv) == 3 and sv[2] < 0.25 * sv[1]
    up = vt[2] if planar else cam_up
    up = up if up @ cam_up >= 0 else -up
    major = vt[0] - (vt[0] @ up) * up
    major = _normalize(major if np.linalg.norm(major) > 1e-6 else np.cross(up, [1.0, 0.0, 0.0]))
    other = np.cross(up, major)
    az = np.radians(azimuth_deg)
    forward = _normalize(np.cos(az) * other + np.sin(az) * major)          # right = major at az 0
    tilt = np.radians(tilt_deg)
    view_dir = _normalize(-np.cos(tilt) * up + np.sin(tilt) * forward)     # looking down, leaning forward
    right = _normalize(np.cross(forward, up))
    screen_up = _normalize(np.cross(-view_dir, right))                     # right x screen_up = toward viewer
    return up, right, screen_up, view_dir


def render_aerial(points, colors, positions, rotations, path_colors, out: Path, *,
                  width: int, tilt: float, azimuth: float, max_height: float | None,
                  point_px: int, line_px: float, arrow_every: int, arrow_length: float,
                  supersample: int = 2) -> None:
    import cv2

    up, right, screen_up, view_dir = aerial_basis(positions, rotations, tilt, azimuth)
    origin = positions.mean(axis=0)
    if max_height is not None:
        keep = (points - origin) @ up <= max_height                          # drop the ceiling
        points, colors = points[keep], (colors[keep] if colors is not None else None)
    if colors is None:
        colors = np.full((len(points), 3), 200, np.uint8)

    def project(p):
        c = p - origin
        return np.stack([c @ right, -(c @ screen_up)], axis=-1), c @ view_dir

    uv, depth = project(points)
    tuv, _ = project(positions)
    both = np.vstack([uv, tuv])
    lo, hi = np.percentile(both, 0.2, axis=0), np.percentile(both, 99.8, axis=0)
    lo, hi = np.minimum(lo, tuv.min(0)), np.maximum(hi, tuv.max(0))           # never crop the path
    S = supersample
    margin = 0.03 * (hi - lo).max()
    lo, hi = lo - margin, hi + margin
    scale = width * S / (hi - lo)[0]
    W, H = int(round((hi - lo)[0] * scale)), int(round((hi - lo)[1] * scale))

    img = np.zeros((H, W, 4), np.uint8)                                      # BGRA, fully transparent
    ij = ((uv - lo) * scale).astype(np.int64)
    r = max(1, point_px * S) // 2
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)] if r > 0 else [(0, 0)]
    xs = np.concatenate([ij[:, 0] + dx for dx, _ in offsets])
    ys = np.concatenate([ij[:, 1] + dy for _, dy in offsets])
    ds = np.tile(depth, len(offsets))
    cs = np.tile(np.arange(len(points)), len(offsets))
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ds, cs = xs[ok], ys[ok], ds[ok], cs[ok]
    order = np.argsort(ds, kind="stable")                                    # nearest first
    _, first = np.unique((ys * W + xs)[order], return_index=True)            # nearest point per pixel
    win = order[first]
    img[ys[win], xs[win], :3] = colors[cs[win]][:, ::-1]
    img[ys[win], xs[win], 3] = 255

    pix = ((tuv - lo) * scale)
    lw = max(1, int(round(line_px * S)))
    bgra = [tuple(int(x) for x in c[::-1]) + (255,) for c in path_colors]
    for k in range(len(pix) - 1):
        cv2.line(img, tuple(np.round(pix[k]).astype(int)), tuple(np.round(pix[k + 1]).astype(int)),
                 bgra[k], lw, cv2.LINE_AA)
    for k in range(len(pix)):
        cv2.circle(img, tuple(np.round(pix[k]).astype(int)), max(1, int(lw * 0.8)), bgra[k], -1, cv2.LINE_AA)
    if arrow_every > 0:
        tips, _ = project(positions + rotations[:, :, 2] * arrow_length)
        tips = (tips - lo) * scale
        for k in range(0, len(pix), arrow_every):
            cv2.arrowedLine(img, tuple(np.round(pix[k]).astype(int)), tuple(np.round(tips[k]).astype(int)),
                            bgra[k], max(1, lw // 2), cv2.LINE_AA, tipLength=0.35)

    if S > 1:                                                                # premultiplied downsample: no dark fringes
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        pre = np.concatenate([img[:, :, :3].astype(np.float32) * a, a * 255.0], axis=2)
        pre = cv2.resize(pre, (W // S, H // S), interpolation=cv2.INTER_AREA)
        alpha = pre[:, :, 3:4] / 255.0
        rgb = np.where(alpha > 1e-4, pre[:, :, :3] / np.maximum(alpha, 1e-4), 0)
        img = np.concatenate([np.clip(rgb, 0, 255), pre[:, :, 3:4]], axis=2).round().astype(np.uint8)

    ys_nz, xs_nz = np.nonzero(img[:, :, 3])                                  # crop to content
    if len(xs_nz):
        pad = 8
        img = img[max(ys_nz.min() - pad, 0):ys_nz.max() + pad + 1, max(xs_nz.min() - pad, 0):xs_nz.max() + pad + 1]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise RuntimeError(f"could not write {out}")
    print(f"[image] wrote {out} ({img.shape[1]}x{img.shape[0]}, tilt {tilt:g}°, azimuth {azimuth:g}°)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="Run folder holding map.ply and trajectory_tum.txt")
    ap.add_argument("--ply", help="Map PLY (default: <run_dir>/map.ply)")
    ap.add_argument("--tum", help="TUM trajectory (default: <run_dir>/trajectory_tum.txt)")
    ap.add_argument("--max_points", type=int, default=3_000_000,
                    help="Random subsample of the map (the viewer struggles past a few million)")
    ap.add_argument("--point_radius", type=float, default=0.01, help="Map point radius (scene units)")
    ap.add_argument("--path_radius", type=float, default=0.03, help="Rainbow path line radius")
    ap.add_argument("--arrow_every", type=int, default=5, help="Viewing-direction arrow every N poses (0 = off)")
    ap.add_argument("--arrow_length", type=float, default=0.3)
    ap.add_argument("--viewer", choices=["spawn", "connect", "serve", "save"], default="spawn")
    ap.add_argument("--addr", default=None, help="gRPC URL for --viewer connect")
    ap.add_argument("--save", default=None, help="Output .rrd for --viewer save")
    img = ap.add_argument_group("image export (no Rerun needed)")
    img.add_argument("--image", default=None, help="Render an aerial PNG with transparent background instead")
    img.add_argument("--width", type=int, default=2400, help="Image width in pixels")
    img.add_argument("--tilt", type=float, default=40.0, help="Degrees from straight down (0 = plan view)")
    img.add_argument("--azimuth", type=float, default=0.0, help="Rotate the view about vertical (degrees)")
    img.add_argument("--max_height", type=float, default=1.2,
                     help="Drop points this far above camera height, so the ceiling doesn't hide the map "
                          "(negative = keep everything)")
    img.add_argument("--point_px", type=int, default=2, help="Map point size in pixels")
    img.add_argument("--line_px", type=float, default=4.0, help="Path line width in pixels")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    ply = Path(args.ply).expanduser() if args.ply else (run_dir / "map.ply" if run_dir else None)
    tum = Path(args.tum).expanduser() if args.tum else (run_dir / "trajectory_tum.txt" if run_dir else None)
    if tum is None or not tum.exists():
        ap.error(f"TUM trajectory not found: {tum}")

    if args.image:
        _, positions, quats = load_tum(tum)
        points, colors = (load_ply(ply, args.max_points) if ply is not None and ply.exists()
                          else (np.empty((0, 3), np.float32), None))
        render_aerial(points, colors, positions, quat_to_matrix(quats), rainbow(len(positions)),
                      Path(args.image).expanduser(), width=args.width, tilt=args.tilt, azimuth=args.azimuth,
                      max_height=None if args.max_height < 0 else args.max_height,
                      point_px=args.point_px, line_px=args.line_px,
                      arrow_every=args.arrow_every, arrow_length=args.arrow_length)
        return

    import rerun as rr

    rr.init(f"da3-slam-map-{run_dir.name if run_dir else tum.stem}")
    start_rerun(rr, args.viewer, args.addr, args.save)
    # DA3 uses the OpenCV camera convention (x right, y down, z forward).
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    if ply is not None and ply.exists():
        points, colors = load_ply(ply, args.max_points)
        rr.log("world/map", rr.Points3D(points, colors=colors, radii=args.point_radius), static=True)
        print(f"[map] {len(points):,} points from {ply}")
    else:
        print(f"[map] no PLY at {ply} — showing the trajectory only")

    stamps, positions, quats = load_tum(tum)
    n = len(positions)
    colors = rainbow(n)
    segments = np.stack([positions[:-1], positions[1:]], axis=1).astype(np.float32)
    rr.log("world/trajectory/path",
           rr.LineStrips3D(segments, colors=colors[:-1], radii=args.path_radius), static=True)
    rr.log("world/trajectory/poses",
           rr.Points3D(positions.astype(np.float32), colors=colors, radii=args.path_radius * 1.5),
           static=True)

    rotations = quat_to_matrix(quats)
    if args.arrow_every > 0:
        idx = np.arange(0, n, args.arrow_every)
        rr.log("world/trajectory/heading",
               rr.Arrows3D(origins=positions[idx].astype(np.float32),
                           vectors=(rotations[idx, :, 2] * args.arrow_length).astype(np.float32),
                           colors=colors[idx]),
               static=True)

    # Scrub the "pose" timeline to move a camera marker along the path.
    for k in range(n):
        rr.set_time("pose", sequence=k)
        rr.set_time("time", timestamp=float(stamps[k]))
        rr.log("world/camera", rr.Transform3D(translation=positions[k].astype(np.float32),
                                              mat3x3=rotations[k].astype(np.float32)))
        rr.log("world/camera/marker", rr.Points3D([[0.0, 0.0, 0.0]], colors=[colors[k]],
                                                  radii=args.path_radius * 3))

    path_len = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    print(f"[trajectory] {n} poses, path length {path_len:.2f}, from {tum}")
    if args.viewer == "save":
        print(f"[rerun] wrote {args.save or 'map.rrd'} — open with: rerun {args.save or 'map.rrd'}")


if __name__ == "__main__":
    main()
