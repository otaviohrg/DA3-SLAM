"""
Unified Autonomy Stack (UAS) dataset I/O for the shared benchmark standard.

NTNU-ARL `unified_autonomy_stack_datasets`
(https://huggingface.co/datasets/ntnu-arl/unified_autonomy_stack_datasets):
robot/drone sequences in subterranean, tunnel and outdoor settings, released as
**ROS 1** bags (`sensors_only.bag`) with ground truth in TUM format alongside.

This module is the UAS analog of `euroc_common.py`: it provides the
dataset-specific I/O (bag → frames, fisheye undistortion, calibration lookup)
that the thin `benchmark_uas.py` adapter feeds into the shared scoring path in
`benchmark_common.py`.  Scoring/metrics/output are unchanged and identical to
the other datasets.

Key dataset facts (from the dataset card + calibration YAMLs):
  * Cameras are `sensor_msgs/CompressedImage` @ ~20 Hz.  Platforms:
      - AR-1 (Hornbill): single camera on topic `/cam0/cam0/compressed`
      - UniPilot (AR-2/GR-1/handheld): `/cam_{front,left,right}/image_raw/compressed`
  * Distortion is the **equidistant / Kannala-Brandt (fisheye)** model with 4
    coefficients — undistorted with cv2.fisheye (NOT cv2.undistort, which is the
    pinhole+radtan model used for EuRoC).
  * Ground truth is a TUM-format `.tum` file (timestamp in **seconds**), only
    available for: fyllingsdalen_tunnel, runehamar_tunnel/hornbill, campus_fog,
    frozen_lake.  bc.load_groundtruth() parses it directly.
  * Bag header stamps are nanoseconds → converted to seconds to match the GT.

Reading ROS 1 bags uses the pure-Python `rosbags` package (no ROS install
needed):  pip install rosbags
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


# ── per-sequence camera registry ────────────────────────────────────────────
# Maps a sequence (matched by directory name) to its camera topic and the
# calibration file (relative to the dataset `calibration/` folder).  The same
# intrinsics apply to every sequence recorded on a given platform.
SEQUENCE_CAMERAS: dict[str, tuple[str, str]] = {
    "fyllingsdalen_tunnel": ("/cam0/cam0/compressed", "ar1_cam0.yaml"),
    "frozen_lake":          ("/cam0/cam0/compressed", "ar1_cam0.yaml"),
    "hornbill":             ("/cam0/cam0/compressed", "ar1_cam0.yaml"),
    "campus_fog": ("/cam_front/image_raw/compressed",
                   "unipilot_handheld/cam_front.yaml"),
}

# Sequences for which the dataset ships ground truth.
SEQUENCES_WITH_GT = ("fyllingsdalen_tunnel", "hornbill", "campus_fog", "frozen_lake")


# ── discovery ────────────────────────────────────────────────────────────────

def find_bag(seq_dir: Path) -> Path | None:
    """Locate the ROS 1 bag for a sequence (sensors_only.bag, else any *.bag)."""
    direct = seq_dir / "sensors_only.bag"
    if direct.is_file():
        return direct
    bags = sorted(seq_dir.rglob("*.bag"))
    return bags[0] if bags else None


def find_gt(seq_dir: Path) -> Path | None:
    """Locate the TUM ground-truth file (*.tum) alongside the bag, if any."""
    tums = sorted(seq_dir.rglob("*.tum"))
    return tums[0] if tums else None


def find_calibration_dir(seq_dir: Path) -> Path | None:
    """Walk up from a sequence dir to find the dataset `calibration/` folder."""
    for parent in [seq_dir, *seq_dir.parents]:
        cand = parent / "calibration"
        if cand.is_dir():
            return cand
    return None


def resolve_camera(
    seq_dir: Path, topic: str | None, calib: str | None, calib_dir: Path | None,
) -> tuple[str, Path]:
    """Resolve (camera topic, calibration yaml path) for a sequence.

    Explicit --topic / --calib win; otherwise the sequence is matched against
    SEQUENCE_CAMERAS by directory name (then by any path component).
    """
    reg_topic, reg_calib = None, None
    names = [seq_dir.name, *[p.name for p in seq_dir.parents]]
    for key, (t, c) in SEQUENCE_CAMERAS.items():
        if any(key == n for n in names) or key in str(seq_dir):
            reg_topic, reg_calib = t, c
            break

    final_topic = topic or reg_topic
    if final_topic is None:
        raise ValueError(
            f"Could not infer camera topic for {seq_dir.name}; pass --topic")

    if calib:
        calib_path = Path(calib)
    else:
        if reg_calib is None:
            raise ValueError(
                f"Could not infer calibration for {seq_dir.name}; pass --calib")
        base = calib_dir or find_calibration_dir(seq_dir)
        if base is None:
            raise FileNotFoundError(
                f"No calibration/ folder found near {seq_dir}; pass --calib_dir")
        calib_path = base / reg_calib
    return final_topic, calib_path


# ── calibration ────────────────────────────────────────────────────────────────

def load_calibration(yaml_path: str | Path) -> dict:
    """Parse a UAS camera YAML → {K (3x3), D (n,1), size (w,h), model}.

    Handles the two formats the dataset ships:
      * ROS camera_info (AR1 sequences): row-major `camera_matrix.data`
        (fx,0,cx, 0,fy,cy, 0,0,1), `distortion_coefficients.data`,
        `image_width`/`image_height`.
      * Kalibr (UniPilot sequences, e.g. campus_fog): a single-camera block
        (`cam0`) with `intrinsics` [fx,fy,cx,cy], `distortion_coeffs`,
        `resolution` [w,h].
    Both use the equidistant / Kannala-Brandt (fisheye) distortion model.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if "camera_matrix" in cfg:  # ROS camera_info
        K = np.array(cfg["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.array(cfg["distortion_coefficients"]["data"],
                     dtype=np.float64).reshape(-1, 1)
        size = (int(cfg["image_width"]), int(cfg["image_height"]))
        model = str(cfg.get("distortion_model", "equidistant")).lower()
        return {"K": K, "D": D, "size": size, "model": model}

    # Kalibr: the camera block is nested (usually under `cam0`).
    cam = cfg.get("cam0") or next(
        (v for v in cfg.values() if isinstance(v, dict) and "intrinsics" in v),
        None)
    if cam is None:
        raise ValueError(f"Unrecognised calibration format: {yaml_path}")
    fx, fy, cx, cy = (float(v) for v in cam["intrinsics"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array(cam["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
    w, h = cam["resolution"]
    size = (int(w), int(h))
    model = str(cam.get("distortion_model", "equidistant")).lower()
    return {"K": K, "D": D, "size": size, "model": model}


# ── bag extraction ──────────────────────────────────────────────────────────

def _stamp_ns(msg, record_ns: int) -> int:
    """Capture-time nanoseconds from a message header, falling back to the bag
    record time.  Handles both rosbags ROS1 (.sec/.nanosec) and legacy
    (.secs/.nsecs) header field names."""
    header = getattr(msg, "header", None)
    if header is not None:
        st = header.stamp
        sec = getattr(st, "sec", getattr(st, "secs", None))
        nsec = getattr(st, "nanosec", getattr(st, "nsecs", None))
        if sec is not None and nsec is not None:
            return int(sec) * 1_000_000_000 + int(nsec)
    return int(record_ns)


def extract_bag_frames(
    bag_path: str | Path, topic: str, out_dir: Path,
    max_frames: int | None = None,
) -> tuple[list[str], list[float]]:
    """Decode a CompressedImage topic from a ROS 1 bag → (paths, timestamps_s).

    Frames are written as `frame_<idx:06d>_<ts_ns>.jpg` so timestamps survive in
    the filename; writes are idempotent (an already-populated cache is reused
    without re-reading the bag).  Returns absolute paths and seconds timestamps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: reuse a previously extracted cache (timestamps from filenames).
    cached = sorted(out_dir.glob("frame_*.jpg"))
    if cached:
        paths = [str(p) for p in cached]
        timestamps = [int(p.stem.split("_")[-1]) * 1e-9 for p in cached]
        if max_frames:
            paths, timestamps = paths[:max_frames], timestamps[:max_frames]
        print(f"  Using {len(paths)} cached frame(s) in {out_dir}")
        return paths, timestamps

    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:
        raise ImportError(
            "Reading ROS 1 bags requires the 'rosbags' package: "
            "pip install rosbags") from e

    typestore = get_typestore(Stores.ROS1_NOETIC)
    paths: list[str] = []
    timestamps: list[float] = []

    with Reader(str(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise ValueError(
                f"Topic {topic!r} not in bag; available image-like topics: "
                f"{[t for t in available if 'image' in t or 'cam' in t]}")

        n = 0
        for conn, record_ns, rawdata in reader.messages(connections=conns):
            if max_frames is not None and n >= max_frames:
                break
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # always 3-channel for DA3
            if img is None:
                print(f"  [warn] failed to decode frame {n}, skipping")
                continue
            ts_ns = _stamp_ns(msg, record_ns)
            dst = out_dir / f"frame_{n:06d}_{ts_ns}.jpg"
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(dst))
            timestamps.append(ts_ns * 1e-9)
            n += 1

    print(f"  Extracted {len(paths)} frame(s) from {topic} → {out_dir}")
    return paths, timestamps


# ── undistortion ────────────────────────────────────────────────────────────

def undistort_frames(
    image_paths: list[str], calibration: dict, cache_dir: Path,
) -> list[str]:
    """Undistort frames into a cache dir using the calibration's distortion model.

    Equidistant → cv2.fisheye (Kannala-Brandt); anything else → pinhole+radtan
    cv2.undistort.  Filenames are preserved (timestamp parsing stays valid) and
    writes are idempotent.  Returns the new paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    K, D, size = calibration["K"], calibration["D"], calibration["size"]
    fisheye = calibration["model"] in ("equidistant", "fisheye", "kannala_brandt")

    map1 = map2 = None
    if fisheye:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, size, cv2.CV_16SC2)

    out_paths: list[str] = []
    written = 0
    for src in image_paths:
        dst = cache_dir / Path(src).name
        if not dst.exists():
            image = cv2.imread(src)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {src}")
            if fisheye:
                out = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            else:
                out = cv2.undistort(image, K, D.ravel())
            cv2.imwrite(str(dst), out)
            written += 1
        out_paths.append(str(dst))
    model = "fisheye" if fisheye else "radtan"
    print(f"  {'Undistorted ' + str(written) if written else 'Using cached'} "
          f"{model} frame(s) in {cache_dir}")
    return out_paths
