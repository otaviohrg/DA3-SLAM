"""
Is there a DOMAIN-AGNOSTIC target for how much a DA3 batch should span?

THE PROBLEM
-----------
TUM and UAS want opposite settings for keyframe density and submap size.  But
both show a U-shaped optimum in *metres of path covered by one DA3 batch* —
they are the same curve sampled on opposite sides:

    TUM best  1.5-2.5 m/batch   (its grid only reached the over-DENSE side)
    UAS best  5-23 m/batch      (its grid only reached the over-SPARSE side)

The ratio of optima (~5.5x) is close to the ratio of scene depths (a desk at
1-3 m vs tunnel/forest at 5-15 m), which suggests the invariant is not distance
but PARALLAX: baseline / scene depth.  If that holds, one scale-free threshold
serves both domains, and it is already measurable online — optical flow in
pixels IS (baseline / depth) x focal length, and the keyframe selector computes
it already.

WHAT THIS MEASURES
------------------
Per sequence, for each frozen keyframe list (density):

  scene depth      median DA3 depth over a sample of batches (metres)
  baseline         GT distance travelled between consecutive keyframes (metres)
  flow             mean Lucas-Kanade optical flow between consecutive
                   keyframes (pixels) — the online-observable proxy
  parallax         baseline / scene_depth, dimensionless and scale-free

Then, for each (density, submap size) cell that the ATE grids already scored,
it reports metres/batch, flow/batch and parallax/batch, so the ATE optimum in
each domain can be read off in all three units.  If the two domains' optima
agree in `parallax` (or `flow`) but disagree in `metres`, the invariant is
confirmed and the threshold falls out of the data.

Usage:
    python scripts/analyze_parallax.py --out outputs/parallax
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import benchmark_common as bc

# (label, dataset, sequence dir, image dir, GT file, keyframe-list glob)
TUM_ROOT = Path("data/tum")
UAS_CACHE = Path("outputs/sweep/step1_uas_resolution/_seqcache")


def sequences() -> list[dict]:
    out = []
    for name in ("desk", "xyz", "room"):
        seq = TUM_ROOT / f"rgbd_dataset_freiburg1_{name}"
        if seq.exists():
            out.append({"domain": "TUM", "name": f"fr1/{name}",
                        "images": seq / "rgb", "gt": seq / "groundtruth.txt",
                        "kf_dir": Path("outputs/kfgrid_segment/keyframes"),
                        "kf_stem": seq.name, "rows": "outputs/kfgrid_segment/kfgrid_rows.jsonl",
                        "row_seq": seq.name})
    uas = {"fyllingsdalen_tunnel": "data/UAS/fyllingsdalen_tunnel",
           "hornbill": "data/UAS/runehamar_tunnel/hornbill",
           "frozen_lake": "data/UAS/frozen_lake",
           "campus_fog": "data/UAS/campus_fog"}
    for name, seqdir in uas.items():
        images = UAS_CACHE / name / "undistorted"
        gt = Path(seqdir) / "gt_odometry.tum"
        if images.is_dir() and gt.exists():
            out.append({"domain": "UAS", "name": name, "images": images, "gt": gt,
                        "kf_dir": Path("outputs/uas_kf"), "kf_stem": name,
                        "rows": "outputs/uas_kfgrid/rows.jsonl", "row_seq": name})
    return out


def load_keyframes(path: Path) -> list[int]:
    return [int(line.split()[0]) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def image_paths(images: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(str(p) for p in images.iterdir() if p.suffix.lower() in exts)


def mean_flow(path_a: str, path_b: str, max_corners: int = 400) -> float | None:
    """Mean Lucas-Kanade displacement (pixels) between two frames.

    This is the quantity the keyframe selector already accumulates, and it is
    the online-observable stand-in for parallax: flow ~ (baseline/depth) * f.
    """
    a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None
    pts = cv2.goodFeaturesToTrack(a, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) < 10:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21),
                                              maxLevel=3)
    good = status.ravel() == 1
    if good.sum() < 10:
        return None
    return float(np.linalg.norm(nxt[good] - pts[good], axis=2).mean())


def scene_depth(estimator, paths: list[str], keyframes: list[int],
                n_batches: int = 3, batch: int = 8) -> float | None:
    """Median DA3 depth over a few batches — the sequence's scene scale."""
    if not keyframes:
        return None
    medians = []
    for i in range(n_batches):
        start = int(i * max(len(keyframes) - batch, 0) / max(n_batches - 1, 1))
        idxs = keyframes[start:start + batch]
        if len(idxs) < 2:
            continue
        images = [paths[j] for j in idxs if j < len(paths)]
        if len(images) < 2:
            continue
        pred = estimator.infer(images)
        d = pred.depth[np.isfinite(pred.depth) & (pred.depth > 0)]
        if d.size:
            medians.append(float(np.median(d)))
    return float(np.median(medians)) if medians else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/parallax")
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--flow_sample", type=int, default=120,
                   help="Keyframe pairs sampled per list for the flow estimate")
    args = p.parse_args()

    from da3_runner import resolve_model_alias
    from da3_slam.backend.inference.depth_estimator import DepthEstimator

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator(model_id=resolve_model_alias(args.depth_model),
                               process_resolution=504, backbone_dtype="bf16")

    records = []
    for seq in sequences():
        paths = image_paths(seq["images"])
        gt = bc.load_groundtruth(seq["gt"])
        gt_ts = np.array([e[0] for e in gt])
        gt_pos = np.array([np.asarray(e[1])[:3, 3] for e in gt])
        print(f"\n{seq['domain']:<4} {seq['name']:<22} {len(paths)} frames, "
              f"{len(gt)} GT poses")

        for kf_path in sorted(seq["kf_dir"].glob(f"{seq['kf_stem']}__d*.txt")):
            density = kf_path.stem.split("__d")[-1].replace("p", ".")
            keyframes = load_keyframes(kf_path)
            if len(keyframes) < 4:
                continue

            depth = scene_depth(estimator, paths, keyframes)

            # baseline between consecutive keyframes, via GT positions nearest
            # in time to each keyframe's frame index
            def pos_of(idx):
                if idx >= len(paths):
                    return None
                try:
                    ts = float(Path(paths[idx]).stem.split("_")[-1]) * 1e-9 \
                        if seq["domain"] == "UAS" else float(Path(paths[idx]).stem)
                except ValueError:
                    return None
                j = int(np.argmin(np.abs(gt_ts - ts)))
                return gt_pos[j] if abs(gt_ts[j] - ts) < 0.2 else None

            baselines = []
            for a, b in zip(keyframes, keyframes[1:]):
                pa, pb = pos_of(a), pos_of(b)
                if pa is not None and pb is not None:
                    baselines.append(float(np.linalg.norm(pb - pa)))

            step = max(1, len(keyframes) // args.flow_sample)
            flows = []
            for a, b in list(zip(keyframes, keyframes[1:]))[::step]:
                if a < len(paths) and b < len(paths):
                    f = mean_flow(paths[a], paths[b])
                    if f is not None:
                        flows.append(f)

            rec = {
                "domain": seq["domain"], "sequence": seq["name"],
                "row_seq": seq["row_seq"], "density": float(density),
                "n_keyframes": len(keyframes),
                "scene_depth_m": depth,
                "baseline_m": statistics.median(baselines) if baselines else None,
                "flow_px": statistics.median(flows) if flows else None,
            }
            if rec["baseline_m"] and depth:
                rec["parallax"] = rec["baseline_m"] / depth
            records.append(rec)
            print(f"   d={density:<5} kf={len(keyframes):<5} depth={depth if depth else float('nan'):7.2f} m"
                  f"  baseline={rec['baseline_m'] if rec['baseline_m'] else float('nan'):6.3f} m"
                  f"  flow={rec['flow_px'] if rec['flow_px'] else float('nan'):6.1f} px"
                  f"  parallax={rec.get('parallax', float('nan')):.4f}")

    path = out_dir / "parallax.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n  records -> {path}")


if __name__ == "__main__":
    main()
