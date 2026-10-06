"""
Why does DA3-SLAM lose on fr1/floor and fr1/plant?

THE TWO FAILURES ARE DIFFERENT, AND THAT IS THE POINT
-----------------------------------------------------
Head-to-head on all 9 TUM fr1 sequences, DA3-SLAM (tuned) beats VGGT-SLAM on 7
and loses on 2 — but for unrelated reasons:

  fr1/floor   Sim3 0.0832  SE3 0.4438  scale 0.651   3 loop closures
              Sim3 alignment is `aligned = scale * R * est + t`, so a scale of
              0.651 means the estimate is 1/0.651 = 1.54x TOO LARGE.  DA3-SLAM
              overestimates distance travelled by 54%, and tuning did not move
              it (0.673 -> 0.651).  A METRIC failure.

  fr1/plant   Sim3 0.0664  SE3 0.0665  scale 1.004   0 loop closures
              The scale is essentially perfect.  The trajectory SHAPE is wrong,
              and no loop closure ever fires.  A DRIFT / retrieval failure.

Diagnosing them with the same experiment would confuse the two.  This script
measures the quantities that separate them, against control sequences the
system wins on (desk, xyz, room).

WHAT IT MEASURES
----------------
1. DEPTH BIAS vs TUM's ground-truth depth maps.  DA3's predicted metric depth is
   what sets the trajectory's scale, so if fr1/floor's scale is 1.54x too large,
   the depth should be too — directly, not by inference.  TUM ships registered
   16-bit depth (÷5000 = metres), so this is a measurement, not a proxy.

2. MOTION COMPOSITION from ground truth: rotation vs translation per keyframe,
   and baseline relative to scene depth.  fr1/plant orbits a single object;
   fr1/floor sweeps a plane.  Both are geometrically unusual, in opposite ways.

3. REVISIT STRUCTURE from ground truth: how many genuine revisits exist, so
   "0 loop closures" can be separated into "nothing to find" versus "retrieval
   missed it" — the same distinction that showed fr1/room has 11,121 revisits
   and found 1.

Usage:
    python scripts/diagnose_sequence.py                       # all 5 sequences
    python scripts/diagnose_sequence.py --sequences floor plant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import benchmark_common as bc

TUM_ROOT = Path("data/tum")
DEPTH_SCALE = 5000.0        # TUM 16-bit depth PNG -> metres


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", nargs="+",
                   default=["floor", "plant", "desk", "xyz", "room"],
                   help="fr1 sequence suffixes; the last three are controls "
                        "(DA3-SLAM wins on all of them)")
    p.add_argument("--keyframes_dir", default="outputs/kfgrid_segment/keyframes",
                   help="Frozen keyframe lists, so the frames analysed are the "
                        "ones the system actually used")
    p.add_argument("--batches", type=int, default=6,
                   help="DA3 batches sampled across each sequence")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--out", default="outputs/seq_diag")
    return p.parse_args()


def load_depth_index(seq: Path) -> list[tuple[float, Path]]:
    """(timestamp, path) for every ground-truth depth map."""
    out = []
    txt = seq / "depth.txt"
    if txt.exists():
        for line in txt.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append((float(parts[0]), seq / parts[1]))
    return sorted(out)


def nearest(index: list[tuple[float, Path]], ts: float,
            tol: float = 0.02) -> Path | None:
    if not index:
        return None
    times = np.array([t for t, _ in index])
    j = int(np.argmin(np.abs(times - ts)))
    return index[j][1] if abs(times[j] - ts) <= tol else None


def depth_bias(estimator, seq: Path, keyframes: list[int], paths: list[str],
               depth_index, n_batches: int, batch: int) -> dict:
    """Median ratio of DA3 predicted depth to TUM ground-truth depth.

    Compared only where GT depth is valid (TUM's structured-light depth is 0
    on specular, distant and out-of-range pixels), and per-frame medians are
    aggregated so a few bad pixels cannot dominate.
    """
    ratios, preds, gts = [], [], []
    for i in range(n_batches):
        start = int(i * max(len(keyframes) - batch, 0) / max(n_batches - 1, 1))
        idxs = keyframes[start:start + batch]
        images = [paths[j] for j in idxs if j < len(paths)]
        if len(images) < 2:
            continue
        pred = estimator.infer(images)
        for k, img_path in enumerate(images):
            try:
                ts = float(Path(img_path).stem)
            except ValueError:
                continue
            dpath = nearest(depth_index, ts)
            if dpath is None or not dpath.exists():
                continue
            gt = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if gt is None:
                continue
            gt_m = gt.astype(np.float32) / DEPTH_SCALE
            pr = pred.depth[k]
            if pr.shape != gt_m.shape:
                pr = cv2.resize(pr, (gt_m.shape[1], gt_m.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            valid = (gt_m > 0.1) & (gt_m < 10.0) & np.isfinite(pr) & (pr > 0)
            if valid.sum() < 1000:
                continue
            ratios.append(float(np.median(pr[valid] / gt_m[valid])))
            preds.append(float(np.median(pr[valid])))
            gts.append(float(np.median(gt_m[valid])))
    if not ratios:
        return {}
    return {
        "n_frames_compared": len(ratios),
        "depth_ratio_median": float(np.median(ratios)),
        "depth_ratio_p10": float(np.percentile(ratios, 10)),
        "depth_ratio_p90": float(np.percentile(ratios, 90)),
        "pred_depth_median_m": float(np.median(preds)),
        "gt_depth_median_m": float(np.median(gts)),
    }


def motion_and_revisits(seq: Path, keyframes: list[int], paths: list[str],
                        scene_depth: float | None) -> dict:
    """Rotation/translation per keyframe and GT revisit count."""
    gt = bc.load_groundtruth(seq / "groundtruth.txt")
    gt_ts = np.array([e[0] for e in gt])
    gt_T = [np.asarray(e[1]) for e in gt]

    pose_at = {}
    for k in keyframes:
        if k >= len(paths):
            continue
        try:
            ts = float(Path(paths[k]).stem)
        except ValueError:
            continue
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) < 0.05:
            pose_at[k] = (ts, gt_T[j])

    keys = sorted(pose_at)
    trans, rot = [], []
    for a, b in zip(keys, keys[1:]):
        Ta, Tb = pose_at[a][1], pose_at[b][1]
        rel = np.linalg.inv(Ta) @ Tb
        trans.append(float(np.linalg.norm(rel[:3, 3])))
        c = (np.trace(rel[:3, :3]) - 1) / 2
        rot.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))

    pos = np.array([pose_at[k][1][:3, 3] for k in keys])
    times = np.array([pose_at[k][0] for k in keys])
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    T = np.abs(times[:, None] - times[None, :])
    iu = np.triu_indices(len(keys), 1)
    revisits = int(((D < 0.3) & (T > 10.0))[iu].sum())

    out = {
        "n_keyframes_with_gt": len(keys),
        "median_translation_m": float(np.median(trans)) if trans else None,
        "median_rotation_deg": float(np.median(rot)) if rot else None,
        "path_length_m": float(np.sum(trans)),
        "scene_extent_m": float(np.linalg.norm(pos.max(0) - pos.min(0))),
        "gt_revisit_pairs": revisits,
        "revisit_frac": revisits / max(len(iu[0]), 1),
    }
    if trans and rot:
        # Rotation-dominant motion (orbiting) versus translation-dominant
        # (sweeping) changes how much a batch's views actually overlap.
        out["rot_per_trans_deg_per_m"] = (
            float(np.median(rot) / np.median(trans)) if np.median(trans) > 1e-6
            else None)
    if trans and scene_depth:
        # The scale-free geometry number: baseline over scene depth.
        out["parallax"] = float(np.median(trans) / scene_depth)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from da3_runner import resolve_model_alias
    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    estimator = DepthEstimator(model_id=resolve_model_alias(args.depth_model),
                               process_resolution=504, backbone_dtype="bf16")

    records = []
    for name in args.sequences:
        seq = TUM_ROOT / f"rgbd_dataset_freiburg1_{name}"
        if not seq.exists():
            print(f"  [skip] {name}: {seq} not found")
            continue
        paths = sorted(str(p) for p in (seq / "rgb").iterdir()
                       if p.suffix.lower() in {".png", ".jpg"})
        kf_file = Path(args.keyframes_dir) / f"{seq.name}__d1.txt"
        if kf_file.exists():
            keyframes = [int(l.split()[0]) for l in kf_file.read_text().splitlines()
                         if l.strip() and not l.startswith("#")]
        else:
            keyframes = list(range(0, len(paths), 8))
            print(f"  [note] {name}: no frozen list, using stride 8")
        keyframes = [k for k in keyframes if k < len(paths)]

        print(f"\n=== fr1/{name} — {len(paths)} frames, {len(keyframes)} keyframes")
        depth_index = load_depth_index(seq)
        d = depth_bias(estimator, seq, keyframes, paths, depth_index,
                       args.batches, args.batch)
        m = motion_and_revisits(seq, keyframes, paths,
                                d.get("gt_depth_median_m"))
        rec = {"sequence": f"fr1/{name}", **d, **m}
        records.append(rec)
        if d:
            print(f"  depth  pred/GT ratio {d['depth_ratio_median']:.3f} "
                  f"(p10 {d['depth_ratio_p10']:.3f}, p90 {d['depth_ratio_p90']:.3f})"
                  f"  pred {d['pred_depth_median_m']:.2f} m  "
                  f"GT {d['gt_depth_median_m']:.2f} m  "
                  f"[{d['n_frames_compared']} frames]")
        print(f"  motion trans {m['median_translation_m']:.3f} m  "
              f"rot {m['median_rotation_deg']:.2f}°  "
              f"rot/trans {m.get('rot_per_trans_deg_per_m') or float('nan'):.1f} °/m  "
              f"parallax {m.get('parallax') or float('nan'):.3f}")
        print(f"  scene  extent {m['scene_extent_m']:.2f} m  "
              f"path {m['path_length_m']:.2f} m  "
              f"GT revisits {m['gt_revisit_pairs']}")

    path = out_dir / "sequence_diagnostics.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n  records -> {path}")

    if records:
        print("\n" + "=" * 78)
        print("  DEPTH BIAS vs the Sim3 scale each sequence needed")
        print("=" * 78)
        observed = {"fr1/floor": 0.651, "fr1/plant": 1.004, "fr1/desk": 1.051,
                    "fr1/xyz": 1.148, "fr1/room": 1.089}
        print(f"  {'sequence':<12}{'depth pred/GT':>15}{'implied scale':>15}"
              f"{'measured Sim3 scale':>21}")
        for r in records:
            if "depth_ratio_median" not in r:
                continue
            ratio = r["depth_ratio_median"]
            print(f"  {r['sequence']:<12}{ratio:>15.3f}{1/ratio:>15.3f}"
                  f"{observed.get(r['sequence'], float('nan')):>21.3f}")
        print("\n  If DA3 over-predicts depth by k, the trajectory is k times too")
        print("  large, and Sim3 must apply ~1/k to align it.  Agreement between")
        print("  the last two columns confirms fr1/floor's failure is DEPTH.")


if __name__ == "__main__":
    main()
