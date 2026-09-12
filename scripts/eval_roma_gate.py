"""
Can dense matching (RoMa v2) separate TRUE from FALSE loop closures?

WHY
---
Loop-closure retrieval uses one global DINO-SALAD descriptor per frame, which
has no notion of partial overlap and no spatial verification.  Measured
consequences:

  * the distance threshold does double duty ("same place?" AND "how much view
    overlap?"), so it does not transfer between domains — TUM wants 0.80, UAS
    wants 0.60;
  * partial revisits score like non-matches (fr1/room: 11,121 genuine revisits
    in the ground truth, 1 closure detected);
  * loosening the threshold fails because verification is too weak — on UAS at
    threshold 1.00, 70 closures survived the confidence and geometric gates and
    drove ATE from 67 m to 108 m.

If a dense matcher can separate true revisits from aliased ones, retrieval
becomes a pure RECALL knob with a real precision gate behind it, and the
per-domain threshold tuning goes away.

WHAT THIS DOES (offline — no pipeline integration)
--------------------------------------------------
For each sequence it replays the pipeline's own retrieval:

  1. loads the frozen keyframe list actually used by the sweeps;
  2. extracts DINO-SALAD descriptors exactly as LoopClosureDetector does
     (224x224, ImageNet normalisation) and forms every eligible pair;
  3. labels each pair from GROUND TRUTH — a pair is a true revisit when the GT
     camera positions are within `--true_radius` metres and the frames are more
     than `--min_time_gap` seconds apart;
  4. scores each pair with RoMa v2 (number of confident correspondences, and
     the model's own overlap prediction);
  5. reports how well each score separates true from false, as AUC plus the
     best achievable precision/recall — against the SALAD distance as baseline.

The question it answers is narrow and falsifiable: does a dense-matching score
separate true from false revisits better than descriptor distance does?  If it
does not, the idea dies here for the cost of one offline run.

Usage:
    python scripts/eval_roma_gate.py \\
        --sequence data/UAS/fyllingsdalen_tunnel \\
        --images outputs/sweep/step1_uas_resolution/_seqcache/fyllingsdalen_tunnel/undistorted \\
        --keyframes outputs/uas_kf/fyllingsdalen_tunnel__d1.txt \\
        --gt data/UAS/fyllingsdalen_tunnel/gt_odometry.tum \\
        --true_radius 8 --out outputs/roma_gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import benchmark_common as bc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Directory of frames")
    p.add_argument("--keyframes", required=True, help="Frozen keyframe list")
    p.add_argument("--gt", required=True, help="Ground-truth .tum / groundtruth.txt")
    p.add_argument("--name", default=None, help="Label for the report")
    p.add_argument("--out", default="outputs/roma_gate")
    p.add_argument("--true_radius", type=float, default=8.0,
                   help="GT distance (m) below which a pair is a true revisit. "
                        "Scene-scale dependent: ~0.3 indoors, ~8 for UAS")
    p.add_argument("--min_time_gap", type=float, default=10.0,
                   help="Minimum seconds apart, so temporal neighbours are not "
                        "counted as revisits")
    p.add_argument("--max_pairs", type=int, default=400,
                   help="Pairs to score with RoMa (it is the expensive part); "
                        "sampled to balance true and false")
    p.add_argument("--salad_cutoff", type=float, default=1.2,
                   help="Only consider pairs whose SALAD distance is below "
                        "this — i.e. what retrieval could ever propose")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def frame_timestamp(path: Path, uas: bool) -> float | None:
    """UAS caches encode ns in the filename tail; TUM filenames are seconds."""
    try:
        return (int(path.stem.split("_")[-1]) * 1e-9) if uas else float(path.stem)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.images).parts[-2]

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in exts)
    uas = "_" in paths[0].stem and paths[0].stem.startswith("frame_")
    keyframes = [int(l.split()[0]) for l in Path(args.keyframes).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    keyframes = [k for k in keyframes if k < len(paths)]

    gt = bc.load_groundtruth(Path(args.gt))
    gt_ts = np.array([e[0] for e in gt])
    gt_pos = np.array([np.asarray(e[1])[:3, 3] for e in gt])

    # keyframe -> (timestamp, GT position); drop keyframes with no GT nearby
    kf = []
    for k in keyframes:
        ts = frame_timestamp(paths[k], uas)
        if ts is None:
            continue
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) > 0.2:
            continue
        kf.append((k, ts, gt_pos[j]))
    print(f"{name}: {len(paths)} frames, {len(keyframes)} keyframes, "
          f"{len(kf)} with GT")
    if len(kf) < 10:
        raise SystemExit("too few keyframes with ground truth")

    # ── SALAD descriptors, exactly as the pipeline computes them ─────────────
    from da3_slam.backend.processing.loop_closure import LoopClosureDetector
    from da3_slam.config import LoopClosureConfig
    # builder is only used by _verify(), which we never call here.
    detector = LoopClosureDetector(
        LoopClosureConfig(distance_threshold=1.0, min_submaps_apart=1,
                          max_loop_closures=1, min_confidence_ratio=0.0),
        builder=None)
    import cv2
    from types import SimpleNamespace
    images = []
    for k, _, _ in kf:
        bgr = cv2.imread(str(paths[k]))
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # _extract_per_frame_descriptors only reads submap.frames[i].image
    desc = np.stack(detector._extract_per_frame_descriptors(
        SimpleNamespace(frames=[SimpleNamespace(image=im) for im in images])))
    print(f"  descriptors: {desc.shape}")

    # ── candidate pairs + ground-truth labels ────────────────────────────────
    pairs = []
    for a in range(len(kf)):
        for b in range(a + 1, len(kf)):
            if abs(kf[b][1] - kf[a][1]) < args.min_time_gap:
                continue
            d = float(np.linalg.norm(desc[a] - desc[b]))
            if d > args.salad_cutoff:
                continue
            gt_dist = float(np.linalg.norm(kf[b][2] - kf[a][2]))
            pairs.append({"a": a, "b": b, "salad": d, "gt_dist": gt_dist,
                          "true": gt_dist <= args.true_radius})
    n_true = sum(p["true"] for p in pairs)
    print(f"  candidate pairs: {len(pairs)}  (true revisits {n_true}, "
          f"false {len(pairs)-n_true})")
    if n_true < 5 or len(pairs) - n_true < 5:
        raise SystemExit("not enough of both classes to evaluate a gate")

    # balance the sample so AUC is not dominated by whichever class is huge
    trues = [p for p in pairs if p["true"]]
    falses = [p for p in pairs if not p["true"]]
    k = min(args.max_pairs // 2, len(trues), len(falses))
    sample = [trues[i] for i in rng.choice(len(trues), k, replace=False)] + \
             [falses[i] for i in rng.choice(len(falses), k, replace=False)]
    print(f"  scoring {len(sample)} pairs with RoMa ({k} true / {k} false)")

    # ── RoMa scoring ─────────────────────────────────────────────────────────
    from romav2 import RoMaV2
    model = RoMaV2()
    for i, p in enumerate(sample):
        pa, pb = str(paths[kf[p["a"]][0]]), str(paths[kf[p["b"]][0]])
        try:
            preds = model.match(pa, pb)
            matches, overlaps, prec_ab, prec_ba = model.sample(preds, 5000)
            ov = np.asarray(overlaps.detach().cpu() if hasattr(overlaps, "detach")
                            else overlaps, dtype=float)
            p["roma_overlap"] = float(np.mean(ov))
            p["roma_conf_frac"] = float(np.mean(ov > 0.5))
            p["roma_precision"] = float(np.mean([
                float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).mean())
                for x in (prec_ab, prec_ba)]))
        except Exception as exc:
            p["roma_error"] = str(exc)[:120]
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(sample)}")

    scored = [p for p in sample if "roma_overlap" in p]
    print(f"  scored ok: {len(scored)}/{len(sample)}")

    # ── separability ─────────────────────────────────────────────────────────
    def auc(score_key: str, higher_is_true: bool) -> float | None:
        vals = [(p[score_key], p["true"]) for p in scored if score_key in p]
        if not vals:
            return None
        pos = [v for v, t in vals if t]
        neg = [v for v, t in vals if not t]
        if not pos or not neg:
            return None
        wins = sum((a > b) if higher_is_true else (a < b)
                   for a in pos for b in neg)
        ties = sum(a == b for a in pos for b in neg)
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    print(f"\n  SEPARATION (AUC: 0.5 = useless, 1.0 = perfect)")
    results = {}
    for key, higher in (("salad", False), ("roma_overlap", True),
                        ("roma_conf_frac", True), ("roma_precision", True)):
        a = auc(key, higher)
        results[key] = a
        label = {"salad": "SALAD distance (baseline)",
                 "roma_overlap": "RoMa mean overlap",
                 "roma_conf_frac": "RoMa confident fraction",
                 "roma_precision": "RoMa precision"}[key]
        print(f"    {label:<28} {a:.3f}" if a is not None else
              f"    {label:<28} n/a")

    path = out_dir / f"{name}_roma_gate.jsonl"
    with open(path, "w") as f:
        for p in scored:
            f.write(json.dumps(p) + "\n")
    (out_dir / f"{name}_auc.json").write_text(json.dumps(
        {"sequence": name, "n_scored": len(scored), "n_true": k, "auc": results,
         "true_radius": args.true_radius}, indent=2))
    print(f"\n  pairs  -> {path}")


if __name__ == "__main__":
    main()
