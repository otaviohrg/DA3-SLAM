"""
Phase 2: WHY fr1/floor's depth is wrong, and WHY fr1/plant retrieves nothing.

Phase 1 (`diagnose_sequence.py`) established that the two losses are unrelated:

  fr1/floor   depth pred/GT 1.512  ->  implied scale 0.661  vs measured 0.651
              A DEPTH failure, confirmed to 1.5%.  But note the SPREAD:
              p10 1.156 -> p90 1.755, a 1.5x range inside one sequence, where
              every control sequence is tight (desk 0.921-1.041).  A CONSTANT
              bias would be absorbed entirely by Sim3 alignment; floor's Sim3
              ATE is still 0.0832, so the damage is the VARIANCE.

  fr1/plant   depth pred/GT 0.944 (healthy), scale 1.004 (near-perfect),
              parallax 0.048 (lowest of five), 164 GT revisits, 0 closures.
              A RETRIEVAL / drift failure.

This script asks the next question for each, and they need different probes.

PROBE A — is floor's bias a LAW or an ODDITY?
---------------------------------------------
Two hypotheses predict floor's over-prediction, and they are distinguishable:

  (A1) RANGE.  DA3's metric depth compresses toward a prior at close range.
       floor's GT median is 0.94 m, the nearest scene of the five.  If true,
       the pred/GT ratio is a function of GT DEPTH and nothing else, so the
       near-range pixels of desk/xyz/room must be inflated by the same factor
       as floor's.  The per-sequence median hides this; binning does not.

  (A2) TEXTURE.  floor sweeps a low-texture plane with no vertical structure,
       so the metric branch has little to condition on and falls back to a
       scene prior (rooms are ~2-3 m).  If true, the ratio is a function of
       LOCAL TEXTURE, and floor's near pixels behave like every other
       sequence's near pixels once texture is controlled for.

Both are measured from exactly the same forward passes phase 1 already ran:
we keep the per-pixel (gt_depth, ratio, local_texture) triples instead of
collapsing to a per-frame median, then bin.  A1 and A2 make opposite
predictions about desk's near-range pixels, so one pass separates them.

PROBE B — is plant's retrieval BLIND or GATED?
----------------------------------------------
"0 loop closures" has three possible causes, and the fix differs for each:

  (B1) BLIND.  SALAD simply does not put genuine revisit pairs close together
       on this sequence, so no candidate ever reaches the gate.  Lowering the
       threshold would only admit false positives.
  (B2) GATED TOO TIGHT.  Genuine revisits DO score well, but above the
       0.6 distance threshold.  Raising the threshold recovers them — and the
       TUM sweep already found 0.80 to be worth -30%.
  (B3) EXCLUDED.  The revisits exist but fall inside `min_submaps_apart`, so
       they are never even compared.

We compute a SALAD descriptor per keyframe, label every keyframe PAIR with
ground truth (revisit = closer than 0.3 m and more than 10 s apart), and
report the distance distribution for genuine revisits against non-revisits,
plus how many genuine revisits fall under each candidate threshold and how
many survive the submap-gap rule.  That splits the three cleanly.

Usage:
    python scripts/diagnose_sequence_phase2.py                  # both probes
    python scripts/diagnose_sequence_phase2.py --probes A
    python scripts/diagnose_sequence_phase2.py --probes B --sequences plant room
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
from diagnose_sequence import (DEPTH_SCALE, TUM_ROOT, load_depth_index, nearest)

# Bin edges in metres.  Chosen so floor's median (0.94) and desk's (1.08) land
# in DIFFERENT bins while both sequences still populate the shared neighbours —
# that overlap is what makes the range hypothesis falsifiable.
DEPTH_BINS = np.array([0.4, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5, 7.0])
# Texture is Laplacian energy in a 15px window, log-spaced because it spans
# orders of magnitude between a blank floor and a cluttered desk.
TEXTURE_BINS = np.array([0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e9])
PIXEL_STRIDE = 6            # subsample; ~14k samples/frame is ample for medians


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", nargs="+",
                   default=["floor", "plant", "desk", "xyz", "room"])
    p.add_argument("--probes", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--keyframes_dir", default="outputs/kfgrid_segment/keyframes")
    p.add_argument("--batches", type=int, default=8,
                   help="DA3 batches sampled per sequence (probe A)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--kf_stride", type=int, default=8,
                   help="Fallback keyframe stride when no frozen list exists")
    # The gate values the system actually ships / the TUM sweep favoured.
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.6, 0.8, 1.0])
    p.add_argument("--min_submaps_apart", type=int, default=2)
    p.add_argument("--submap_size", type=int, default=8)
    p.add_argument("--out", default="outputs/seq_diag")
    return p.parse_args()


# ── shared sequence loading ───────────────────────────────────────────────────

def load_sequence(name: str, args) -> tuple[Path, list[str], list[int]]:
    """Resolve a sequence to its directory, RGB paths and keyframe indices."""
    seq = TUM_ROOT / f"rgbd_dataset_freiburg1_{name}"
    if not seq.exists():
        raise SystemExit(f"missing sequence directory: {seq}")
    paths = sorted(str(p) for p in (seq / "rgb").glob("*.png"))

    # Same frozen lists phase 1 used, so both phases analyse the frames the
    # system actually ran on.
    frozen = Path(args.keyframes_dir) / f"{seq.name}__d1.txt"
    if frozen.exists():
        keyframes = [int(l.split()[0]) for l in frozen.read_text().splitlines()
                     if l.strip() and not l.startswith("#")]
    else:
        print(f"  [note] {name}: no frozen list, using stride {args.kf_stride}")
        keyframes = list(range(0, len(paths), args.kf_stride))
    keyframes = [k for k in keyframes if k < len(paths)]
    return seq, paths, keyframes


def keyframe_gt_poses(seq: Path, paths: list[str], keyframes: list[int]):
    """(indices, timestamps, 4x4 GT poses) for keyframes with a GT match."""
    gt = bc.load_groundtruth(seq / "groundtruth.txt")
    gt_ts = np.array([e[0] for e in gt])
    gt_T = [np.asarray(e[1]) for e in gt]

    idxs, times, poses = [], [], []
    for k in keyframes:
        if k >= len(paths):
            continue
        try:
            ts = float(Path(paths[k]).stem)
        except ValueError:
            continue
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) < 0.05:
            idxs.append(k)
            times.append(gt_ts[j])
            poses.append(gt_T[j])
    return idxs, np.array(times), poses


# ── probe A: what predicts the depth bias? ────────────────────────────────────

def local_texture(image_bgr: np.ndarray) -> np.ndarray:
    """Laplacian energy in a local window — high on structure, ~0 on a plane.

    This is the quantity the metric branch has to condition on.  A blank floor
    gives it nothing, and the hypothesis is that it then falls back on a scene
    prior rather than on the image.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    # Mean of the squared response over a 15px box = local variance proxy.
    return cv2.blur(lap * lap, (15, 15))


def probe_a(estimator, name: str, seq: Path, paths: list[str],
            keyframes: list[int], args) -> dict:
    """Collect per-pixel (gt_depth, ratio, texture) instead of a frame median."""
    depth_index = load_depth_index(seq)
    gts, ratios, texs = [], [], []

    n_batches, batch = args.batches, args.batch
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
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if gt is None or bgr is None:
                continue
            gt_m = gt.astype(np.float32) / DEPTH_SCALE
            pr = pred.depth[k]
            if pr.shape != gt_m.shape:
                pr = cv2.resize(pr, (gt_m.shape[1], gt_m.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            tex = local_texture(bgr)

            s = PIXEL_STRIDE
            g, p, t = gt_m[::s, ::s], pr[::s, ::s], tex[::s, ::s]
            valid = (g > 0.1) & (g < 10.0) & np.isfinite(p) & (p > 0)
            if valid.sum() < 200:
                continue
            gts.append(g[valid])
            ratios.append((p[valid] / g[valid]))
            texs.append(t[valid])

    if not gts:
        return {}
    gts = np.concatenate(gts)
    ratios = np.concatenate(ratios)
    texs = np.concatenate(texs)

    def binned(values, edges):
        out = []
        who = np.digitize(values, edges) - 1
        for b in range(len(edges) - 1):
            m = who == b
            out.append({
                "lo": float(edges[b]), "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "ratio_median": float(np.median(ratios[m])) if m.sum() > 200
                                else None,
            })
        return out

    return {
        "sequence": f"fr1/{name}",
        "n_pixels": int(gts.size),
        "ratio_median_overall": float(np.median(ratios)),
        "gt_depth_median": float(np.median(gts)),
        "texture_median": float(np.median(texs)),
        "by_depth": binned(gts, DEPTH_BINS),
        "by_texture": binned(texs, TEXTURE_BINS),
    }


def report_a(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("PROBE A — what predicts the depth bias?")
    print("=" * 78)

    print("\n  pred/GT ratio BINNED BY GROUND-TRUTH DEPTH (metres)")
    print("  hypothesis A1 (range): every sequence's column should agree "
          "within a row.")
    hdr = "  " + f"{'sequence':<12}"
    for lo, hi in zip(DEPTH_BINS, DEPTH_BINS[1:]):
        hdr += f"{lo:g}-{hi:g}".rjust(10)
    print(hdr)
    for r in records:
        row = "  " + f"{r['sequence']:<12}"
        for b in r["by_depth"]:
            row += ("  n/a" if b["ratio_median"] is None
                    else f"{b['ratio_median']:.2f}").rjust(10)
        print(row)

    print("\n  pred/GT ratio BINNED BY LOCAL TEXTURE (Laplacian energy)")
    print("  hypothesis A2 (texture): the bias should fall as texture rises, "
          "and floor should\n  stop being an outlier once texture is matched.")
    hdr = "  " + f"{'sequence':<12}"
    for lo, hi in zip(TEXTURE_BINS, TEXTURE_BINS[1:]):
        label = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        hdr += label.rjust(10)
    print(hdr)
    for r in records:
        row = "  " + f"{r['sequence']:<12}"
        for b in r["by_texture"]:
            row += ("  n/a" if b["ratio_median"] is None
                    else f"{b['ratio_median']:.2f}").rjust(10)
        print(row)

    print("\n  context")
    print(f"  {'sequence':<12}{'overall':>10}{'GT depth':>10}{'texture':>10}")
    for r in records:
        print(f"  {r['sequence']:<12}{r['ratio_median_overall']:>10.3f}"
              f"{r['gt_depth_median']:>10.2f}{r['texture_median']:>10.1f}")


# ── probe B: blind, gated, or excluded? ───────────────────────────────────────

def probe_b(detector, name: str, seq: Path, paths: list[str],
            keyframes: list[int], args) -> dict:
    """SALAD distance on ground-truth-labelled keyframe pairs."""
    import torch
    from PIL import Image as PILImage

    idxs, times, poses = keyframe_gt_poses(seq, paths, keyframes)
    if len(idxs) < 4:
        return {}

    # Descriptors exactly as the pipeline computes them: 224x224, ImageNet
    # normalisation, L2-normalised output.  Batched to keep memory flat.
    feats = []
    with torch.no_grad():
        for i in range(0, len(idxs), 32):
            chunk = idxs[i:i + 32]
            tens = torch.stack([
                detector._TRANSFORM(
                    PILImage.fromarray(cv2.cvtColor(
                        cv2.imread(paths[k], cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB)))
                for k in chunk
            ]).to(detector.device)
            f = detector._model(tens).cpu().numpy().astype(np.float32)
            feats.append(f)
    F = np.concatenate(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8

    pos = np.array([T[:3, 3] for T in poses])
    D_space = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    D_time = np.abs(times[:, None] - times[None, :])
    # SALAD descriptors are ~8k-dimensional, so the broadcast form would
    # allocate ~1 GB.  They are L2-normalised, so ||a-b||^2 = 2 - 2 a.b exactly.
    D_desc = np.sqrt(np.maximum(2.0 - 2.0 * (F @ F.T), 0.0))

    # The pipeline compares FRAMES but gates on SUBMAP separation, so the
    # exclusion rule has to be evaluated in submap index, not keyframe index.
    submap_of = np.array([i // args.submap_size for i in range(len(idxs))])
    gap_ok = np.abs(submap_of[:, None] - submap_of[None, :]) > args.min_submaps_apart

    iu = np.triu_indices(len(idxs), 1)
    revisit = (D_space < 0.3) & (D_time > 10.0)
    pos_mask = revisit[iu]
    neg_mask = (~revisit & (D_time > 10.0))[iu]
    d = D_desc[iu]
    gap = gap_ok[iu]

    out = {
        "sequence": f"fr1/{name}",
        "n_keyframes": len(idxs),
        "n_revisit_pairs": int(pos_mask.sum()),
        "n_revisit_pairs_after_gap": int((pos_mask & gap).sum()),
        "neg_dist_p01": float(np.percentile(d[neg_mask], 1)) if neg_mask.any() else None,
        "neg_dist_median": float(np.median(d[neg_mask])) if neg_mask.any() else None,
    }
    if pos_mask.any():
        out.update({
            "pos_dist_min": float(d[pos_mask].min()),
            "pos_dist_p10": float(np.percentile(d[pos_mask], 10)),
            "pos_dist_median": float(np.median(d[pos_mask])),
        })
        # How many genuine revisits each candidate threshold would admit,
        # AFTER the submap-gap rule the pipeline applies first.
        admits = {}
        for th in args.thresholds:
            admits[f"{th:g}"] = int((pos_mask & gap & (d < th)).sum())
        out["revisits_admitted"] = admits
        # And how many NON-revisits it would admit — the cost side.
        out["false_admitted"] = {
            f"{th:g}": int((neg_mask & gap & (d < th)).sum())
            for th in args.thresholds}
    return out


def report_b(records: list[dict], thresholds: list[float]) -> None:
    print("\n" + "=" * 78)
    print("PROBE B — is plant's retrieval BLIND, GATED, or EXCLUDED?")
    print("=" * 78)

    print(f"\n  {'sequence':<12}{'revisits':>10}{'after gap':>11}"
          f"{'best d':>9}{'p10 d':>9}{'med d':>9}{'neg p01':>9}")
    for r in records:
        if "pos_dist_min" not in r:
            print(f"  {r['sequence']:<12}{r['n_revisit_pairs']:>10}"
                  f"{r['n_revisit_pairs_after_gap']:>11}"
                  f"{'—':>9}{'—':>9}{'—':>9}"
                  f"{r['neg_dist_p01']:>9.3f}")
            continue
        print(f"  {r['sequence']:<12}{r['n_revisit_pairs']:>10}"
              f"{r['n_revisit_pairs_after_gap']:>11}"
              f"{r['pos_dist_min']:>9.3f}{r['pos_dist_p10']:>9.3f}"
              f"{r['pos_dist_median']:>9.3f}{r['neg_dist_p01']:>9.3f}")

    print("\n  genuine revisits ADMITTED at each distance threshold "
          "(false positives in brackets)")
    hdr = "  " + f"{'sequence':<12}"
    for th in thresholds:
        hdr += f"d<{th:g}".rjust(16)
    print(hdr)
    for r in records:
        if "revisits_admitted" not in r:
            continue
        row = "  " + f"{r['sequence']:<12}"
        for th in thresholds:
            k = f"{th:g}"
            row += f"{r['revisits_admitted'][k]} [{r['false_admitted'][k]}]".rjust(16)
        print(row)

    print("\n  READING IT")
    print("    BLIND    -> best d on genuine revisits sits at or above the "
          "negatives' p01:")
    print("                SALAD cannot separate them, and no threshold helps.")
    print("    GATED    -> revisits score well below the negatives but above "
          "0.6: raising")
    print("                the threshold recovers them at a countable false-"
          "positive cost.")
    print("    EXCLUDED -> 'after gap' collapses to ~0: the pairs are never "
          "compared at all.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {n: load_sequence(n, args) for n in args.sequences}

    if "A" in args.probes:
        from da3_runner import resolve_model_alias
        from da3_slam.backend.inference.depth_estimator import DepthEstimator
        # Identical to phase 1, so the two phases' numbers are comparable.
        estimator = DepthEstimator(model_id=resolve_model_alias(args.depth_model),
                                   process_resolution=504, backbone_dtype="bf16")
        records = []
        for name, (seq, paths, keyframes) in loaded.items():
            print(f"=== probe A: fr1/{name}")
            rec = probe_a(estimator, name, seq, paths, keyframes, args)
            if rec:
                records.append(rec)
        (out_dir / "phase2_depth.json").write_text(json.dumps(records, indent=2))
        report_a(records)
        del estimator

    if "B" in args.probes:
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector
        from da3_slam.config import load_slam_config
        cfg = load_slam_config("config/default.yaml")
        detector = LoopClosureDetector(cfg.loop_closure, builder=None)
        records = []
        for name, (seq, paths, keyframes) in loaded.items():
            print(f"=== probe B: fr1/{name}")
            rec = probe_b(detector, name, seq, paths, keyframes, args)
            if rec:
                records.append(rec)
        (out_dir / "phase2_retrieval.json").write_text(json.dumps(records, indent=2))
        report_b(records, args.thresholds)


if __name__ == "__main__":
    main()
