"""
Does the boundary depth-ratio's DISPERSION predict its ERROR?

THE CLAIM UNDER TEST
--------------------
`boundary_scale_damping` is a global constant trust level, and it has to be set
per domain: chaining the measured ratios (damping 0.0) is worth -24% on TUM and
-86% on Replica, but +29..58% on UAS, where it drives scale to 0.097.

The proposed domain-independent replacement is variance-weighted shrinkage,

    log s_applied = lambda * log r ,   lambda = sp^2 / (sp^2 + sm^2)

with `sm` measured per boundary instead of assumed per dataset.  That recovers
damping 0.0 where the ratio is reliable and damping 1.0 where it is not, with
no dataset constant — BUT ONLY IF dispersion actually tracks error.  It might
not: a confidently wrong DA3 prediction would be tight AND biased, fooling the
shrinkage exactly when it matters most.  That is what this measures.

WHAT IS MEASURED, PER SUBMAP BOUNDARY
-------------------------------------
  r       the ratio the pipeline uses: median(prev_anchor_depth /
          curr_anchor_depth) over valid pixels — identical to
          `_estimate_depth_scale`, which is the median of exactly this set.
  sm      the dispersion that `_estimate_depth_scale` DISCARDS: the MAD of the
          per-pixel log ratios, in log units.
  r_true  ground truth for that boundary.  Each submap i needs a true metric
          scale s_i, recovered by Sim3-aligning that submap's own DA3 camera
          centres to GT.  The pipeline forms accumulated_scale_i =
          accumulated_scale_{i-1} * delta_i, so the ideal delta is
          r_true = s_i / s_{i-1}.

Then, in log space, three estimators are compared per boundary:
    damping 1.0 (ignore):  err = |log r_true|
    damping 0.0 (apply) :  err = |log r - log r_true|
    shrinkage lambda(sm):  err = |lambda*log r - log r_true|

If dispersion is informative, lambda(sm) beats BOTH fixed extremes on the
pooled set, and the per-domain optimal lambda* correlates with median sm.

Usage:
    python scripts/measure_boundary_dispersion.py \\
        --tum desk floor --replica office0 --uas campus_fog --submap_size 16
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
REPLICA_ROOT = Path("data/Replica")
UAS_ROOT = Path("data/UAS")
UAS_CACHE = Path("outputs/sweep/step1_uas_resolution/_seqcache")
UAS_GT = {
    "fyllingsdalen_tunnel": "fyllingsdalen_tunnel/gt_odometry.tum",
    "hornbill": "runehamar_tunnel/hornbill/gt_odometry.tum",
    "campus_fog": "campus_fog/gt_odometry.tum",
    "frozen_lake": "frozen_lake/gt_odometry.tum",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tum", nargs="*", default=["desk", "floor", "room"])
    p.add_argument("--replica", nargs="*", default=["office0", "office1"])
    p.add_argument("--uas", nargs="*", default=["campus_fog", "fyllingsdalen_tunnel"])
    p.add_argument("--submap_size", type=int, default=16)
    p.add_argument("--max_boundaries", type=int, default=25,
                   help="Cap per sequence; UAS has hundreds and the trend is "
                        "visible well before that")
    p.add_argument("--kf_stride", type=int, default=8)
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--out", default="outputs/boundary_dispersion")
    return p.parse_args()


# ── sequence loading ──────────────────────────────────────────────────────────

def load_tum(name: str):
    seq = TUM_ROOT / f"rgbd_dataset_freiburg1_{name}"
    paths = sorted(str(p) for p in (seq / "rgb").glob("*.png"))
    ts = [float(Path(p).stem) for p in paths]
    return paths, ts, bc.load_groundtruth(seq / "groundtruth.txt"), 0.02


def load_replica(name: str):
    scene = REPLICA_ROOT / name
    paths = sorted(str(p) for p in (scene / "results").glob("frame*.jpg"))
    if not paths:
        paths = sorted(str(p) for p in (scene / "results").glob("frame*.png"))
    # Replica GT is one pose per frame, index-aligned; synthesise 30 fps stamps
    # so the same association path can be reused.
    ts = [i / 30.0 for i in range(len(paths))]
    gt_file = scene / "gt_tum.txt"
    gt = bc.load_groundtruth(gt_file) if gt_file.exists() else []
    return paths, ts, gt, 0.02


def load_uas(name: str):
    d = UAS_CACHE / name / "undistorted"
    paths = sorted(str(p) for p in d.glob("frame_*.jpg"))
    ts = [int(Path(p).stem.split("_")[-1]) * 1e-9 for p in paths]
    return paths, ts, bc.load_groundtruth(UAS_ROOT / UAS_GT[name]), 0.05


# ── per-boundary measurement ──────────────────────────────────────────────────

def submap_true_scale(extrinsics, kf_ts, gt, max_diff) -> tuple:
    """True metric scale of one submap, by PATH LENGTH and (for comparison) Sim3.

    Sim3 alignment over a single submap is ILL-CONDITIONED: measured on GT, the
    camera path inside a 16-keyframe window is near-planar on every dataset
    here (third singular value 0.005-0.035 of the first), so the fitted scale is
    dominated by noise.  A first version of this script used it and produced
    unusable ground truth.

    Path length is well determined under exactly that geometry — it is a scalar
    arc length, and stays meaningful even for perfectly collinear motion — so it
    is the primary estimate.  The Sim3 value is returned alongside purely so the
    two can be compared and the degeneracy shown rather than asserted.
    """
    gt_ts = [e[0] for e in gt]
    pairs = bc.associate(list(kf_ts), gt_ts, max_diff=max_diff)
    if len(pairs) < 3:
        return None, None, None
    est = np.array([extrinsics[ia][:3, 3] for ia, _ in pairs])
    ref = np.array([np.asarray(gt[ib][1])[:3, 3] for _, ib in pairs])

    est_len = float(np.linalg.norm(np.diff(est, axis=0), axis=1).sum())
    ref_len = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    s_path = ref_len / est_len if est_len > 1e-9 else None

    s_sim3 = None
    if np.linalg.norm(est - est.mean(0), axis=1).max() > 1e-6:
        T = bc.sim3_align(est, ref)
        s_sim3 = float(np.linalg.det(T[:3, :3]) ** (1 / 3))

    # Conditioning of this window, so degenerate ones can be excluded rather
    # than silently trusted.
    X = ref - ref.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    cond = float(sv[2] / sv[0]) if sv[0] > 1e-12 and len(sv) > 2 else 0.0
    return s_path, s_sim3, cond


def ratio_and_dispersion(depth_ref, depth_new) -> tuple[float, float, int]:
    """The pipeline's median ratio, plus the dispersion it throws away."""
    if depth_ref.shape != depth_new.shape:
        depth_new = cv2.resize(depth_new, (depth_ref.shape[1], depth_ref.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    valid = ((depth_ref > 0) & (depth_new > 0) &
             np.isfinite(depth_ref) & np.isfinite(depth_new))
    if valid.sum() < 100:
        return 1.0, np.nan, int(valid.sum())
    ratios = depth_ref[valid] / depth_new[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 100:
        return 1.0, np.nan, int(ratios.size)
    logr = np.log(ratios)
    med = float(np.median(logr))
    # MAD -> sigma, the robust dispersion of the SAME set the median came from.
    mad = float(np.median(np.abs(logr - med))) * 1.4826
    return float(np.exp(med)), mad, int(ratios.size)


def measure_sequence(est, name, dataset, paths, ts, gt, max_diff, args) -> list[dict]:
    S = args.submap_size
    kf = list(range(0, len(paths), args.kf_stride))
    starts = list(range(0, max(len(kf) - S, 0), S - 1))[: args.max_boundaries + 1]
    if len(starts) < 2:
        return []

    prev = None
    rows = []
    for si, st in enumerate(starts):
        idxs = kf[st:st + S]
        if len(idxs) < 3:
            break
        images = [paths[j] for j in idxs]
        pred = est.infer(images)
        s_path, s_sim3, cond = submap_true_scale(
            pred.extrinsics, [ts[j] for j in idxs], gt, max_diff)
        cur = {
            "idx": si,
            "anchor_first_depth": pred.depth[0],
            "anchor_last_depth": pred.depth[len(idxs) - 1],
            "s_path": s_path, "s_sim3": s_sim3, "cond": cond,
        }
        if prev is not None:
            # Shared anchor: prev's LAST keyframe is curr's FIRST.
            r, sm, n = ratio_and_dispersion(prev["anchor_last_depth"],
                                            cur["anchor_first_depth"])

            def ratio(key):
                a, b = prev[key], cur[key]
                return b / a if a and b and a > 0 and b > 0 else None

            r_true = ratio("s_path")
            r_sim3 = ratio("s_sim3")
            rows.append({
                "dataset": dataset, "sequence": name, "boundary": si,
                "r": r, "sigma": sm, "n_px": n,
                "r_true": r_true, "r_true_sim3": r_sim3,
                "cond": min(prev["cond"] or 0.0, cur["cond"] or 0.0),
                "log_r": float(np.log(r)) if r > 0 else None,
                "log_r_true": (float(np.log(r_true))
                               if r_true and r_true > 0 else None),
                "log_r_true_sim3": (float(np.log(r_sim3))
                                    if r_sim3 and r_sim3 > 0 else None),
            })
        prev = cur
    return rows


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(rows: list[dict]) -> None:
    ok = [r for r in rows
          if r["log_r"] is not None and r["log_r_true"] is not None
          and np.isfinite(r["sigma"])]
    if not ok:
        print("\n  no usable boundaries — GT association or DA3 output failed")
        return

    print(f"\n  {len(ok)} usable boundaries of {len(rows)} measured\n")
    print("  PER-DOMAIN: dispersion, and which fixed estimator wins")
    print(f"  {'dataset':<9}{'n':>5}{'med sigma':>11}{'med |log r|':>13}"
          f"{'err ignore':>12}{'err apply':>11}{'lambda*':>9}")
    by = {}
    for r in ok:
        by.setdefault(r["dataset"], []).append(r)

    for ds, rs in by.items():
        lr = np.array([r["log_r"] for r in rs])
        lt = np.array([r["log_r_true"] for r in rs])
        sm = np.array([r["sigma"] for r in rs])
        e_ignore = np.mean(np.abs(lt))                 # damping 1.0
        e_apply = np.mean(np.abs(lr - lt))             # damping 0.0
        # lambda minimising mean |lambda*log r - log r_true| on this domain
        grid = np.linspace(0, 1, 101)
        errs = [np.mean(np.abs(g * lr - lt)) for g in grid]
        lam_star = float(grid[int(np.argmin(errs))])
        print(f"  {ds:<9}{len(rs):>5}{np.median(sm):>11.4f}"
              f"{np.median(np.abs(lr)):>13.4f}{e_ignore:>12.4f}"
              f"{e_apply:>11.4f}{lam_star:>9.2f}")

    print("\n  err ignore = damping 1.0 (today's default)   "
          "err apply = damping 0.0 (chaining)")
    print("  lambda* = the best fixed trust for that domain, 0=ignore 1=apply")

    # The decisive test: does sigma track the ratio's error WITHIN a domain?
    print("\n  DOES DISPERSION PREDICT ERROR?  (correlation of sigma with "
          "|log r - log r_true|)")
    print(f"  {'dataset':<9}{'pearson':>10}{'spearman':>10}")
    for ds, rs in by.items():
        sm = np.array([r["sigma"] for r in rs])
        err = np.abs(np.array([r["log_r"] for r in rs])
                     - np.array([r["log_r_true"] for r in rs]))
        if len(rs) < 4 or np.std(sm) < 1e-12:
            print(f"  {ds:<9}{'n/a':>10}{'n/a':>10}")
            continue
        pear = float(np.corrcoef(sm, err)[0, 1])
        rs_ = np.argsort(np.argsort(sm)).astype(float)
        re_ = np.argsort(np.argsort(err)).astype(float)
        spear = float(np.corrcoef(rs_, re_)[0, 1])
        print(f"  {ds:<9}{pear:>10.3f}{spear:>10.3f}")

    # And the practical question: can one GLOBAL rule beat per-domain constants?
    lr = np.array([r["log_r"] for r in ok])
    lt = np.array([r["log_r_true"] for r in ok])
    sm = np.array([r["sigma"] for r in ok])
    print("\n  POOLED across domains (this is what a domain-independent rule "
          "must win)")
    print(f"    damping 1.0 (ignore)        mean |err| = {np.mean(np.abs(lt)):.4f}")
    print(f"    damping 0.0 (apply)         mean |err| = "
          f"{np.mean(np.abs(lr - lt)):.4f}")
    best = (None, np.inf)
    for sp in np.logspace(-3, 0.5, 60):
        lam = sp ** 2 / (sp ** 2 + sm ** 2)
        e = np.mean(np.abs(lam * lr - lt))
        if e < best[1]:
            best = (float(sp), float(e))
    print(f"    shrinkage lambda(sigma)     mean |err| = {best[1]:.4f}   "
          f"(sigma_prior = {best[0]:.4f})")
    grid = np.linspace(0, 1, 101)
    e_fixed = min(np.mean(np.abs(g * lr - lt)) for g in grid)
    print(f"    best FIXED lambda (oracle)  mean |err| = {e_fixed:.4f}")
    print("\n  Shrinkage is only worth building if it beats the best fixed "
          "lambda — otherwise\n  a single better constant would do, with no "
          "new machinery.")

    # Ground truth sanity: path-length scale vs the ill-conditioned Sim3 one.
    both = [r for r in ok if r.get("log_r_true_sim3") is not None]
    if both:
        a = np.array([r["log_r_true"] for r in both])
        b = np.array([r["log_r_true_sim3"] for r in both])
        c = np.array([r["cond"] for r in both])
        print("\n  GROUND-TRUTH CHECK: path-length vs Sim3 estimate of the same "
              "boundary ratio")
        print(f"    n={len(both)}  correlation {np.corrcoef(a, b)[0, 1]:>+.3f}"
              f"   median |difference| {np.median(np.abs(a - b)):.4f}")
        print(f"    median window conditioning (s3/s1) {np.median(c):.4f}"
              f"  — below ~0.02 the Sim3 scale is not identifiable")
        print("    Disagreement here means the earlier Sim3-based version of "
              "this measurement\n    was reporting noise, not boundary error.")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from da3_runner import resolve_model_alias
    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    est = DepthEstimator(model_id=resolve_model_alias(args.depth_model),
                         process_resolution=504, backbone_dtype="bf16")

    rows = []
    for ds, names, loader in (("tum", args.tum, load_tum),
                              ("replica", args.replica, load_replica),
                              ("uas", args.uas, load_uas)):
        for n in names:
            try:
                paths, ts, gt, md = loader(n)
            except Exception as exc:
                print(f"  [skip] {ds}/{n}: {exc}")
                continue
            if not paths or not gt:
                print(f"  [skip] {ds}/{n}: images={len(paths)} gt={len(gt)}")
                continue
            print(f"=== {ds}/{n}: {len(paths)} frames, {len(gt)} GT poses")
            r = measure_sequence(est, n, ds, paths, ts, gt, md, args)
            print(f"    {len(r)} boundaries measured")
            rows.extend(r)

    (out / "boundaries.json").write_text(json.dumps(rows, indent=2, default=float))
    analyse(rows)


if __name__ == "__main__":
    main()
