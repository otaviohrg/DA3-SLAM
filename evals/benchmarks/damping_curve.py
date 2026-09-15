"""Summarise the boundary_scale_damping sweep.

    python damping_curve.py            # all three datasets + per-scene 7-Scenes

Each inter-submap boundary ratio is applied as delta^(1-g), so cumulative scale
is S_k = prod_j delta_j^(1-g).  g=0 keeps the measured ratios (shipped); g=1
forces every ratio to 1.0, i.e. no chaining — DA3's own metric consistency
alone.  The sweep asks whether error is monotone in g and whether the optimum
differs per dataset, which the two endpoints alone cannot show.
"""
from __future__ import annotations
import glob, json, statistics as st
from pathlib import Path

ROOT = Path("/home/pierre-yves/otavio/SLAM/DA3-SLAM/outputs/ablation_damping")
GS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
DS = [("tum", 9), ("replica", 8), ("7scenes", 7)]


def arm(g: str, ds: str, n: int):
    fs = glob.glob(str(ROOT / f"g{g}" / ds / "*" / "results.json"))
    if len(fs) != n:
        return None
    d = [json.load(open(p)) for p in fs]
    return {
        "sim3": st.mean(x["ate_sim3"]["rmse"] for x in d),
        "se3": st.mean(x["ate_se3"]["rmse"] for x in d),
        "scale": st.mean(x["ate_sim3"].get("scale", float("nan")) for x in d),
        "per": {x["sequence"].replace("_seq-01", "")
                .replace("rgbd_dataset_freiburg1_", ""): x["ate_sim3"]["rmse"]
                for x in d},
        "perscale": {x["sequence"].replace("_seq-01", "")
                     .replace("rgbd_dataset_freiburg1_", ""):
                     x["ate_sim3"].get("scale", float("nan")) for x in d},
    }


def main() -> None:
    print("  Sim(3) ATE RMSE (m) vs boundary_scale_damping g")
    print(f"  {'g':<6}" + "".join(f"{d:>12}" for d, _ in DS))
    rows = {}
    for g in GS:
        cells, rows[g] = [], {}
        for ds, n in DS:
            a = arm(g, ds, n)
            rows[g][ds] = a
            cells.append(f"{a['sim3']:>12.4f}" if a else f"{'—':>12}")
        print(f"  {g:<6}" + "".join(cells))

    # best arm per dataset
    print()
    for ds, _ in DS:
        have = [(g, rows[g][ds]["sim3"]) for g in GS if rows[g].get(ds)]
        if len(have) < 2:
            continue
        bg, bv = min(have, key=lambda t: t[1])
        base = dict(have).get("0.0")
        note = ""
        if base:
            note = (f"  shipped g=0.0 is best" if bg == "0.0"
                    else f"  BETTER than shipped g=0.0 ({base:.4f}) by {base/bv:.2f}x")
        print(f"  {ds:<9} best g={bg}  {bv:.4f}{note}")

    # 7-Scenes per scene: stairs is the question
    print("\n  7-Scenes per scene (Sim3 / scale) — stairs is the motivating case")
    have = [g for g in GS if rows[g].get("7scenes")]
    if have:
        scenes = sorted(rows[have[0]]["7scenes"]["per"])
        print(f"    {'scene':<12}" + "".join(f"{'g='+g:>16}" for g in have))
        for s in scenes:
            cells = []
            for g in have:
                a = rows[g]["7scenes"]
                cells.append(f"{a['per'][s]:>9.4f}/{a['perscale'][s]:<6.3f}")
            print(f"    {s:<12}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
