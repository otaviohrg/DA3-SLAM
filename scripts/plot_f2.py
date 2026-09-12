"""
F2 figure for the model-size ladder (plan Step 2): ATE and the map-detail proxy
vs model size, to look for the trajectory-vs-map dissociation (H2 — does ATE
saturate at a smaller size than the map does?).

Consumes a T2 JSON:
    {"<size>": [ate_mean, ate_median, backbone_s, peak_gb, map_points, chamfer_m]}

Left panel : ATE Sim3 vs model size (ordered small→nested-giant), latency
             annotated at each point.
Right panel: map-detail vs size — Chamfer to the nested-giant cloud and the
             point count. Omitted when map-detail is absent (UAS km-scale),
             leaving a single ATE panel.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["small", "base", "large", "giant", "nested-giant"]


def main() -> None:
    p = argparse.ArgumentParser(description="F2: ATE + map-detail vs model size")
    p.add_argument("--t2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Model-size ladder")
    p.add_argument("--ate_unit", default="m", help="ATE axis unit label")
    args = p.parse_args()

    t2 = json.load(open(args.t2))
    sizes = [s for s in ORDER if s in t2]
    x = list(range(len(sizes)))
    ate = [t2[s][0] for s in sizes]
    bb = [t2[s][2] for s in sizes]
    pts = [t2[s][4] for s in sizes]
    cham = [t2[s][5] for s in sizes]
    has_md = any(p_ > 0 for p_ in pts)

    if has_md:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    else:
        fig, axL = plt.subplots(1, 1, figsize=(7, 5.2))

    axL.plot(x, ate, "o-", color="tab:blue", zorder=3)
    for xi, a, b in zip(x, ate, bb):
        axL.annotate(f"{b:.0f}s", (xi, a), textcoords="offset points",
                     xytext=(0, 9), fontsize=8, ha="center", color="tab:blue")
    best = int(min(range(len(sizes)), key=lambda i: ate[i]))
    axL.scatter([x[best]], [ate[best]], s=360, facecolor="none",
                edgecolor="red", linewidth=2, zorder=4, label=f"best ({sizes[best]})")
    axL.set_xticks(x)
    axL.set_xticklabels(sizes, rotation=20, ha="right")
    axL.set_ylabel(f"ATE Sim3 RMSE ({args.ate_unit})")
    axL.set_xlabel("model size  (latency annotated)")
    axL.set_title("Trajectory ATE vs model size")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=9)

    if has_md:
        axR.plot(x, [c for c in cham], "s--", color="tab:orange",
                 label="Chamfer vs nested (m)", zorder=3)
        axR.set_ylabel("Chamfer to nested-giant cloud (m)", color="tab:orange")
        axR.tick_params(axis="y", labelcolor="tab:orange")
        ax2 = axR.twinx()
        ax2.plot(x, [pt / 1e6 for pt in pts], "^-", color="tab:green",
                 label="map points (M)", zorder=3)
        ax2.set_ylabel("map points (millions)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.set_ylim(0, max(pt / 1e6 for pt in pts) * 1.25)
        axR.set_xticks(x)
        axR.set_xticklabels(sizes, rotation=20, ha="right")
        axR.set_xlabel("model size")
        axR.set_title("Map detail vs model size")
        axR.grid(True, alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"  F2 → {args.out}  (best size {sizes[best]})")


if __name__ == "__main__":
    main()
