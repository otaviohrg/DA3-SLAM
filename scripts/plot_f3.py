"""
F3 — the headline Pareto scatter (plan Step 3): ATE vs backbone latency, one
point per (size, resolution) config, coloured by model size and marker-sized by
resolution, with the Pareto frontier drawn and the default config starred. A
second panel plots peak GPU memory vs ATE.

Consumes a combined JSONL of sweep rows for ONE dataset (size ladder + resolution
sweep + any interior grid cells). Aggregates to one point per (size, resolution)
by averaging over sequences and repeats. For UAS, pass rows whose
`ate_sim3_rmse` has already been converted to % of trajectory length.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {
    "depth-anything/DA3-SMALL": "small", "depth-anything/DA3-BASE": "base",
    "depth-anything/DA3-LARGE-1.1": "large", "depth-anything/DA3-GIANT-1.1": "giant",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": "nested-giant",
}
ORDER = ["small", "base", "large", "giant", "nested-giant"]
DEFAULT = ("nested-giant", 504)


def short(s):
    return SHORT.get(s, s)


def aggregate(rows):
    """→ {(size, res): {ate, backbone_s, peak_mb}} averaged over seq+repeat."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (short(r["backbone_size"]), int(r["resolution"]))
        cells[k]["ate"].append(r["ate_sim3_rmse"])
        cells[k]["bb"].append(r["backbone_forward_s"])
        cells[k]["mem"].append(r["peak_gpu_mem_mb"])
    return {k: {"ate": float(np.mean(v["ate"])),
                "bb": float(np.mean(v["bb"])),
                "mem": float(np.mean(v["mem"]))} for k, v in cells.items()}


def pareto(points):
    """Indices of non-dominated points, minimising both (x, y)."""
    keep = []
    for i, (xi, yi) in enumerate(points):
        if not any(xj <= xi and yj <= yi and (xj < xi or yj < yi)
                   for j, (xj, yj) in enumerate(points) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def main():
    p = argparse.ArgumentParser(description="F3 Pareto scatter")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Pareto frontier")
    p.add_argument("--ate_unit", default="m")
    args = p.parse_args()
    ate_lbl = f"ATE Sim3 RMSE ({args.ate_unit})"

    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    # dedup by (size,res,repeat,seq) — keep latest
    seen = {}
    for r in rows:
        seen[(short(r["backbone_size"]), r["resolution"], r["repeat"],
              r["sequence"])] = r
    agg = aggregate(list(seen.values()))

    keys = list(agg)
    sizes = [s for s in ORDER if s in {k[0] for k in keys}]
    cmap = plt.get_cmap("viridis", max(len(sizes), 2))
    color = {s: cmap(i) for i, s in enumerate(sizes)}
    resolutions = sorted({k[1] for k in keys})
    rmin, rmax = min(resolutions), max(resolutions)

    def msize(res):
        f = 0.0 if rmax == rmin else (res - rmin) / (rmax - rmin)
        return 70 + 300 * f

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, xkey, xlabel in ((axL, "bb", "backbone latency (s)"),
                             (axR, "mem", "peak GPU memory (MB)")):
        for k in keys:
            s, res = k
            ax.scatter(agg[k][xkey], agg[k]["ate"], s=msize(res),
                       color=color[s], edgecolor="k", linewidth=.5,
                       alpha=.85, zorder=3)
            if k == DEFAULT:
                ax.scatter(agg[k][xkey], agg[k]["ate"], marker="*", s=460,
                           facecolor="none", edgecolor="red", linewidth=1.8,
                           zorder=5, label="default (nested@504)")
        pts = [(agg[k][xkey], agg[k]["ate"]) for k in keys]
        front = pareto(pts)
        ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                "--", color="grey", zorder=1, label="Pareto frontier")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ate_lbl)
        ax.grid(True, alpha=.3)

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[s],
                          markeredgecolor="k", label=s) for s in sizes]
    handles += [plt.Line2D([], [], marker="*", ls="", color="w",
                           markeredgecolor="red", markersize=13, label="default"),
                plt.Line2D([], [], ls="--", color="grey", label="Pareto frontier")]
    axL.legend(handles=handles, fontsize=8, loc="best")
    # marker-size legend (resolution)
    for res in resolutions:
        axR.scatter([], [], s=msize(res), color="grey", edgecolor="k",
                    label=f"res {res}")
    axR.legend(fontsize=8, loc="best", title="marker size")

    fig.suptitle(f"{args.title}\n(colour = model size · marker size = resolution "
                 "· ★ = default)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"  F3 → {args.out}  ({len(keys)} configs, "
          f"{len(pareto([(agg[k]['bb'], agg[k]['ate']) for k in keys]))} on frontier)")


if __name__ == "__main__":
    main()
