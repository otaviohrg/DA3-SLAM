"""
Aggregate + plot the compute sweep (plan Step 0f; consumes sweep_compute.py's
JSONL rows).

Every config is measured over N repeats to capture DA3's bf16 non-determinism,
so nothing here reports a bare number: each metric is mean ± noise, where the
noise band is the std over repeats (averaged across the dev sequences).  A
config only "wins" if it beats another BEYOND that band.

Outputs (into --out_dir):
  * printed table + sweep_agg.csv  — per (size, resolution): ATE mean±std,
    backbone-only latency, peak memory, token proxy, averaged over sequences.
  * frontier.png                   — the headline Pareto scatter: ATE vs
    backbone latency, coloured by size, marker-sized by resolution, non-
    dominated frontier drawn, the default config starred; + a peak-memory panel.

    python scripts/plot_sweep.py --rows outputs/sweep/sweep_rows.jsonl \\
        --out_dir outputs/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Which config is the un-tuned default (starred on the frontier / baseline).
DEFAULT_SIZE = "nested-giant"
DEFAULT_RESOLUTION = 504

# Mirror of da3_runner.DA3_MODEL_ALIASES (kept local so plotting stays
# dependency-light).  Sweep rows store the resolved HF ID; map it back to the
# short label for display and for matching the default config.
_ID_TO_SHORT = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": "nested-giant",
    "depth-anything/DA3-GIANT-1.1": "giant",
    "depth-anything/DA3-LARGE-1.1": "large",
    "depth-anything/DA3-BASE": "base",
    "depth-anything/DA3-SMALL": "small",
}


def _short(size: str) -> str:
    """Short label for a model ID (alias if known, else basename)."""
    return _ID_TO_SHORT.get(size, size.rsplit("/", 1)[-1].lower())


def load_rows(path: str | Path) -> list[dict]:
    """Read a JSONL sweep log (one run per line), deduped.

    The sweep appends, so re-running (or resuming after a crash) can write the
    same (size, resolution, repeat, sequence) twice.  Keep the last occurrence
    of each — the most recent run wins — and report how many were dropped.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seen: dict[tuple, dict] = {}
    for r in rows:
        seen[(r.get("backbone_size"), r.get("resolution"),
              r.get("repeat"), r.get("sequence"))] = r
    deduped = list(seen.values())
    if len(deduped) < len(rows):
        print(f"  [dedupe] dropped {len(rows) - len(deduped)} duplicate "
              f"row(s) from re-runs (kept the latest of each config)")
    return deduped


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse repeats → mean ± noise per (size, resolution).

    Per (size, resolution, sequence) the std over repeats is the noise band;
    the headline ATE is the mean over sequences of the per-sequence means, and
    the reported band is the RMS of the per-sequence repeat-stds (so it
    reflects run-to-run noise, not the spread between different scenes).
    """
    by_config_seq: dict[tuple, dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        key = (_short(r["backbone_size"]), r["resolution"], r["sequence"])
        for m in ("ate_sim3_rmse", "ate_se3_rmse", "backbone_forward_s",
                  "peak_gpu_mem_mb", "total_s", "fps"):
            if r.get(m) is not None:
                by_config_seq[key][m].append(r[m])
        by_config_seq[key]["tokens_per_frame"].append(r.get("tokens_per_frame"))

    # Per (size, res, seq): mean + std-over-repeats.
    seq_stats: dict[tuple, dict] = {}
    for (size, res, seq), metrics in by_config_seq.items():
        seq_stats[(size, res, seq)] = {
            m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items()
            if v and v[0] is not None
        }

    # Per (size, res): average the per-seq means; RMS the per-seq repeat-stds.
    by_config: dict[tuple, list] = defaultdict(list)
    for (size, res, seq), stats in seq_stats.items():
        by_config[(size, res)].append(stats)

    out = []
    for (size, res), seq_list in sorted(by_config.items()):
        agg = {"backbone_size": size, "resolution": res,
               "n_sequences": len(seq_list)}
        for m in ("ate_sim3_rmse", "ate_se3_rmse", "backbone_forward_s",
                  "peak_gpu_mem_mb", "total_s", "fps", "tokens_per_frame"):
            means = [s[m][0] for s in seq_list if m in s]
            stds = [s[m][1] for s in seq_list if m in s]
            if means:
                agg[m] = float(np.mean(means))
                agg[m + "_noise"] = float(np.sqrt(np.mean(np.square(stds))))
        out.append(agg)
    return out


def print_table(agg: list[dict]) -> None:
    """Human-readable per-config table (ATE mean±noise, latency, memory)."""
    hdr = (f"{'size':<14} {'res':>5} {'ATE sim3 (m)':>20} "
           f"{'backbone (s)':>13} {'peak (MB)':>10} {'tokens/f':>9} {'seqs':>5}")
    print("\n" + hdr)
    print("─" * len(hdr))
    for a in agg:
        ate = a.get("ate_sim3_rmse", float("nan"))
        ate_n = a.get("ate_sim3_rmse_noise", 0.0)
        print(f"{a['backbone_size']:<14} {a['resolution']:>5} "
              f"{ate:>10.4f} ± {ate_n:<7.5f} "
              f"{a.get('backbone_forward_s', float('nan')):>13.1f} "
              f"{a.get('peak_gpu_mem_mb', float('nan')):>10.0f} "
              f"{a.get('tokens_per_frame', float('nan')):>9.0f} "
              f"{a['n_sequences']:>5}")


def save_csv(agg: list[dict], path: Path) -> None:
    if not agg:
        return
    keys = sorted({k for a in agg for k in a})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(agg)
    print(f"\n  table → {path}")


def _pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points for minimise-both axes (x, y)."""
    front = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            xj <= xi and yj <= yi and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: points[i][0])


def plot_frontier(agg: list[dict], out_path: Path) -> None:
    """F3: ATE vs backbone latency (colour = size, marker size = resolution),
    Pareto frontier drawn, default config starred; + a peak-memory panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available — figures not rendered")
        return

    pts = [a for a in agg if a.get("ate_sim3_rmse") is not None
           and a.get("backbone_forward_s") is not None]
    if not pts:
        print("  [skip] no complete points to plot")
        return

    sizes = sorted({a["backbone_size"] for a in pts})
    cmap = plt.get_cmap("viridis", max(len(sizes), 2))
    color = {s: cmap(i) for i, s in enumerate(sizes)}
    resolutions = sorted({a["resolution"] for a in pts})
    rmin, rmax = min(resolutions), max(resolutions)

    def msize(res: int) -> float:
        frac = 0.0 if rmax == rmin else (res - rmin) / (rmax - rmin)
        return 60 + 260 * frac

    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, xkey, xlabel in ((ax_lat, "backbone_forward_s", "backbone latency (s)"),
                             (ax_mem, "peak_gpu_mem_mb", "peak GPU memory (MB)")):
        for a in pts:
            ax.scatter(a[xkey], a["ate_sim3_rmse"], s=msize(a["resolution"]),
                       color=color[a["backbone_size"]], edgecolor="k",
                       linewidth=0.5, alpha=0.85, zorder=3)
            xerr = a.get(xkey + "_noise", 0.0) if xkey != "peak_gpu_mem_mb" else 0.0
            ax.errorbar(a[xkey], a["ate_sim3_rmse"],
                        yerr=a.get("ate_sim3_rmse_noise", 0.0), xerr=xerr,
                        fmt="none", ecolor="grey", alpha=0.5, zorder=2)
            if (a["backbone_size"] == DEFAULT_SIZE
                    and a["resolution"] == DEFAULT_RESOLUTION):
                ax.scatter(a[xkey], a["ate_sim3_rmse"], marker="*", s=420,
                           facecolor="none", edgecolor="red", linewidth=1.8,
                           zorder=4, label="default")
        front = _pareto_front([(a[xkey], a["ate_sim3_rmse"]) for a in pts])
        ax.plot([pts[i][xkey] for i in front],
                [pts[i]["ate_sim3_rmse"] for i in front],
                "--", color="grey", zorder=1, label="Pareto frontier")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ATE Sim3 RMSE (m)")
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[s],
                          markeredgecolor="k", label=s) for s in sizes]
    handles.append(plt.Line2D([], [], ls="--", color="grey", label="Pareto frontier"))
    ax_lat.legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Compute sweep — ATE vs backbone latency / peak memory\n"
                 "(colour = size, marker size = resolution, ★ = default)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  frontier → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate + plot the compute sweep")
    p.add_argument("--rows", required=True, help="sweep_rows.jsonl from sweep_compute.py")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: alongside --rows)")
    args = p.parse_args()

    rows = load_rows(args.rows)
    if not rows:
        print(f"No rows in {args.rows}")
        return
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.rows).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(rows)
    print(f"Loaded {len(rows)} runs → {len(agg)} configs")
    print_table(agg)
    save_csv(agg, out_dir / "sweep_agg.csv")
    plot_frontier(agg, out_dir / "frontier.png")


if __name__ == "__main__":
    main()
