"""
Report the keyframe-density × submap-size grid (consumes sweep_keyframe_grid.py).

The grid exists to settle one question: is ATE driven by SUBMAP SIZE, or by how
much of the sequence one DA3 batch spans?  At a fixed keyframe density the two
are proportional and indistinguishable — this grid breaks that tie by varying
density independently.

The report therefore leads with the discriminating test rather than the raw
grid:

  * T1  the 2-D grid (density × submap size) — the raw picture.
  * T2  ISO-SPAN groups.  Cells with similar frames-per-batch but *different*
        submap sizes.  If span is the controlling variable they agree; if
        submap size is, they do not.
  * T3  the competing fits.  ATE regressed on log(submap size) and on
        log(frames per batch), with per-sequence intercepts so that sequences
        of differing difficulty cannot flatter either model.  R^2 decides.

Outputs (into --out_dir):
  report.md, summary.json, kfgrid_agg.csv
  figures/grid.png        — ATE vs submap size, one line per density
  figures/iso_span.png    — ATE vs frames-per-batch, all cells collapsed
  figures/heatmap.png     — density × submap size heatmap, iso-span diagonals

Usage:
    python scripts/report_keyframe_grid.py --rows outputs/kfgrid_segment/kfgrid_rows.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    deduped = {}
    for row in rows:
        deduped[(row.get("sequence"), row.get("submap_size"),
                 row.get("kf_density"), row.get("repeat"))] = row
    dropped = len(rows) - len(deduped)
    if dropped:
        print(f"  [info] {dropped} duplicate row(s) superseded")
    return list(deduped.values())


def _stats(values) -> tuple[float | None, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None, 0.0
    return float(np.mean(clean)), float(np.std(clean))


def aggregate(rows: list[dict], metric: str) -> dict:
    """One entry per (density, submap size): repeats averaged within a sequence
    first, then across sequences, so an unevenly-completed sequence cannot
    dominate a cell."""
    per_seq = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        per_seq[(row.get("kf_density"), row.get("submap_size"))][
            row.get("sequence")].append(row)

    out = {}
    for cell, seqs in per_seq.items():
        entry = {"kf_density": cell[0], "submap_size": cell[1],
                 "n_runs": sum(len(v) for v in seqs.values())}
        # A cell that yields <=2 submaps is not testing "submap size N" — with
        # one submap the whole sequence is a single DA3 batch and there are no
        # inter-submap boundaries at all, so the pose graph degenerates to a
        # within-batch chain.  TUM fr1 sequences are short enough that the
        # low-density / large-submap corner falls into this.  Marked, not
        # dropped: the cells are still the extreme of "one giant batch".
        submap_counts = [r.get("n_submaps") for runs in seqs.values()
                         for r in runs if r.get("n_submaps")]
        entry["min_submaps"] = min(submap_counts) if submap_counts else None
        entry["degenerate"] = bool(submap_counts and min(submap_counts) <= 2)
        for field in (metric, "ate_se3_rmse", "rpe_trans_rmse", "n_keyframes",
                      "n_submaps", "frames_per_submap", "frames_per_keyframe",
                      "backbone_forward_s", "peak_gpu_mem_mb",
                      "n_loop_closures"):
            means, stds = [], []
            for runs in seqs.values():
                mean, std = _stats([r.get(field) for r in runs])
                if mean is not None:
                    means.append(mean)
                    stds.append(std)
            entry[field] = float(np.mean(means)) if means else None
            entry[f"{field}_std"] = float(np.mean(stds)) if stds else 0.0
        out[cell] = entry
    return out


def fit_models(rows: list[dict], metric: str) -> dict:
    """Regress ATE on log(submap size) vs log(frames per batch).

    Both fits include a per-sequence intercept (fixed effects).  Without it a
    sequence that is simply harder inflates whichever predictor happens to
    correlate with which sequences completed, and the comparison is worthless.
    """
    # Degenerate runs (<=2 submaps) are excluded from the fits: with a single
    # batch there is no submap structure for either hypothesis to explain.
    ok = [r for r in rows
          if r.get("status", "ok") == "ok" and r.get(metric) is not None
          and r.get("frames_per_submap") and (r.get("n_submaps") or 0) > 2]
    if len(ok) < 6:
        return {}
    sequences = sorted({r["sequence"] for r in ok})
    y = np.array([r[metric] for r in ok], dtype=float)

    def fit(values) -> tuple[float, float]:
        x = np.log(np.array(values, dtype=float))
        dummies = np.array([[1.0 if r["sequence"] == s else 0.0
                             for s in sequences] for r in ok])
        design = np.column_stack([x, dummies])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return float(coef[0]), 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    slope_size, r2_size = fit([r["submap_size"] for r in ok])
    slope_span, r2_span = fit([r["frames_per_submap"] for r in ok])
    return {
        "n": len(ok), "sequences": sequences,
        "submap_size": {"slope": slope_size, "r2": r2_size},
        "frames_per_submap": {"slope": slope_span, "r2": r2_span},
        "winner": ("frames_per_submap" if r2_span > r2_size else "submap_size"),
    }


def iso_span_groups(agg: dict, metric: str, tolerance: float = 0.25) -> list[dict]:
    """Group cells whose frames-per-batch agree within `tolerance` (relative).

    Only groups spanning at least two different submap sizes are interesting —
    those are the ones where the two hypotheses make different predictions.
    """
    cells = [dict(v, cell=k) for k, v in agg.items()
             if v.get("frames_per_submap") and v.get(metric) is not None
             and not v.get("degenerate")]
    cells.sort(key=lambda c: c["frames_per_submap"])
    groups, current = [], []
    for cell in cells:
        if current and cell["frames_per_submap"] > \
                current[0]["frames_per_submap"] * (1 + tolerance):
            groups.append(current)
            current = []
        current.append(cell)
    if current:
        groups.append(current)
    return [g for g in groups if len({c["submap_size"] for c in g}) > 1]


def write_report(rows, agg, fits, metric, metric_label, digits, out_dir,
                 rows_path) -> dict:
    densities = sorted({c[0] for c in agg})
    submaps = sorted({c[1] for c in agg})
    mode = next((r.get("selection_mode") for r in rows), "?")
    dataset = next((r.get("dataset") for r in rows), "?")
    repeats = max((r.get("repeat", 0) for r in rows), default=0) + 1
    sequences = sorted({r.get("sequence") for r in rows if r.get("sequence")})

    lines, add = [], None
    add = lines.append
    add(f"# Keyframe density × submap size — {dataset.upper()}, "
        f"`{mode}` selection")
    add("")
    add(f"{len(sequences)} sequence(s) · {repeats} repeat(s) · rows `{rows_path}`")
    add("")
    add("Keyframes are frozen per **(sequence, density)** and replayed across "
        "every submap size, so selection never depends on batching. Density is "
        "a multiplier on the shipped default (2.0 = twice as many keyframes).")
    add("")
    n_degen = sum(1 for v in agg.values() if v.get("degenerate"))
    if n_degen:
        add(f"⚠ **{n_degen} cell(s) are degenerate** (≤2 submaps — the whole "
            "sequence in one or two DA3 batches, so there are no inter-submap "
            "boundaries to speak of). They are marked `⚠Nsub` in T1 and are "
            "**excluded from T2 and T3**, which test submap structure.")
    add("")

    add(f"## T1 — {metric_label} (density × submap size)")
    add("")
    add("| density | " + " | ".join(f"submap {s}" for s in submaps) +
        " | frames/keyframe |")
    add("|---:|" + "---:|" * (len(submaps) + 1))
    for density in densities:
        cells = []
        fpk = None
        for submap in submaps:
            entry = agg.get((density, submap))
            if entry is None or entry.get(metric) is None:
                cells.append("—")
                continue
            fpk = fpk or entry.get("frames_per_keyframe")
            std = entry.get(f"{metric}_std")
            text = f"{entry[metric]:.{digits}f}"
            if std and round(std, digits) > 0:
                text += f" ± {std:.{digits}f}"
            if entry.get("degenerate"):
                text += f" ⚠{entry.get('min_submaps')}sub"
            cells.append(text)
        add(f"| {density:g} | " + " | ".join(cells) +
            f" | {fpk:.1f} |" if fpk else f"| {density:g} | " +
            " | ".join(cells) + " | — |")
    add("")

    add("## T1b — frames per DA3 batch (the span variable)")
    add("")
    add("| density | " + " | ".join(f"submap {s}" for s in submaps) + " |")
    add("|---:|" + "---:|" * len(submaps))
    for density in densities:
        cells = []
        for submap in submaps:
            entry = agg.get((density, submap))
            span = entry.get("frames_per_submap") if entry else None
            cells.append(f"{span:.0f}" if span else "—")
        add(f"| {density:g} | " + " | ".join(cells) + " |")
    add("")

    groups = iso_span_groups(agg, metric)
    add("## T2 — iso-span groups (the discriminating test)")
    add("")
    add("Cells with similar frames-per-batch but **different submap sizes**. If "
        "span is what matters they agree; if submap size is, they do not.")
    add("")
    add("| span (frames/batch) | cells (density → submap) | "
        f"{metric_label} range | spread |")
    add("|---:|:--|:--|---:|")
    for group in groups:
        spans = [c["frames_per_submap"] for c in group]
        values = [c[metric] for c in group]
        cells = ", ".join(f"{c['kf_density']:g}→{c['submap_size']}"
                          for c in group)
        add(f"| {min(spans):.0f}–{max(spans):.0f} | {cells} | "
            f"{min(values):.{digits}f}–{max(values):.{digits}f} | "
            f"{max(values) - min(values):.{digits}f} |")
    if not groups:
        add("| — | (no iso-span group spans two submap sizes) | — | — |")
    add("")

    add("## T3 — competing fits (per-sequence intercepts)")
    add("")
    if fits:
        add(f"| predictor | slope | R² |")
        add("|:--|---:|---:|")
        for key, label in (("submap_size", "log(submap size)"),
                           ("frames_per_submap", "log(frames per batch)")):
            add(f"| {label} | {fits[key]['slope']:+.4f} | {fits[key]['r2']:.3f} |")
        add("")
        winner = "frames per batch" if fits["winner"] == "frames_per_submap" \
            else "submap size"
        add(f"**Better predictor: {winner}** "
            f"(n={fits['n']} cells, {len(fits['sequences'])} sequences).")
    else:
        add("_Too few cells to fit._")
    add("")

    best = min((v for v in agg.values()
                if v.get(metric) is not None and not v.get("degenerate")),
               key=lambda v: v[metric], default=None)
    if best:
        add("## Best cell")
        add("")
        add(f"density **{best['kf_density']:g}** × submap **{best['submap_size']}** "
            f"→ {metric_label} **{best[metric]:.{digits}f}**, "
            f"{best.get('frames_per_submap', 0):.0f} frames/batch, "
            f"{best.get('n_keyframes', 0):.0f} keyframes, "
            f"backbone {best.get('backbone_forward_s', 0):.1f} s.")
        add("")
        edge = []
        if best["kf_density"] in (min(densities), max(densities)):
            edge.append("density")
        if best["submap_size"] in (min(submaps), max(submaps)):
            edge.append("submap size")
        if edge:
            add(f"> ⚠ The best cell sits at the edge of the swept range in "
                f"{' and '.join(edge)} — the optimum may lie outside the grid.")
            add("")

    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"  report  → {out_dir / 'report.md'}")
    return {"dataset": dataset, "selection_mode": mode, "repeats": repeats,
            "sequences": sequences, "metric": metric, "fits": fits,
            "cells": [dict(v, cell=list(k)) for k, v in agg.items()],
            "best": best}


def write_csv(agg: dict, path: Path) -> None:
    fields = ["kf_density", "submap_size", "n_runs", "degenerate",
              "min_submaps", "ate_sim3_rmse",
              "ate_sim3_rmse_std", "ate_se3_rmse", "rpe_trans_rmse",
              "n_keyframes", "n_submaps", "frames_per_submap",
              "frames_per_keyframe", "backbone_forward_s", "peak_gpu_mem_mb",
              "n_loop_closures"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cell in sorted(agg):
            writer.writerow(agg[cell])
    print(f"  csv     → {path}")


def plot(agg, rows, metric, metric_label, out_dir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    densities = sorted({c[0] for c in agg})
    submaps = sorted({c[1] for c in agg})
    cmap = plt.get_cmap("viridis", max(len(densities), 2))
    color = {d: cmap(i) for i, d in enumerate(densities)}

    # ── grid: ATE vs submap size, one line per density ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for density in densities:
        xs, ys, es = [], [], []
        for submap in submaps:
            entry = agg.get((density, submap))
            if entry is None or entry.get(metric) is None:
                continue
            xs.append(submap)
            ys.append(entry[metric])
            es.append(entry.get(f"{metric}_std") or 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                        color=color[density], label=f"density {density:g}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(submaps)
    ax.set_xticklabels([str(s) for s in submaps])
    ax.set_xlabel("submap size (keyframes)")
    ax.set_ylabel(metric_label)
    ax.set_title("If submap size were the driver,\nthe lines would coincide")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "grid.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'grid.png'}")

    # ── iso-span: everything against frames-per-batch ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for density in densities:
        xs, ys = [], []
        for submap in submaps:
            entry = agg.get((density, submap))
            if entry is None or entry.get(metric) is None \
                    or not entry.get("frames_per_submap"):
                continue
            xs.append(entry["frames_per_submap"])
            ys.append(entry[metric])
        if xs:
            ax.scatter(xs, ys, s=70, color=color[density], edgecolor="black",
                       linewidth=0.5, label=f"density {density:g}", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("frames covered by one DA3 batch (span)")
    ax.set_ylabel(metric_label)
    ax.set_title("If span were the driver,\nall densities would fall on one curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "iso_span.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'iso_span.png'}")

    # ── heatmap with iso-span diagonals ──────────────────────────────────────
    grid = np.full((len(densities), len(submaps)), np.nan)
    for i, density in enumerate(densities):
        for j, submap in enumerate(submaps):
            entry = agg.get((density, submap))
            if entry and entry.get(metric) is not None:
                grid[i, j] = entry[metric]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(submaps)))
    ax.set_xticklabels([str(s) for s in submaps])
    ax.set_yticks(range(len(densities)))
    ax.set_yticklabels([f"{d:g}" for d in densities])
    ax.set_xlabel("submap size (keyframes)")
    ax.set_ylabel("keyframe density (× default)")
    for i in range(len(densities)):
        for j in range(len(submaps)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label=metric_label)
    ax.set_title("Iso-span cells run along the anti-diagonal\n"
                 "(double the density, double the submap = same span)")
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'heatmap.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--metric", default=None,
                   choices=["ate_sim3_rmse", "ate_sim3_pct", "ate_se3_rmse"])
    args = p.parse_args()

    rows_path = Path(args.rows)
    out_dir = Path(args.out_dir) if args.out_dir else rows_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(rows_path)
    if not rows:
        raise SystemExit(f"no rows in {rows_path}")
    dataset = next((r.get("dataset") for r in rows), "")
    # Metres everywhere, including UAS.  The %-of-path form is still available
    # via --metric ate_sim3_pct, but it is no longer the default: it hides the
    # absolute magnitude, and comparing across sequences of very different
    # length is what the per-sequence tables are for.
    metric = args.metric or "ate_sim3_rmse"
    metric_label, digits = {
        "ate_sim3_rmse": ("ATE Sim3 (m)", 4),
        "ate_sim3_pct": ("ATE Sim3 (% of path)", 2),
        "ate_se3_rmse": ("ATE SE3 (m)", 4),
    }[metric]

    n_ok = sum(1 for r in rows if r.get("status", "ok") == "ok")
    print(f"  {len(rows)} runs: {n_ok} ok, {len(rows) - n_ok} failed")

    agg = aggregate(rows, metric)
    fits = fit_models(rows, metric)
    summary = write_report(rows, agg, fits, metric, metric_label, digits,
                           out_dir, rows_path)
    write_csv(agg, out_dir / "kfgrid_agg.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))
    print(f"  summary → {out_dir / 'summary.json'}")
    plot(agg, rows, metric, metric_label, out_dir)


if __name__ == "__main__":
    main()
