"""
Aggregate + report the token-merging × submap-size sweep (consumes
sweep_merging.py's JSONL rows).

Every cell is measured over N repeats to capture DA3's bf16 non-determinism, so
nothing here reports a bare number: each metric is mean ± std over repeats.  A
cell only beats another if it does so BEYOND that band — the tables print both
and the summary flags which comparisons clear it.

    OOM cells are reported, not dropped.  A submap size that only runs with
    merging is the central claim of the study, so cells that ran out of memory
    are shown as "OOM" with the number of repeats affected, and drawn as ghost
    markers on the frontier plot.

Outputs (into --out_dir):
  report.md            — the tables, ready to paste into a results README
  summary.json         — the same aggregates, machine-readable
  merging_agg.csv      — one row per (submap size, arm) cell
  figures/frontier.png — ATE vs backbone latency + peak-memory panel, OOM ghosted
  figures/submap_scaling.png — ATE / latency / memory vs submap size, per arm
  figures/per_sequence.png   — ATE vs submap size, one panel per sequence

Usage:
    python scripts/report_merging.py --rows outputs/merging/merging_rows.jsonl \\
        --out_dir outputs/merging
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── loading ───────────────────────────────────────────────────────────────────

def arm_key(row: dict) -> str:
    """Stable identity for a merge arm, matching sweep_merging's --merge_starts."""
    if not row.get("merging"):
        return "off"
    return f"m{row.get('merge_start')}"


def arm_label(key: str) -> str:
    return "off (baseline)" if key == "off" else f"merge from block {key[1:]}"


def load_rows(path: str | Path) -> list[dict]:
    """Read the JSONL log, keeping the last row per (seq, submap, arm, repeat).

    The sweep appends, so re-running or resuming after a crash can write the
    same cell twice; the most recent run wins.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    deduped: dict[tuple, dict] = {}
    for row in rows:
        deduped[(row.get("sequence"), row.get("submap_size"),
                 arm_key(row), row.get("repeat"))] = row
    dropped = len(rows) - len(deduped)
    if dropped:
        print(f"  [info] {dropped} duplicate row(s) superseded by later runs")
    return list(deduped.values())


# ── aggregation ───────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> tuple[float | None, float]:
    """(mean, std) over the repeats, ignoring missing values."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None, 0.0
    return float(np.mean(clean)), float(np.std(clean))


def aggregate(rows: list[dict], metric: str) -> dict:
    """Aggregate to one entry per (submap_size, arm).

    Two levels, in this order: repeats are averaged WITHIN a sequence first,
    then sequences are averaged.  Doing it the other way round would let a
    sequence with more surviving repeats dominate the cell.
    """
    per_seq: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    status: dict[tuple, list[str]] = defaultdict(list)

    for row in rows:
        cell = (row.get("submap_size"), arm_key(row))
        status[cell].append(row.get("status", "ok"))
        if row.get("status", "ok") != "ok":
            continue
        seq = row.get("sequence")
        per_seq[cell][seq].append(row)

    out = {}
    for cell, seqs in per_seq.items():
        entry: dict = {"submap_size": cell[0], "arm": cell[1],
                       "sequences": sorted(seqs)}
        for field in (metric, "ate_sim3_rmse", "ate_se3_rmse", "rpe_trans_rmse",
                      "rpe_rot_rmse_deg", "backbone_forward_s", "total_s",
                      "peak_gpu_mem_mb", "n_submaps", "n_keyframes",
                      "n_loop_closures", "merge_token_ratio",
                      "gt_path_length_m"):
            seq_means, seq_stds = [], []
            for runs in seqs.values():
                mean, std = _stats([r.get(field) for r in runs])
                if mean is not None:
                    seq_means.append(mean)
                    seq_stds.append(std)
            entry[field] = float(np.mean(seq_means)) if seq_means else None
            # The noise band is the within-sequence std averaged over sequences
            # — the run-to-run band, not the between-sequence spread.
            entry[f"{field}_std"] = float(np.mean(seq_stds)) if seq_stds else 0.0
        entry["n_runs"] = sum(len(r) for r in seqs.values())
        out[cell] = entry

    # Cells that produced no usable run at all still need an entry, or an OOM
    # submap size vanishes from the report entirely.
    oom_peaks: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "oom" and row.get("peak_gpu_mem_mb"):
            oom_peaks[(row.get("submap_size"), arm_key(row))].append(
                row["peak_gpu_mem_mb"])

    for cell, states in status.items():
        entry = out.setdefault(cell, {"submap_size": cell[0], "arm": cell[1],
                                      "sequences": [], "n_runs": 0})
        entry["n_oom"] = sum(1 for s in states if s == "oom")
        entry["n_error"] = sum(1 for s in states if s == "error")
        entry["n_total"] = len(states)
        entry["oom"] = entry["n_oom"] > 0 and entry["n_runs"] == 0
        entry["partial_oom"] = entry["n_oom"] > 0 and entry["n_runs"] > 0
        # An OOM cell has no successful run to take memory from, but the
        # high-water mark it reached before dying is exactly the "where is the
        # wall" number the frontier plot and T2 want.
        if entry["oom"] and oom_peaks.get(cell):
            entry["peak_gpu_mem_mb"] = float(np.max(oom_peaks[cell]))
    return out


def per_sequence(rows: list[dict], metric: str) -> dict:
    """(sequence, submap_size, arm) → (mean, std, n_oom) for the breakdown table."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("sequence"), row.get("submap_size"),
                 arm_key(row))].append(row)
    out = {}
    for key, runs in buckets.items():
        ok = [r for r in runs if r.get("status", "ok") == "ok"]
        mean, std = _stats([r.get(metric) for r in ok])
        out[key] = (mean, std, sum(1 for r in runs if r.get("status") == "oom"))
    return out


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(value, std=None, digits=3, suffix="") -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}{suffix}"
    # Only show a band that is actually visible at this precision — "± 0.0" is
    # noise in the layout, not information.
    if std and round(std, digits) > 0:
        text += f" ± {std:.{digits}f}"
    return text


def _cell_text(entry: dict, metric: str, digits: int, suffix: str) -> str:
    if entry.get("oom"):
        return f"**OOM** ({entry.get('n_oom')}/{entry.get('n_total')})"
    text = _fmt(entry.get(metric), entry.get(f"{metric}_std"), digits, suffix)
    if entry.get("partial_oom"):
        text += f" ⚠{entry['n_oom']}oom"
    return text


def write_report(agg: dict, seq_table: dict, rows: list[dict], metric: str,
                 metric_label: str, digits: int, suffix: str,
                 out_dir: Path, rows_path: Path) -> dict:
    submaps = sorted({c[0] for c in agg})
    arms = sorted({c[1] for c in agg}, key=lambda a: (a != "off", a))
    sequences = sorted({r.get("sequence") for r in rows if r.get("sequence")})
    dataset = next((r.get("dataset") for r in rows if r.get("dataset")), "?")
    model = next((r.get("backbone_size") for r in rows), "?")
    resolution = next((r.get("resolution") for r in rows), "?")
    repeats = max((r.get("repeat", 0) for r in rows), default=0) + 1

    lines: list[str] = []
    add = lines.append
    add(f"# Token merging × submap size — {dataset.upper()}")
    add("")
    add(f"Model **{model}** @ {resolution} · {len(sequences)} sequence(s) · "
        f"{repeats} repeat(s) · rows `{rows_path}`")
    add("")
    add("Every cell is mean ± std over repeats (the bf16 noise band), averaged "
        "over sequences. Keyframes are frozen across the whole grid, so submap "
        "size is purely a batching parameter. **OOM** = the cell ran out of "
        "GPU memory — that is a result, not a gap.")
    add("")

    # ── T1: the grid ──────────────────────────────────────────────────────────
    add(f"## T1 — {metric_label} (headline)")
    add("")
    add("| submap | " + " | ".join(arm_label(a) for a in arms) + " |")
    add("|---:|" + "---:|" * len(arms))
    for submap in submaps:
        cells = []
        for arm in arms:
            entry = agg.get((submap, arm))
            cells.append(_cell_text(entry, metric, digits, suffix)
                         if entry else "—")
        add(f"| {submap} | " + " | ".join(cells) + " |")
    add("")

    # ── T2: the triple ────────────────────────────────────────────────────────
    add("## T2 — the triple (accuracy, speed, memory) + submap count")
    add("")
    add("| submap | arm | " + f"{metric_label} | ATE Sim3 (m) | backbone (s) | "
        "peak (GB) | submaps | tok ratio | runs |")
    add("|---:|:--|---:|---:|---:|---:|---:|---:|---:|")
    for submap in submaps:
        for arm in arms:
            entry = agg.get((submap, arm))
            if entry is None:
                continue
            if entry.get("oom"):
                add(f"| {submap} | {arm_label(arm)} | **OOM** | — | — | "
                    f"{_fmt((entry.get('peak_gpu_mem_mb') or 0) / 1024, digits=1)} | "
                    f"— | — | 0 |")
                continue
            mem = entry.get("peak_gpu_mem_mb")
            add(
                f"| {submap} | {arm_label(arm)} "
                f"| {_cell_text(entry, metric, digits, suffix)} "
                f"| {_fmt(entry.get('ate_sim3_rmse'), entry.get('ate_sim3_rmse_std'), 4)} "
                f"| {_fmt(entry.get('backbone_forward_s'), entry.get('backbone_forward_s_std'), 1)} "
                f"| {_fmt(mem / 1024 if mem else None, digits=2)} "
                f"| {_fmt(entry.get('n_submaps'), digits=1)} "
                f"| {_fmt(entry.get('merge_token_ratio'), digits=3)} "
                f"| {entry.get('n_runs', 0)} |")
    add("")

    # ── T3: merged vs unmerged at equal submap size ───────────────────────────
    add("## T3 — merging vs baseline at equal submap size")
    add("")
    add("Does merging pay for itself *at the same configuration*? Δ is merged − "
        "baseline; the accuracy Δ is only meaningful if it exceeds the noise "
        "band in the `beyond noise?` column.")
    add("")
    add(f"| submap | arm | Δ {metric_label} | noise band | beyond noise? "
        "| Δ backbone | Δ peak mem |")
    add("|---:|:--|---:|---:|:--:|---:|---:|")
    comparisons = []
    for submap in submaps:
        base = agg.get((submap, "off"))
        if base is None or base.get("oom"):
            continue
        for arm in arms:
            if arm == "off":
                continue
            entry = agg.get((submap, arm))
            if entry is None or entry.get("oom") or entry.get(metric) is None:
                continue
            d_acc = entry[metric] - base[metric]
            band = max(entry.get(f"{metric}_std", 0.0),
                       base.get(f"{metric}_std", 0.0))
            # With a single repeat the std is 0, and "bigger than 0" would call
            # every difference significant.  There is simply no band to clear,
            # so say so rather than claim a result the data cannot support.
            has_band = repeats > 1 and band > 0
            beyond = bool(has_band and abs(d_acc) > band)
            verdict = ("yes" if beyond else "no") if has_band \
                else f"n/a ({repeats} rep)"
            d_lat = _ratio(entry.get("backbone_forward_s"),
                           base.get("backbone_forward_s"))
            d_mem = _ratio(entry.get("peak_gpu_mem_mb"),
                           base.get("peak_gpu_mem_mb"))
            comparisons.append({
                "submap_size": submap, "arm": arm,
                "delta_metric": d_acc, "noise_band": band,
                "beyond_noise": beyond, "has_noise_band": has_band,
                "latency_ratio": d_lat, "memory_ratio": d_mem,
            })
            add(f"| {submap} | {arm_label(arm)} "
                f"| {d_acc:+.{digits}f}{suffix} | ± {band:.{digits}f} "
                f"| {verdict} "
                f"| {_pct_change(d_lat)} | {_pct_change(d_mem)} |")
    if not comparisons:
        add("| — | — | — | — | — | — | — |")
    add("")

    # ── T4: per-sequence breakdown ────────────────────────────────────────────
    add(f"## T4 — per-sequence {metric_label}")
    add("")
    header = "| sequence | submap | " + " | ".join(arm_label(a) for a in arms) + " |"
    add(header)
    add("|:--|---:|" + "---:|" * len(arms))
    for seq in sequences:
        for submap in submaps:
            cells = []
            present = False
            for arm in arms:
                mean, std, n_oom = seq_table.get((seq, submap, arm),
                                                 (None, 0.0, 0))
                if mean is None:
                    cells.append(f"OOM ({n_oom})" if n_oom else "—")
                else:
                    present = True
                    cells.append(_fmt(mean, std, digits, suffix))
            if present or any("OOM" in c for c in cells):
                add(f"| {seq} | {submap} | " + " | ".join(cells) + " |")
    add("")

    # ── the read ──────────────────────────────────────────────────────────────
    add("## How to read this")
    add("")
    add("1. **T3 is the 'is merging free?' question.** If accuracy Δ never "
        "clears the noise band while Δ backbone is a few percent, merging is "
        "neither helping nor hurting at that submap size — expected at 16, "
        "where cross-view attention is only ~10% of the forward.")
    add("2. **T1/T2 are the real question.** Look for a submap size that is "
        "**OOM without merging but runs with it**, and compare its accuracy "
        "against the largest baseline cell that does run. That is the only way "
        "this method wins: not by being faster, but by making a better "
        "operating point reachable.")
    add("3. **Watch `submaps`.** Fewer submaps = fewer anchor-frame "
        "compositions and metric-scale hops. If a larger submap improves "
        "accuracy, that column is the mechanism.")
    add("4. **`tok ratio`** is the *realised* merged/full token count. "
        "Attention cost scales with its square; if it is not well below 1, "
        "merging is not actually engaging (check `min_frames` vs submap size).")
    add("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  report  → {report_path}")

    return {
        "dataset": dataset, "model": model, "resolution": resolution,
        "sequences": sequences, "repeats": repeats,
        "metric": metric, "metric_label": metric_label,
        "cells": [
            {**{k: v for k, v in entry.items() if k != "sequences"}}
            for entry in agg.values()
        ],
        "comparisons": comparisons,
    }


def _ratio(value, base):
    if value is None or not base:
        return None
    return value / base


def _pct_change(ratio) -> str:
    if ratio is None:
        return "—"
    return f"{(ratio - 1.0) * 100:+.1f}%"


def write_csv(agg: dict, path: Path) -> None:
    fields = ["submap_size", "arm", "n_runs", "n_oom", "n_total", "oom",
              "ate_sim3_rmse", "ate_sim3_rmse_std", "ate_sim3_pct",
              "ate_sim3_pct_std", "ate_se3_rmse", "rpe_trans_rmse",
              "rpe_rot_rmse_deg", "backbone_forward_s",
              "backbone_forward_s_std", "total_s", "peak_gpu_mem_mb",
              "n_submaps", "n_keyframes", "n_loop_closures",
              "merge_token_ratio", "gt_path_length_m"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cell in sorted(agg, key=lambda c: (c[0], c[1] != "off", c[1])):
            writer.writerow(agg[cell])
    print(f"  csv     → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def plot(agg: dict, metric: str, metric_label: str, out_dir: Path,
         seq_table: dict | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available — figures not rendered")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    submaps = sorted({c[0] for c in agg})
    arms = sorted({c[1] for c in agg}, key=lambda a: (a != "off", a))
    cmap = plt.get_cmap("viridis", max(len(arms), 2))
    color = {a: cmap(i) for i, a in enumerate(arms)}
    # Marker size encodes submap size, so the frontier shows both axes at once.
    size_of = {s: 60 + 220 * i / max(len(submaps) - 1, 1)
               for i, s in enumerate(submaps)}

    # ── frontier: accuracy vs latency, and vs memory ──────────────────────────
    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    points = []
    for (submap, arm), entry in agg.items():
        if entry.get("oom") or entry.get(metric) is None:
            continue
        points.append((entry.get("backbone_forward_s"), entry[metric],
                       entry.get("peak_gpu_mem_mb"), arm, submap, entry))

    for ax, x_index, xlabel in ((ax_lat, 0, "backbone-only latency (s)"),
                                (ax_mem, 2, "peak GPU memory (GB)")):
        for x, y, mem, arm, submap, entry in points:
            xv = (x, y, mem)[x_index]
            if xv is None:
                continue
            if x_index == 2:
                xv = xv / 1024
            ax.scatter(xv, y, s=size_of[submap], color=color[arm],
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.annotate(f"{submap}", (xv, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
            err = entry.get(f"{metric}_std")
            if err:
                ax.errorbar(xv, y, yerr=err, color=color[arm], alpha=0.4,
                            capsize=2, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)

    # Ghost markers for OOM cells: placed at the memory the run reached before
    # dying, so the plot shows *where the wall is*, which is the argument.
    # y is fixed before the loop — each scatter can shift the limits, and the
    # ghosts should sit on one line rather than drift up the axis.
    ghost_y = ax_mem.get_ylim()[1] - 0.03 * (ax_mem.get_ylim()[1]
                                             - ax_mem.get_ylim()[0])
    ghosts = False
    for (submap, arm), entry in agg.items():
        mem = entry.get("peak_gpu_mem_mb")
        if not entry.get("oom") or not mem:
            continue
        ghosts = True
        ax_mem.scatter(mem / 1024, ghost_y, marker="X", s=size_of[submap],
                       color=color[arm], alpha=0.45, zorder=3)
        ax_mem.annotate(f"{submap} OOM", (mem / 1024, ghost_y), fontsize=7,
                        ha="right", xytext=(-6, -3),
                        textcoords="offset points", alpha=0.8)
    if ghosts:
        # Headroom so a ghost at the far right is not clipped by the axis.
        ax_mem.margins(x=0.12)

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[a],
                          label=arm_label(a)) for a in arms]
    handles.append(plt.Line2D([], [], marker="X", ls="", color="grey",
                              label="OOM (wall reached)"))
    ax_lat.legend(handles=handles, fontsize=8, loc="best")
    ax_lat.set_title("Accuracy vs speed")
    ax_mem.set_title("Accuracy vs memory (marker size = submap)")
    fig.suptitle("Token merging × submap size — is a better operating point "
                 "reachable?")
    fig.tight_layout()
    fig.savefig(fig_dir / "frontier.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'frontier.png'}")

    # ── scaling: each metric against submap size, one line per arm ────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = ((metric, metric_label, 1.0),
              ("backbone_forward_s", "backbone latency (s)", 1.0),
              ("peak_gpu_mem_mb", "peak GPU memory (GB)", 1 / 1024))
    for ax, (field, label, scale) in zip(axes, panels):
        for arm in arms:
            xs, ys, es = [], [], []
            for submap in submaps:
                entry = agg.get((submap, arm))
                if entry is None or entry.get("oom") or entry.get(field) is None:
                    continue
                xs.append(submap)
                ys.append(entry[field] * scale)
                es.append((entry.get(f"{field}_std") or 0.0) * scale)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            color=color[arm], label=arm_label(arm))
            # Mark where this arm hits the memory wall.
            for submap in submaps:
                entry = agg.get((submap, arm))
                if entry is not None and entry.get("oom"):
                    ax.axvline(submap, color=color[arm], ls=":", alpha=0.5)
        ax.set_xlabel("submap size (keyframes)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Scaling with submap size (dotted line = OOM for that arm)")
    fig.tight_layout()
    fig.savefig(fig_dir / "submap_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'submap_scaling.png'}")

    # ── per-sequence small multiples ─────────────────────────────────────────
    # The cell average hides a lot when sequences disagree — and on UAS they do
    # (whole percentage points apart).  One panel per sequence shows whether a
    # trend is real or one sequence dragging the mean.
    if not seq_table:
        return
    sequences = sorted({k[0] for k in seq_table if k[0]})
    if len(sequences) < 2:
        return
    cols = min(len(sequences), 4)
    rows_n = (len(sequences) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.8 * rows_n),
                             squeeze=False)
    for ax, seq in zip([a for row in axes for a in row], sequences):
        for arm in arms:
            xs, ys, es = [], [], []
            for submap in submaps:
                mean, std, _ = seq_table.get((seq, submap, arm), (None, 0.0, 0))
                if mean is None:
                    continue
                xs.append(submap)
                ys.append(mean)
                es.append(std)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            color=color[arm], label=arm_label(arm))
            for submap in submaps:
                _, _, n_oom = seq_table.get((seq, submap, arm), (None, 0.0, 0))
                if n_oom:
                    ax.axvline(submap, color=color[arm], ls=":", alpha=0.5)
        ax.set_title(seq, fontsize=10)
        ax.set_xlabel("submap size")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
    for ax in [a for row in axes for a in row][len(sequences):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{metric_label} vs submap size, per sequence")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_sequence.png", dpi=150)
    plt.close(fig)
    print(f"  figure  → {fig_dir / 'per_sequence.png'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate + report the token-merging sweep")
    p.add_argument("--rows", required=True, help="JSONL log from sweep_merging.py")
    p.add_argument("--out_dir", default=None,
                   help="Where to write report/figures (default: alongside --rows)")
    p.add_argument("--metric", default=None,
                   choices=["ate_sim3_pct", "ate_sim3_rmse", "ate_se3_rmse"],
                   help="Headline metric.  Default: %% of path length on UAS "
                        "(km-scale), Sim3 ATE in metres elsewhere")
    args = p.parse_args()

    rows_path = Path(args.rows)
    out_dir = Path(args.out_dir) if args.out_dir else rows_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(rows_path)
    if not rows:
        raise SystemExit(f"no rows in {rows_path}")

    dataset = next((r.get("dataset") for r in rows if r.get("dataset")), "")
    # Metres everywhere, including UAS.  The %-of-path form is still available
    # via --metric ate_sim3_pct, but it is no longer the default: it hides the
    # absolute magnitude, and comparing across sequences of very different
    # length is what the per-sequence tables are for.
    metric = args.metric or "ate_sim3_rmse"
    metric_label, digits, suffix = {
        "ate_sim3_pct": ("ATE Sim3 (% of path)", 2, "%"),
        "ate_sim3_rmse": ("ATE Sim3 (m)", 4, ""),
        "ate_se3_rmse": ("ATE SE3 (m)", 4, ""),
    }[metric]

    n_ok = sum(1 for r in rows if r.get("status", "ok") == "ok")
    n_oom = sum(1 for r in rows if r.get("status") == "oom")
    n_err = len(rows) - n_ok - n_oom
    print(f"  {len(rows)} runs: {n_ok} ok, {n_oom} OOM, {n_err} other")

    agg = aggregate(rows, metric)
    seq_table = per_sequence(rows, metric)

    summary = write_report(agg, seq_table, rows, metric, metric_label,
                           digits, suffix, out_dir, rows_path)
    write_csv(agg, out_dir / "merging_agg.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  summary → {out_dir / 'summary.json'}")
    plot(agg, metric, metric_label, out_dir, seq_table)


if __name__ == "__main__":
    main()
