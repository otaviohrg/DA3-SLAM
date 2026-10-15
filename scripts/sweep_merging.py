"""
Token-merging × submap-size sweep (plan_fastvggt-token-merging.txt, Step 2).

Runs DA3-SLAM over the grid

    {merging off, merging on at one or more start blocks} × {submap sizes}
        × {repeats} × {sequences}

and logs one JSONL row per run with the full triple — ATE (Sim3 + SE3, in
metres and as % of ground-truth path length), backbone-only latency, peak GPU
memory — plus submap counts and the *realised* merge behaviour.  Defaults to
the UAS dataset, where the question actually bites (km-scale, many submaps).

WHY THIS SWEEP IS SHAPED THIS WAY
---------------------------------
FastVGGT's merging buys ~4% at submap 16 (see the plan's first measurements):
the cross-view attention it removes is only ~10% of the forward at that batch
size.  The interesting claim is not "faster at the same config" but "merging
makes a LARGER submap affordable, and larger submaps have fewer boundaries" —
every submap boundary is an anchor-frame pose composition plus a metric-scale
hop, so halving the submap count halves the places drift enters.  The grid is
therefore merging × submap size, not merging alone.

    OOM IS A RESULT, NOT A FAILURE.  Large submaps without merging are expected
    to run out of memory; those cells are caught, logged with status="oom" and
    their high-water memory, and reported as ghost points.  They are the
    evidence for the method, so they must never be silently dropped.

FROZEN KEYFRAMES — THE CONTROL THAT MAKES THIS COMPARISON VALID
---------------------------------------------------------------
Submap size is NOT purely a batching parameter by default: `KeyframeSelectorConfig
.max_submap_size` is kept in sync with `submap_size`, and the selector force-emits
a keyframe once that many frames have passed without one.  Sweeping submap size
naively therefore changes *which frames are keyframes*, and any ATE difference
is confounded by that.

So this sweep generates ONE frozen keyframe list per sequence (at --freeze_at,
by default the largest submap size in the grid — the largest cap means the
fewest forced keyframes, i.e. the most flow-driven list) and every cell replays
it.  With `keyframes_from` set, optical-flow selection is bypassed entirely
(`slam.py` swaps in the replay selector), so submap size becomes purely a
batching parameter and the comparison is clean.

Usage:
    python scripts/sweep_merging.py \\
        --seq_dir data/UAS/fyllingsdalen_tunnel \\
                  data/UAS/runehamar_tunnel/hornbill \\
                  data/UAS/campus_fog \\
                  data/UAS/frozen_lake \\
        --submap_sizes 16 24 32 48 --merge_starts off 0 7 --repeats 3 \\
        --out_dir outputs/merging

    python scripts/report_merging.py --rows outputs/merging/merging_rows.jsonl \\
        --out_dir outputs/merging

Note the default --depth_model: config/default.yaml ships DA3METRIC-LARGE,
which has alt_start=-1 and NO cross-view attention, so it cannot merge at all.
The merging study runs nested-giant, the same model as the Branch B results.
"""

from __future__ import annotations

import argparse
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import results_log
from da3_runner import model_short_name, resolve_model_alias
from da3_slam.config import DEFAULT_YAML, TokenMergingConfig, load_slam_config
from sweep_compute import _score, prepare_sequence
from tum_eval_common import SharedSLAM


@dataclass(frozen=True)
class MergeArm:
    """One arm of the A/B: either unmerged, or merging from a given block."""

    key: str                              # short label, e.g. "off" / "m0"
    config: TokenMergingConfig | None      # None = detach the wrapper entirely

    @property
    def label(self) -> str:
        if self.config is None:
            return "off (baseline)"
        return f"merge from block {self.config.start}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA3-SLAM token-merging × submap-size sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seq_dir", nargs="+", required=True,
                   help="Sequence directories (see --dataset for the layout)")
    p.add_argument("--dataset", choices=["uas", "tum", "replica"], default="uas",
                   help="Sequence format")
    p.add_argument("--out_dir", default="outputs/merging",
                   help="Root output directory")
    p.add_argument("--results_row", default=None,
                   help="JSONL log (default: <out_dir>/merging_rows.jsonl)")

    # ── the grid ──────────────────────────────────────────────────────────────
    p.add_argument("--submap_sizes", nargs="+", type=int,
                   default=[16, 24, 32, 48],
                   help="Keyframes per submap (including the anchor overlap)")
    p.add_argument("--merge_starts", nargs="+", default=["off", "0"],
                   help="Merge arms: 'off' for the unmerged baseline, or an "
                        "integer = first merging block as a POSITION among the "
                        "global blocks (0 = all of them, FastVGGT's default)")
    p.add_argument("--merge_ratio", type=float, default=0.9,
                   help="Fraction of tokens absorbed (all merged arms)")
    p.add_argument("--merge_min_frames", type=int, default=8,
                   help="Never merge batches smaller than this (keeps the "
                        "loop-closure re-inference exact)")
    p.add_argument("--no_merge_protect", dest="merge_protect",
                   action="store_false",
                   help="Ablate FastVGGT's protected-token set")
    p.add_argument("--repeats", type=int, default=3,
                   help="Runs per cell (>=3 to capture the bf16 noise band)")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells already present in --results_row.  The grid "
                        "is long enough that a run can be interrupted (the host "
                        "OOM killer is a real risk on the largest sequences), "
                        "so this makes it restartable without redoing work")
    p.add_argument("--repeat_offset", type=int, default=0,
                   help="Index of the first repeat.  The log is append-only and "
                        "the report dedupes by (seq, submap, arm, repeat), so a "
                        "long grid can be run one repeat at a time: pass 0, "
                        "then 1, then 2 against the same --results_row")

    # ── model / config ────────────────────────────────────────────────────────
    p.add_argument("--depth_model", default="nested-giant",
                   help="Model alias or HF ID.  MUST have cross-view attention: "
                        "DA3METRIC-LARGE (the YAML default) has alt_start=-1 "
                        "and cannot merge")
    p.add_argument("--resolution", type=int, default=None,
                   help="DA3 processing resolution (default: the YAML value)")
    p.add_argument("--backbone_dtype", choices=["fp32", "bf16"], default=None,
                   help="ViT backbone weight precision (default: the YAML "
                        "value).  bf16 cuts peak GPU memory ~40%% and is what "
                        "makes submap sizes above ~48 reachable at all")
    p.add_argument("--config", default=str(DEFAULT_YAML), help="Base YAML config")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap frames per sequence (quick tests)")
    p.add_argument("--no_loop_closure", action="store_true",
                   help="Disable loop closure for the whole sweep")

    # ── experimental controls ─────────────────────────────────────────────────
    p.add_argument("--freeze_at", type=int, default=None,
                   help="Submap size used to GENERATE the frozen keyframe list "
                        "(default: max of --submap_sizes).  Every cell then "
                        "replays that one list, so submap size is purely a "
                        "batching parameter — see the module docstring")
    p.add_argument("--keyframes_dir", default=None,
                   help="Where frozen keyframe lists live (default: "
                        "<out_dir>/keyframes).  Point several sweeps at one "
                        "directory to share the control across studies")
    p.add_argument("--max_diff", type=float, default=None,
                   help="Estimate↔GT association tolerance (s); dataset-aware "
                        "default (uas 0.05, tum 0.02, replica 0.5/fps)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Replica frame rate (synthetic timestamps = idx/fps)")

    # UAS-only I/O (mirrors benchmark_uas.py / sweep_compute.py)
    p.add_argument("--topic", default=None, help="UAS camera topic override")
    p.add_argument("--calib", default=None, help="UAS calibration YAML override")
    p.add_argument("--calib_dir", type=Path, default=None,
                   help="UAS calibration/ folder override")
    p.add_argument("--no_undistort", dest="undistort", action="store_false",
                   help="UAS: skip fisheye undistortion")
    p.add_argument("--seqcache", default=None,
                   help="Extracted/undistorted frame cache (default: "
                        "<out_dir>/_seqcache).  UAS bag extraction is ~20 GB "
                        "and idempotent, so point this at an existing cache "
                        "(e.g. outputs/sweep/step1_uas_resolution/_seqcache) "
                        "to reuse it instead of decoding the bags again")
    return p.parse_args()


def build_arms(args: argparse.Namespace) -> list[MergeArm]:
    """Turn --merge_starts into merge arms, preserving the order given."""
    arms: list[MergeArm] = []
    for token in args.merge_starts:
        if str(token).lower() in ("off", "none", "baseline"):
            arms.append(MergeArm(key="off", config=None))
            continue
        try:
            start = int(token)
        except ValueError:
            raise SystemExit(
                f"--merge_starts takes 'off' or an integer block position, "
                f"got {token!r}")
        arms.append(MergeArm(
            key=f"m{start}",
            config=TokenMergingConfig(
                enable=True,
                start=start,
                merge_ratio=args.merge_ratio,
                protect=args.merge_protect,
                min_frames=args.merge_min_frames,
            )))
    if not any(a.config is None for a in arms):
        print("  [warn] no 'off' arm in --merge_starts — the sweep will have no "
              "unmerged baseline to compare against")
    return arms


def main() -> None:
    args = parse_args()
    if args.max_diff is None:
        args.max_diff = {"uas": 0.05, "replica": 0.5 / args.fps}.get(
            args.dataset, 0.02)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_row = Path(args.results_row) if args.results_row \
        else out_dir / "merging_rows.jsonl"
    if results_row.exists():
        print(f"  [warn] {results_row} exists — rows are APPENDED. "
              f"report_merging dedupes by (seq, submap, arm, repeat) keeping "
              f"the latest; delete the file for a clean run.")
    keyframes_dir = Path(args.keyframes_dir) if args.keyframes_dir \
        else out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = [Path(s) for s in args.seq_dir]
    arms = build_arms(args)
    submap_sizes = sorted(set(args.submap_sizes))
    freeze_at = args.freeze_at or max(submap_sizes)

    model_id = resolve_model_alias(args.depth_model)
    cfg = load_slam_config(args.config, depth_model=model_id,
                           depth_model_resolution=args.resolution,
                           backbone_dtype=args.backbone_dtype,
                           submap_size=submap_sizes[0])
    if args.no_loop_closure:
        cfg.enable_loop_closure = False

    n_runs = len(arms) * len(submap_sizes) * args.repeats * len(seq_dirs)
    print(f"\n{'━' * 66}")
    print(f"  Token-merging sweep: {len(arms)} arms × {len(submap_sizes)} "
          f"submap sizes × {args.repeats} repeats × {len(seq_dirs)} sequences"
          f" = {n_runs} runs")
    print(f"  model      {model_short_name(model_id)} @ "
          f"{cfg.depth_model_resolution}  backbone {cfg.backbone_dtype}")
    print(f"  arms       {', '.join(a.label for a in arms)}")
    print(f"  submaps    {submap_sizes}   (keyframes frozen at {freeze_at})")
    print(f"  log     → {results_row}")
    print(f"{'━' * 66}\n")

    done = _completed_cells(results_row) if args.resume else set()
    if done:
        print(f"  resume: {len(done)} cell(s) already logged will be skipped\n")

    model = SharedSLAM(cfg)
    model.set_build_pointclouds(False)      # lean: no clouds in a timing sweep

    prepared: dict[str, tuple | None] = {}
    seqcache = Path(args.seqcache) if args.seqcache else out_dir / "_seqcache"

    # The frozen keyframe list is the experimental control — generate it for
    # every sequence FIRST, at --freeze_at, before any grid cell runs.  Doing it
    # up front (rather than lazily on the first cell to touch a sequence) means
    # the list never depends on the order the grid happens to be walked in.
    _freeze_keyframes(model, seq_dirs, args, freeze_at, keyframes_dir,
                      prepared, seqcache)

    run_idx = 0
    sweep_t0 = time.time()
    # Repeat is the OUTER loop so a full pass over the grid happens before any
    # config is measured twice: if the GPU thermally throttles over a long
    # sweep, it affects every arm roughly equally instead of penalising
    # whichever arm happened to run last.
    for repeat in range(args.repeat_offset,
                        args.repeat_offset + args.repeats):
        for submap_size in submap_sizes:
            for arm in arms:
                model.config.submap_size = submap_size
                model.set_token_merging(arm.config)
                for seq_dir in seq_dirs:
                    run_idx += 1
                    tag = (f"[{run_idx}/{n_runs}] submap={submap_size} "
                           f"{arm.key} rep={repeat} {seq_dir.name}")
                    if (seq_dir.name, submap_size, arm.key, repeat) in done:
                        print(f"  {tag}  [skip: already logged]")
                        continue
                    _run_cell(model, seq_dir, args, arm, submap_size, repeat,
                              results_row, keyframes_dir, tag, prepared,
                              seqcache)
                    _report_host_memory(tag)

    print(f"\n{'═' * 66}\n  Sweep done: {run_idx} runs in "
          f"{time.time() - sweep_t0:.0f}s\n  rows → {results_row}")
    print(f"  next: python scripts/report_merging.py --rows {results_row} "
          f"--out_dir {out_dir}")


def _completed_cells(results_row: Path) -> set[tuple]:
    """(sequence, submap_size, arm_key, repeat) already present in the log."""
    import json
    if not results_row.exists():
        return set()
    done = set()
    with open(results_row) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            arm = "off" if not row.get("merging") else f"m{row.get('merge_start')}"
            done.add((row.get("sequence"), row.get("submap_size"), arm,
                      row.get("repeat")))
    return done


def _report_host_memory(tag: str) -> None:
    """Print the process's resident HOST memory after each cell.

    Not a nicety: a 56-cell grid over km-scale sequences was killed by the
    kernel OOM killer at 24 GB RSS partway through, with nothing in the log to
    say so (SIGKILL leaves no traceback).  Printing RSS per cell makes the
    growth visible before it becomes fatal.
    """
    try:
        with open("/proc/self/status") as f:
            rss_kb = next(int(l.split()[1]) for l in f
                          if l.startswith("VmRSS:"))
    except Exception:
        return
    gb = rss_kb / (1024 * 1024)
    flag = "  <-- HIGH" if gb > 16 else ""
    print(f"    host RSS after cell: {gb:.1f} GB{flag}")


def _freeze_keyframes(model, seq_dirs, args, freeze_at: int,
                      keyframes_dir: Path, prepared: dict,
                      seqcache: Path) -> None:
    """Generate the one frozen keyframe list per sequence (see the docstring).

    Runs each sequence once at --freeze_at with keyframe *selection* live, then
    every grid cell replays the recorded seq_idxs.  Sequences whose list already
    exists are skipped, so a re-run or a second sweep sharing --keyframes_dir
    reuses the same control.
    """
    from da3_runner import run_da3

    todo = [s for s in seq_dirs if not (keyframes_dir / f"{s.name}.txt").exists()]
    if not todo:
        print("  frozen keyframes: all lists already present — reusing them\n")
        return

    print(f"  frozen keyframes: generating {len(todo)} list(s) at "
          f"submap_size={freeze_at}")
    saved_submap = model.config.submap_size
    saved_max = model.config.keyframe.max_submap_size
    model.config.submap_size = freeze_at
    model.config.keyframe.max_submap_size = freeze_at
    model.set_token_merging(None)           # never generate the control merged
    try:
        for seq_dir in todo:
            seq = prepare_sequence(seq_dir, args, seqcache, prepared)
            if seq is None:
                continue
            image_paths, timestamps, gt_all = seq
            _warn_gt_overlap(seq_dir.name, timestamps, gt_all, args)
            kf_path = keyframes_dir / f"{seq_dir.name}.txt"
            print(f"    {seq_dir.name} → {kf_path}")
            model.set_keyframe_io(dump_keyframes=str(kf_path))
            try:
                run_da3(image_paths, timestamps, model,
                        Namespace(max_frames=args.max_frames, repeat=0,
                                  snapshot_interval=0.0))
            except Exception as exc:
                print(f"    [ERROR] keyframe generation for {seq_dir.name}: "
                      f"{exc}")
                traceback.print_exc()
    finally:
        model.config.submap_size = saved_submap
        model.config.keyframe.max_submap_size = saved_max
        model.set_keyframe_io()
    print()


def _warn_gt_overlap(seq_name: str, timestamps: list, gt_all: list,
                     args) -> None:
    """Warn when the frame window barely overlaps the ground truth.

    UAS ground truth does not necessarily start with the bag: frozen_lake's GT
    begins ~43 s after the first frame, so `--max_frames 600` yields zero
    matchable poses and every cell scores 'unmatched'.  That looks like a broken
    sweep but is just a truncated window, so say so before burning the grid.
    """
    if not timestamps or not gt_all:
        return
    gt_ts = [entry[0] for entry in gt_all]
    lo, hi = min(timestamps), max(timestamps)
    inside = sum(1 for t in gt_ts if lo <= t <= hi)
    if inside >= 3:
        return
    print(f"    [warn] {seq_name}: only {inside} GT pose(s) fall inside the "
          f"frame window [{lo:.1f}, {hi:.1f}] (GT spans "
          f"[{min(gt_ts):.1f}, {max(gt_ts):.1f}]).  Every cell will score "
          f"'unmatched'"
          + (" — raise or drop --max_frames." if args.max_frames else "."))


def _run_cell(model, seq_dir: Path, args, arm: MergeArm, submap_size: int,
              repeat: int, results_row: Path, keyframes_dir: Path, tag: str,
              prepared: dict, seqcache: Path) -> None:
    """Run one grid cell and append its row — including failure rows."""
    from da3_runner import run_da3

    seq = prepare_sequence(seq_dir, args, seqcache, prepared)
    if seq is None:
        print(f"  [SKIP] {tag}: sequence could not be prepared")
        return
    image_paths, timestamps, gt_all = seq
    print(f"\n  {tag}")

    kf_path = keyframes_dir / f"{seq_dir.name}.txt"
    if not kf_path.exists():
        print(f"    [SKIP] {tag}: no frozen keyframe list at {kf_path}")
        return
    model.set_keyframe_io(keyframes_from=str(kf_path))

    merger = getattr(model._slam.estimator, "_merger", None)
    if merger is not None:
        merger.reset_stats()

    run_args = Namespace(max_frames=args.max_frames, repeat=repeat,
                         snapshot_interval=0.0)
    try:
        est_ts_to_pose, timings, counts = run_da3(
            image_paths, timestamps, model, run_args)
    except Exception as exc:
        status, peak_mb = _classify_failure(exc)
        note = "OOM" if status == "oom" else type(exc).__name__
        print(f"    [{note}] {tag}: {exc}")
        if status != "oom":
            traceback.print_exc()
        _append_failure(results_row, model, run_args, seq_dir, args,
                        status, peak_mb, str(exc)[:300])
        return

    metrics = _score(est_ts_to_pose, gt_all, seq_dir.name, timings, counts,
                     args.dataset, args.max_diff)
    if metrics is None:
        print(f"    [SKIP] {seq_dir.name}: too few matched poses")
        _append_failure(results_row, model, run_args, seq_dir, args,
                        "unmatched", timings.get("peak_gpu_mem_mb"), "")
        return

    if merger is not None:
        stats = merger.stats
        metrics["merge_token_ratio"] = stats.token_ratio
        metrics["merge_calls"] = stats.calls_merged
        metrics["merge_calls_passthrough"] = stats.calls_passthrough

    row = results_log.build_row(model.config, run_args, metrics)
    results_log.append_row(results_row, row)

    pct = row.get("ate_sim3_pct")
    pct_str = f" ({pct:.2f}% of path)" if pct is not None else ""
    ratio = row.get("merge_token_ratio")
    ratio_str = f"  tok={ratio:.3f}" if arm.config is not None and ratio else ""
    print(f"    ATE sim3={metrics['ate_sim3']['rmse']:.4f}m{pct_str}  "
          f"submaps={metrics.get('n_submaps')}  "
          f"backbone={timings['backbone_forward_s']:.1f}s  "
          f"peak={timings['peak_gpu_mem_mb']:.0f}MB{ratio_str}")


def _classify_failure(exc: Exception) -> tuple[str, float | None]:
    """('oom'|'error', peak MB).  OOM is an expected outcome for large submaps
    without merging, so it is classified, its high-water memory recorded, and
    the allocator reset — otherwise the next cell inherits a poisoned pool."""
    import torch

    message = str(exc).lower()
    is_oom = isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()))
    is_oom = is_oom or "out of memory" in message or "cuda oom" in message

    peak_mb = None
    if torch.cuda.is_available():
        try:
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return ("oom" if is_oom else "error"), peak_mb


def _append_failure(results_row: Path, model, run_args, seq_dir: Path,
                    args, status: str, peak_mb: float | None,
                    message: str) -> None:
    """Log a cell that did not produce a trajectory.

    An OOM row is the whole point of the sweep — it is what says a submap size
    is unreachable without merging — so it carries the same config identity as a
    successful row, with the metrics left null.
    """
    metrics = {
        "system": "DA3-SLAM", "dataset": args.dataset,
        "sequence": seq_dir.name, "status": status,
        "timings": {"peak_gpu_mem_mb": peak_mb},
    }
    row = results_log.build_row(model.config, run_args, metrics)
    row["error"] = message
    results_log.append_row(results_row, row)


if __name__ == "__main__":
    main()
