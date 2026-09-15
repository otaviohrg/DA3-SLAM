"""Steady-state throughput from the two-point sweep.

Wall time is t(N) = t_init + c*N for a fixed per-frame cost c, so running the
SAME sequence at two truncation lengths cancels the intercept:

    fps_steady = (N2 - N1) / (t2 - t1)
    t_init     = t1 - N1 * (t2 - t1) / (N2 - N1)

This needs no instrumentation inside any system — only --max_frames — so one
definition applies to all of them.  That matters because the figures already in
results.json are not comparable: VGGT-SLAM's internal `fps` field and
frames/wall disagree by 9x, the external baselines only ever had harness
wall-clock, and ViSTA-SLAM recorded one shared wall time across every scene.

The end-to-end figure at N2 is printed alongside, because that (model load
included) is what most published FPS columns actually measure even when they do
not say so.  Report whichever you like, but say which in the caption.
"""
import glob
import json
from pathlib import Path

FPS = Path("/tmp/fps")
DASH = Path("/home/pierre-yves/otavio/SLAM/DA3-SLAM/outputs/fps")
N1, N2 = 250, 1000


def wall(base: Path):
    """(frames, seconds) from the single results.json under `base`, or None."""
    hits = glob.glob(str(base / "*" / "results.json"))
    if not hits:
        return None
    d = json.load(open(hits[0]))
    t = d.get("timings") or {}
    secs = t.get("total_s") or t.get("total")
    n = t.get("n_frames") or d.get("n_frames")
    return (n, secs) if n and secs else None


def main() -> None:
    systems = sorted({p.name for p in FPS.iterdir() if p.is_dir()}) if FPS.is_dir() else []
    if DASH.is_dir():
        systems.append("DASH-SLAM")
    if not systems:
        print("  no runs yet")
        return

    print(f"  {'system':<16}{'dataset':<10}{'steady fps':>12}"
          f"{'end-to-end':>12}{'init (s)':>10}")
    for s in systems:
        for ds in ("tum", "replica", "7scenes"):
            root = (DASH / ds) if s == "DASH-SLAM" else (FPS / s / ds)
            a, b = wall(root / f"n{N1}"), wall(root / f"n{N2}")
            if not (a and b):
                continue
            (n1, t1), (n2, t2) = a, b
            if n2 <= n1 or t2 <= t1:
                # Non-monotone: the short run was not actually shorter (a system
                # may floor at a minimum keyframe count), so the slope is
                # meaningless and only the end-to-end figure is reportable.
                print(f"  {s:<16}{ds:<10}{'n/a':>12}{n2 / t2:>12.2f}{'—':>10}"
                      f"   non-monotone; end-to-end only")
                continue
            c = (t2 - t1) / (n2 - n1)
            print(f"  {s:<16}{ds:<10}{1.0 / c:>12.2f}{n2 / t2:>12.2f}"
                  f"{t1 - n1 * c:>10.1f}")


if __name__ == "__main__":
    main()
