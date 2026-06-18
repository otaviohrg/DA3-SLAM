"""
EuRoC support helpers for the evo wrappers (evo_batch.py, evo_compare.py).

EuRoC differs from the TUM/Replica datasets in two ways the evo scripts must
handle:

  1. Ground truth is not a ready trajectory file — it is the IMU/body-frame
     state CSV, which must be converted to a camera-frame TUM file (via the
     T_BS extrinsic).  euroc_gt_tum() does that, reusing euroc_common.

  2. Estimated-trajectory timestamps may be nanoseconds (run_slam.py on the
     raw EuRoC frames, whose filenames are ns integers) or seconds
     (benchmark_euroc.py).  normalize_to_seconds() rescales ns→s so evo's
     default association tolerance (0.01 s) works regardless of source.
"""

from __future__ import annotations

from pathlib import Path

from euroc_common import find_mav0, ensure_euroc_gt_tum

# Timestamps above this are nanoseconds, not seconds.  EuRoC ns stamps are
# ~1.4e18; both TUM (~1.3e9) and EuRoC-seconds (~1.4e9) stamps fall well below.
_NANOSECOND_THRESHOLD = 1e12


def is_euroc_sequence(path: str | Path) -> bool:
    """True if `path` is (or contains) a EuRoC sequence (has a mav0/ dir)."""
    return find_mav0(Path(path)) is not None


def euroc_gt_tum(seq_dir: str | Path, cam: str = "cam0") -> Path:
    """Camera-frame TUM ground-truth file (seconds) for a EuRoC sequence."""
    return ensure_euroc_gt_tum(seq_dir, cam=cam, seconds=True)


def find_euroc_seq_dir(gt_dir: str | Path, seq_name: str) -> Path | None:
    """Locate the EuRoC sequence directory named `seq_name` under `gt_dir`."""
    gt_dir = Path(gt_dir)
    direct = gt_dir / seq_name
    if is_euroc_sequence(direct):
        return direct
    for candidate in gt_dir.rglob(seq_name):
        if candidate.is_dir() and is_euroc_sequence(candidate):
            return candidate
    return None


def sequence_name_from_run(run_name: str) -> str:
    """Recover the EuRoC sequence name from an output run directory name.

    eval_euroc.sh names run dirs '<seq>_run<k>_w<n>_<model>[...]', and EuRoC
    sequence names never contain '_run', so splitting on it recovers the
    sequence (e.g. 'MH_01_easy_run1_w20_nested-giant' → 'MH_01_easy').
    """
    return run_name.split("_run")[0]


def normalize_to_seconds(traj):
    """Return `traj` with timestamps in seconds, rescaling from ns if needed.

    `traj` is an evo PoseTrajectory3D.  Only nanosecond-range timestamps are
    rescaled, so this is a no-op for trajectories already in seconds (TUM,
    benchmark_euroc output).
    """
    timestamps = traj.timestamps
    if len(timestamps) and timestamps[0] > _NANOSECOND_THRESHOLD:
        from evo.core.trajectory import PoseTrajectory3D
        return PoseTrajectory3D(poses_se3=traj.poses_se3, timestamps=timestamps / 1e9)
    return traj
