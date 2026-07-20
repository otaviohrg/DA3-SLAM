"""
Download the TUM RGB-D freiburg1 sequences used in the DA3-SLAM benchmark.

Sequences: 360, desk, desk2, floor, plant, room, rpy, teddy, xyz
Target:    data/tum/<sequence_name>/

Usage:
    python scripts/download_tum.py
    python scripts/download_tum.py --data_dir /custom/path
    python scripts/download_tum.py --seq desk xyz          # subset only
    python scripts/download_tum.py --check                 # verify existing downloads
"""

import argparse
import tarfile
import time
import urllib.request
from pathlib import Path

# ── sequence catalogue ────────────────────────────────────────────────────────
_BASE = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1"

SEQUENCES = {
    "360":   f"{_BASE}/rgbd_dataset_freiburg1_360.tgz",
    "desk":  f"{_BASE}/rgbd_dataset_freiburg1_desk.tgz",
    "desk2": f"{_BASE}/rgbd_dataset_freiburg1_desk2.tgz",
    "floor": f"{_BASE}/rgbd_dataset_freiburg1_floor.tgz",
    "plant": f"{_BASE}/rgbd_dataset_freiburg1_plant.tgz",
    "room":  f"{_BASE}/rgbd_dataset_freiburg1_room.tgz",
    "rpy":   f"{_BASE}/rgbd_dataset_freiburg1_rpy.tgz",
    "teddy": f"{_BASE}/rgbd_dataset_freiburg1_teddy.tgz",
    "xyz":   f"{_BASE}/rgbd_dataset_freiburg1_xyz.tgz",
}


# ── helpers ───────────────────────────────────────────────────────────────────

class _ProgressHook:
    """Callable progress hook for urllib.request.urlretrieve."""

    def __init__(self, label: str):
        self._label    = label
        self._start    = time.time()
        self._last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct == self._last_pct:
            return
        self._last_pct = pct
        mb       = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        elapsed  = time.time() - self._start
        rate     = mb / elapsed if elapsed > 0 else 0
        bar      = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  {self._label}  [{bar}] {pct:3d}%  "
              f"{mb:.1f}/{total_mb:.1f} MB  {rate:.1f} MB/s",
              end="", flush=True)
        if pct == 100:
            print()


def download_and_extract(name: str, url: str, data_dir: Path) -> Path:
    """Download .tgz and extract to data_dir. Returns extracted sequence path."""
    seq_dir = data_dir / f"rgbd_dataset_freiburg1_{name}"
    tgz_path = data_dir / f"rgbd_dataset_freiburg1_{name}.tgz"

    if seq_dir.exists() and (seq_dir / "rgb.txt").exists():
        print(f"  [{name}] already extracted → {seq_dir}")
        return seq_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if not tgz_path.exists():
        print(f"  [{name}] downloading from {url}")
        urllib.request.urlretrieve(url, tgz_path, _ProgressHook(name))
    else:
        print(f"  [{name}] archive already present, extracting…")

    # Extract
    print(f"  [{name}] extracting…")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir)

    # Remove archive to save space
    tgz_path.unlink()
    print(f"  [{name}] done → {seq_dir}")
    return seq_dir


def check_sequence(name: str, data_dir: Path) -> bool:
    """Return True if sequence is present and has the expected files."""
    seq_dir = data_dir / f"rgbd_dataset_freiburg1_{name}"
    required = ["rgb.txt", "groundtruth.txt", "rgb"]
    ok = all((seq_dir / f).exists() for f in required)
    status = "✓" if ok else "✗ MISSING"
    n_rgb = len(list((seq_dir / "rgb").glob("*.png"))) if (seq_dir / "rgb").exists() else 0
    print(f"  {status}  {name:<10}  {n_rgb:5d} RGB frames  {seq_dir}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """Download (or, with --check, verify) the requested sequences and print
    a ready-to-paste benchmark command for what was downloaded."""
    parser = argparse.ArgumentParser(
        description="Download TUM RGB-D freiburg1 sequences"
    )
    parser.add_argument(
        "--data_dir", default="data/tum",
        help="Root directory for TUM sequences (default: data/tum)",
    )
    parser.add_argument(
        "--seq", nargs="*", default=list(SEQUENCES.keys()),
        choices=list(SEQUENCES.keys()),
        help="Which sequences to download (default: all 9)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only verify existing downloads, do not download",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.check:
        print(f"\nChecking sequences in {data_dir}/\n")
        results = {name: check_sequence(name, data_dir) for name in args.seq}
        missing = [n for n, ok in results.items() if not ok]
        if missing:
            print(f"\nMissing: {', '.join(missing)}")
            print("Run without --check to download them.")
        else:
            print("\nAll sequences present.")
        return

    print(f"\nDownloading {len(args.seq)} TUM freiburg1 sequence(s) → {data_dir}/\n")
    downloaded = []
    failed = []
    for name in args.seq:
        try:
            path = download_and_extract(name, SEQUENCES[name], data_dir)
            downloaded.append((name, path))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            failed.append(name)

    print(f"\n{'─'*60}")
    print(f"Downloaded: {len(downloaded)}/{len(args.seq)}")
    for name, path in downloaded:
        print(f"  ✓  {name:<10} → {path}")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    print("\nTo run the benchmark:")
    seq_dirs = " \\\n              ".join(
        f"data/tum/rgbd_dataset_freiburg1_{n}" for n, _ in downloaded
    )
    print(f"  python scripts/benchmark_tum.py \\\n"
          f"      --seq_dir {seq_dirs} \\\n"
          f"      --out_dir /app/outputs/benchmark")


if __name__ == "__main__":
    main()
