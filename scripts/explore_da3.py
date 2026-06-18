"""
Step 1: Explore Depth Anything 3 outputs on a batch of images.

Usage:
    python scripts/explore_da3.py --images path/to/img1.jpg path/to/img2.jpg ...
    python scripts/explore_da3.py --image_dir path/to/folder --max_frames 8
    python scripts/explore_da3.py --image_dir path/to/folder --save_dir outputs/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_paths(args) -> list[str]:
    if args.images:
        paths = args.images
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        )
    else:
        print("Error: provide --images or --image_dir")
        sys.exit(1)

    if args.max_frames:
        paths = paths[: args.max_frames]

    if not paths:
        print("No images found.")
        sys.exit(1)

    print(f"Loaded {len(paths)} image(s):")
    for p in paths:
        print(f"  {p}")
    return paths


def print_array(name: str, arr) -> None:
    if arr is None:
        print(f"  {name}: None")
        return
    a = np.asarray(arr)
    print(
        f"  {name}: shape={a.shape}  dtype={a.dtype}"
        f"  min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}"
    )


def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def save_depth_colormap(depth: np.ndarray, path: str) -> None:
    """Save a (H, W) depth map as a colourised PNG."""
    import cv2
    d = depth.astype(np.float32)
    # normalise to 0-255
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    d_uint8 = (d * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, coloured)


def save_conf_map(conf: np.ndarray, path: str) -> None:
    import cv2
    c = conf.astype(np.float32)
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c = (c - c_min) / (c_max - c_min)
    cv2.imwrite(path, (c * 255).astype(np.uint8))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore DA3 outputs")
    parser.add_argument("--images", nargs="+", help="Explicit image paths")
    parser.add_argument("--image_dir", help="Directory of images")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="Max frames to process (default: 8)")
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE",
                        help="HuggingFace model ID")
    parser.add_argument("--process_res", type=int, default=504,
                        help="Processing resolution (default: 504)")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save depth/conf visualisations")
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ── images ────────────────────────────────────────────────────────────────
    image_paths = load_image_paths(args)

    # ── model ─────────────────────────────────────────────────────────────────
    print_section("Loading model")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()
    print(f"  Model: {args.model}")

    # ── inference ─────────────────────────────────────────────────────────────
    print_section("Running inference")
    with torch.no_grad():
        pred = model.inference(
            image_paths,
            process_res=args.process_res,
        )
    print("  Done.")

    # ── inspect outputs ───────────────────────────────────────────────────────
    print_section("Prediction fields")

    print_array("depth",             pred.depth)
    print_array("conf",              pred.conf)
    print_array("sky",               pred.sky)
    print_array("extrinsics",        pred.extrinsics)
    print_array("intrinsics",        pred.intrinsics)
    print_array("processed_images",  pred.processed_images)

    is_metric = getattr(pred, "is_metric", None)
    scale_factor = getattr(pred, "scale_factor", None)
    print(f"  is_metric:    {is_metric}")
    print(f"  scale_factor: {scale_factor}")

    aux = getattr(pred, "aux", None)
    if aux:
        print(f"  aux keys:     {list(aux.keys())}")

    # ── per-frame depth stats ─────────────────────────────────────────────────
    if pred.depth is not None:
        print_section("Per-frame depth stats")
        for i, d in enumerate(pred.depth):
            print(
                f"  frame {i:02d}  min={d.min():.3f}  max={d.max():.3f}"
                f"  mean={d.mean():.3f}  std={d.std():.3f}"
            )

    # ── camera poses ─────────────────────────────────────────────────────────
    if pred.extrinsics is not None:
        print_section("Estimated camera extrinsics  [world-to-cam 4x4]")
        for i, ext in enumerate(pred.extrinsics):
            R = ext[:3, :3]
            t = ext[:3, 3]
            print(f"  frame {i:02d}  t={t}  det(R)={np.linalg.det(R):.4f}")

    if pred.intrinsics is not None:
        print_section("Estimated camera intrinsics  [K 3x3]")
        for i, K in enumerate(pred.intrinsics):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            print(f"  frame {i:02d}  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    # ── save visuals ──────────────────────────────────────────────────────────
    if args.save_dir and pred.depth is not None:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for i, depth in enumerate(pred.depth):
            save_depth_colormap(depth, str(save_path / f"depth_{i:02d}.png"))

        if pred.conf is not None:
            for i, conf in enumerate(pred.conf):
                save_conf_map(conf, str(save_path / f"conf_{i:02d}.png"))

        print_section(f"Saved visualisations to {args.save_dir}")
        for f in sorted(save_path.iterdir()):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
