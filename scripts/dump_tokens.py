"""
Dump per-frame encoder tokens, depth and inter-keyframe motion (Branch C, Step 0).

Produces the offline material the temporal-redundancy analysis (Step 1) reads:
for one sequence, replayed on a FROZEN keyframe list, every submap's encoder
tokens plus the depth it predicted, and the optical flow between consecutive
keyframes.

Faithfulness to a real run is the point of this script, so it reuses the
pipeline's own pieces rather than approximating them: the same keyframe
selectors, the same `_KeyframeBatcher` (so submap composition and the shared
anchor frame are identical), the same cv2 RGB loading, and the same
`DepthEstimator`.  Run it with `--keyframes_from` so the frames are
byte-identical to whatever else you compare against.

Output layout (`--out_dir`):
    meta.json            model / resolution / submap layout / token shape
    manifest.jsonl       one row per keyframe occurrence (see `_manifest_row`)
    submap_000.npz       tokens (S, n_tokens, dim) fp16, depth + conf (S, H, W) fp16
    flow.npz             per consecutive-keyframe-pair corners + displacements

Size: tokens dominate — ~4 MB per frame at giant/504 (1297 x 1536 fp16).  Use
`--keyframe_stride` / `--max_keyframes` on long sequences, and note that
submaps overlap, so an anchor frame is stored once per submap it appears in.

Usage:
    python scripts/dump_tokens.py --image_dir data/tum/<seq>/rgb \\
        --keyframes_from outputs/frozen/<seq>.txt --out_dir outputs/tokens/<seq>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_slam import collect_image_paths

from da3_slam.config import load_slam_config
from da3_slam.frontend.keyframe_selector import (
    OnlineKeyframeSelector,
    ReplayKeyframeSelector,
    SegmentKeyframeSelector,
    keyframe_flow,
    load_keyframe_list,
    save_keyframe_list,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--keyframes_from", default=None,
                        help="frozen keyframe list to replay (strongly recommended)")
    parser.add_argument("--dump_keyframes", default=None,
                        help="write the selected keyframe list here (when not replaying)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="cap on input frames read from --image_dir")
    parser.add_argument("--max_keyframes", type=int, default=None,
                        help="cap on keyframes dumped (storage guard)")
    parser.add_argument("--keyframe_stride", type=int, default=1,
                        help="keep every Nth keyframe (storage guard; changes submap "
                             "composition, so leave at 1 for numbers that must match a run)")
    parser.add_argument("--submap_size", type=int, default=None)
    parser.add_argument("--depth_model", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--no_tokens", action="store_true",
                        help="dump depth + flow only (cheap; for a quick look at F4)")
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    """cv2 → RGB, exactly as the pipeline's frame source loads frames."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def select_keyframes(image_paths: list[str], config) -> list[tuple[int, str]]:
    """
    (seq_idx, path) of every keyframe, using the pipeline's selectors.

    Mirrors `DA3SLAM._frontend`'s selector choice — replay when a frozen list is
    given, else the configured policy.
    """
    if config.keyframes_from:
        selector = ReplayKeyframeSelector(load_keyframe_list(config.keyframes_from))
        list_mode = True
    elif config.keyframe.selection_mode == "segment":
        selector = SegmentKeyframeSelector(config.keyframe)
        list_mode = True
    else:
        selector = OnlineKeyframeSelector(config.keyframe)
        list_mode = False

    keyframes: list[tuple[int, str]] = []
    for seq_idx, path in enumerate(image_paths):
        image = load_rgb(path)
        if list_mode:
            selected = selector.step(image, seq_idx, path)
        else:
            selected = [(path, image, seq_idx)] if selector.step(image) else []
        keyframes.extend((idx, label) for label, _, idx in selected)
    if list_mode:
        keyframes.extend((idx, label) for label, _, idx in selector.flush())
    return keyframes


def _manifest_row(kf_pos: int, seq_idx: int, label: str, submap_idx: int,
                  pos_in_submap: int, is_anchor: bool, disparity: float) -> dict:
    return {
        "keyframe_index": kf_pos,       # position in the keyframe list
        "seq_idx": seq_idx,             # position in the input frame list
        "label": label,                 # source image path
        "submap_idx": submap_idx,
        "pos_in_submap": pos_in_submap,  # row index inside submap_XXX.npz
        "is_anchor": is_anchor,         # shared with the previous submap
        "mean_disparity_to_prev": disparity,  # px, to the previous KEYFRAME
    }


def main():
    args = parse_args()

    # Deferred: these pull in torch / gtsam / DA3, and everything above works
    # (and `--help` prints) without the GPU stack.
    from da3_slam.slam import _KeyframeBatcher, _effective_overlap

    overrides = {}
    if args.submap_size is not None:
        overrides["submap_size"] = args.submap_size
    if args.depth_model is not None:
        overrides["depth_model"] = args.depth_model
    if args.resolution is not None:
        overrides["depth_model_resolution"] = args.resolution
    config = load_slam_config(args.config, **overrides)
    # Not a load_slam_config override — like run_slam.py, it is set on the
    # returned object (only YAML-backed scalars can be passed as overrides).
    config.keyframes_from = args.keyframes_from

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(args.image_dir, args.max_frames)
    print(f"[dump_tokens] {len(image_paths)} input frames from {args.image_dir}")

    keyframes = select_keyframes(image_paths, config)
    print(f"[dump_tokens] {len(keyframes)} keyframes "
          f"({'replayed' if config.keyframes_from else config.keyframe.selection_mode})")
    if args.dump_keyframes and not config.keyframes_from:
        save_keyframe_list(args.dump_keyframes, keyframes)
        print(f"[dump_tokens] keyframe list → {args.dump_keyframes}")

    if args.keyframe_stride > 1:
        keyframes = keyframes[::args.keyframe_stride]
    if args.max_keyframes:
        keyframes = keyframes[:args.max_keyframes]
    if len(keyframes) < 2:
        raise SystemExit("need at least 2 keyframes to measure redundancy")

    # ── inter-keyframe motion (the signal F3 correlates redundancy against) ───
    # Consecutive keyframe pairs, computed here rather than read off the gate:
    # segment mode's equal-stride fast path never runs optical flow, and replay
    # mode runs none at all.
    flow_arrays: dict[str, np.ndarray] = {}
    disparities = [0.0]
    previous = load_rgb(keyframes[0][1])
    for pair_idx in range(1, len(keyframes)):
        current = load_rgb(keyframes[pair_idx][1])
        points, displacements, mean = keyframe_flow(previous, current, config.keyframe)
        flow_arrays[f"points_{pair_idx - 1:05d}"] = points
        flow_arrays[f"disp_{pair_idx - 1:05d}"] = displacements
        disparities.append(mean)
        previous = current
    np.savez_compressed(out_dir / "flow.npz",
                        mean_disparity=np.asarray(disparities, dtype=np.float32),
                        flow_downsample=np.int32(config.keyframe.flow_downsample_factor),
                        **flow_arrays)
    print(f"[dump_tokens] flow → {out_dir / 'flow.npz'} "
          f"(mean disparity {np.mean(disparities[1:]):.1f} px)")

    # ── batch exactly like the pipeline, then run DA3 with the tap armed ─────
    from da3_slam.backend.inference.depth_estimator import DepthEstimator

    estimator = DepthEstimator(
        model_id=config.depth_model,
        process_resolution=config.depth_model_resolution,
        use_ray_pose=config.use_ray_pose,
    )
    tap_info = estimator.token_tap().info
    print(f"[dump_tokens] {tap_info.describe()}")

    overlap = _effective_overlap(config)
    batcher = _KeyframeBatcher(config.submap_size, overlap)
    batches: list[tuple[list[str], list[np.ndarray], list[int]]] = []
    for kf_pos, (seq_idx, path) in enumerate(keyframes):
        # `label` carries the keyframe's position in the list so the manifest can
        # tie a row back to the flow pairs above.
        batch = batcher.add(f"{kf_pos}|{path}", load_rgb(path), seq_idx)
        if batch is not None:
            batches.append(batch)
    tail = batcher.tail()
    if tail is not None:
        batches.append(tail)

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.unlink(missing_ok=True)
    token_shape = None

    with open(manifest_path, "a") as manifest:
        for submap_idx, (labels, images, indices) in enumerate(batches):
            prediction = estimator.infer(images, capture_tokens=not args.no_tokens)

            arrays = {
                "seq_idx": np.asarray(indices, dtype=np.int32),
                "depth": prediction.depth.astype(np.float16),
                "confidence": prediction.confidence.astype(np.float16),
                "intrinsics": prediction.intrinsics.astype(np.float32),
                "extrinsics": prediction.extrinsics.astype(np.float32),
            }
            if prediction.tokens is not None:
                tokens = np.stack([t.float().cpu().numpy().astype(np.float16)
                                   for t in prediction.tokens])
                arrays["tokens"] = tokens
                token_shape = list(tokens.shape[1:])

            # Uncompressed: tokens are dense float noise (compression buys ~
            # nothing and costs minutes per submap), and the analysis mmaps them.
            np.savez(out_dir / f"submap_{submap_idx:03d}.npz", **arrays)

            for pos, (label, seq_idx) in enumerate(zip(labels, indices)):
                kf_pos, path = label.split("|", 1)
                kf_pos = int(kf_pos)
                manifest.write(json.dumps(_manifest_row(
                    kf_pos=kf_pos,
                    seq_idx=seq_idx,
                    label=path,
                    submap_idx=submap_idx,
                    pos_in_submap=pos,
                    is_anchor=submap_idx > 0 and pos < overlap,
                    disparity=disparities[kf_pos],
                )) + "\n")

            print(f"  submap {submap_idx:03d}: {len(indices)} frames "
                  f"(seq {indices[0]}..{indices[-1]})")

    meta = {
        "image_dir": args.image_dir,
        "n_input_frames": len(image_paths),
        "n_keyframes": len(keyframes),
        "n_submaps": len(batches),
        "submap_size": config.submap_size,
        "submap_overlap": overlap,
        "keyframes_from": config.keyframes_from,
        "selection_mode": config.keyframe.selection_mode,
        "keyframe_stride": args.keyframe_stride,
        "depth_model": config.depth_model,
        "resolution": config.depth_model_resolution,
        "token_shape": token_shape,
        "token_dtype": "float16",
        "backbone": {
            "branch": tap_info.branch,
            "n_blocks": tap_info.n_blocks,
            "alt_start": tap_info.alt_start,
            "prefix_len": tap_info.prefix_len,
            "embed_dim": tap_info.embed_dim,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dump_tokens] {len(batches)} submaps → {out_dir}")


if __name__ == "__main__":
    main()
