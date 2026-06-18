"""
Loop closure detection and pose estimation.

Detection:
    Every frame of a new submap is described by a DINO-SALAD descriptor and
    matched (L2 distance) against all stored per-frame descriptors of all
    eligible previous submaps.  A fixed-capacity priority queue keeps the
    top-K candidates (lowest distance = most similar).

Verification & transform estimation:
    DA3 is re-run on the matched image pair [query_frame, detected_frame].
    The relative pose between the two frames in that fresh 2-frame inference
    defines the loop constraint directly — no ICP or point cloud matching.
    A mean depth-confidence gate (analogous to VGGT-SLAM's image_match_ratio
    threshold) rejects bad pairs.

Graph integration (performed by DA3SLAM, see slam.py):
    Each accepted LoopClosure carries `relative_b_to_a`, the measured
    cam-to-world transform from the query frame to the detected frame.
    DA3SLAM scales its translation to the global metric unit and posts a
    single between-factor between the two keyframe nodes
    (PoseGraph.add_between(..., loop=True)).
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PILImage

from da3_slam.backend.inference.submap import Submap, SubmapBuilder

# Re-exported so callers can import the config next to the component it tunes.
from da3_slam.config import LoopClosureConfig

__all__ = [
    "LoopClosureConfig",
    "LoopCandidate",
    "LoopClosure",
    "LoopMatchQueue",
    "LoopClosureDetector",
]


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class LoopCandidate:
    """Frame-level loop closure candidate from descriptor matching.

    Naming convention: "a" is the detected (older) side of the loop,
    "b" is the query (current) side.
    """
    submap_idx_a: int    # detected (older) submap
    frame_idx_a: int     # frame index within the detected submap
    submap_idx_b: int    # query (current) submap
    frame_idx_b: int     # frame index within the query submap
    distance: float      # L2 norm of descriptor difference (lower = better)


@dataclass
class LoopClosure:
    """Verified loop closure.

    `lc_submap` is the 2-frame submap built from the DA3 re-inference on
    [query_frame, detected_frame]; its frame 0 is the query and frame 1 the
    detected frame.  Its depth maps are kept so the caller can estimate the
    metric scale of the constraint (see DA3SLAM._loop_closure_worker).
    """
    candidate: LoopCandidate

    # 2-frame submap from re-inference: frames[0]=query, frames[1]=detected
    lc_submap: Submap

    # (4, 4) float64 — measured relative cam-to-world transform from the
    # query frame (b) to the detected frame (a), expressed in the LC
    # submap's own (arbitrary) metric scale.  This is exactly the
    # between-factor measurement for cam-to-world graph nodes:
    # node_b.inverse() ⊗ node_a ≈ relative_b_to_a.
    relative_b_to_a: np.ndarray

    # Mean DA3 depth confidence over both LC frames (diagnostic)
    lc_confidence: float


# ── priority queue ────────────────────────────────────────────────────────────

class LoopMatchQueue:
    """
    Fixed-capacity priority queue that keeps the N best LoopCandidates.

    Internally a max-heap (on negated distance) of size max_size so that
    heappushpop evicts the worst (largest distance) element when full.
    A monotone counter breaks ties without requiring LoopCandidate.__lt__.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._heap: list = []
        self._counter = 0

    def push(self, candidate: LoopCandidate) -> None:
        if self.max_size <= 0:
            return
        item = (-candidate.distance, self._counter, candidate)
        self._counter += 1
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, item)
        else:
            heapq.heappushpop(self._heap, item)

    def get_best(self) -> list[LoopCandidate]:
        """Return candidates sorted by distance ascending (best first)."""
        return [c for _, _, c in sorted(self._heap, reverse=True)]


# ── detector ──────────────────────────────────────────────────────────────────

class LoopClosureDetector:
    """
    Detects and verifies loop closures using frame-level DINO-SALAD matching
    followed by DA3 re-inference on the matched image pair.

    Usage:
        detector = LoopClosureDetector(config, builder)
        for submap in submaps:
            for closure in detector.process(submap):
                # scale closure.relative_b_to_a translation to global units,
                # then post a between-factor (see DA3SLAM._loop_closure_worker)
                ...
    """

    # SALAD input size (224×224) and ImageNet normalisation — mirrors VGGT-SLAM
    _INPUT_SIZE = 224
    _TRANSFORM = T.Compose([
        T.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _SALAD_CHECKPOINT_URL = (
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    )

    def __init__(
        self,
        config: LoopClosureConfig,
        builder: SubmapBuilder,
        device: torch.device | None = None,
    ):
        self.config = config
        self.builder = builder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # All previously seen (non-LC) submaps, keyed by submap idx
        self._submaps: dict[int, Submap] = {}
        # Per-frame descriptors indexed by (submap_idx, frame_idx)
        self._frame_descriptors: dict[tuple[int, int], np.ndarray] = {}

        # LC submap indices are negative to avoid collision with regular submaps
        self._next_lc_idx = -1

        self._model = self._load_salad_model()

    def _load_salad_model(self) -> torch.nn.Module:
        """Load DINO-SALAD, downloading the checkpoint on first use."""
        print("[LoopClosure] Loading DINO-SALAD...")
        from salad.eval import load_model
        ckpt_path = os.path.join(torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[LoopClosure] Checkpoint not found — downloading to {ckpt_path}")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.hub.download_url_to_file(self._SALAD_CHECKPOINT_URL, ckpt_path)
        model = load_model(ckpt_path).to(self.device).eval()
        print("[LoopClosure] Ready.")
        return model

    @torch.no_grad()
    def process(self, submap: Submap) -> list[LoopClosure]:
        """
        Register a new submap and return verified loop closures against
        previously registered submaps.

        Side effect: stores per-frame DINO-SALAD descriptors on the submap's
        frames (frame.retrieval_vector) and in the detector's index.
        """
        self._submaps[submap.idx] = submap

        per_frame = self._extract_per_frame_descriptors(submap)
        submap.set_all_retrieval_vectors(per_frame)
        for frame_idx, descriptor in enumerate(per_frame):
            self._frame_descriptors[(submap.idx, frame_idx)] = descriptor

        closures = []
        for candidate in self._find_candidates(submap):
            closure = self._verify(candidate)
            if closure is not None:
                closures.append(closure)
        return closures

    # ── descriptor extraction ─────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_per_frame_descriptors(self, submap: Submap) -> list[np.ndarray]:
        """
        Extract one L2-normalised DINO-SALAD descriptor per frame in the submap.

        All frames are processed as a single batch.  Each (H, W, 3) uint8
        image is resized to 224×224 and normalised with ImageNet statistics —
        identical to VGGT-SLAM's ImageRetrieval.get_batch_descriptors().
        """
        tensors = torch.stack([
            self._TRANSFORM(PILImage.fromarray(f.image))
            for f in submap.frames
        ]).to(self.device)                               # (N, 3, 224, 224)

        feats = self._model(tensors).cpu().numpy().astype(np.float32)  # (N, D)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return [feats[i] for i in range(len(submap.frames))]

    # ── frame-level candidate detection ──────────────────────────────────────

    def _find_candidates(self, query_submap: Submap) -> list[LoopCandidate]:
        """
        Compare every frame of query_submap against every stored frame of all
        eligible previous submaps using L2 distance on DINO-SALAD descriptors.

        Returns the top-K candidates (K = max_loop_closures) below the
        distance threshold.
        """
        config = self.config
        queue = LoopMatchQueue(config.max_loop_closures)

        eligible = [
            idx for idx, submap in self._submaps.items()
            if abs(idx - query_submap.idx) > config.min_submaps_apart
            and not submap.is_lc_submap
        ]

        best_distance = float("inf")
        for frame_idx_b, frame_b in enumerate(query_submap.frames):
            if frame_b.retrieval_vector is None:
                continue
            for idx_a in eligible:
                for frame_idx_a in range(len(self._submaps[idx_a].frames)):
                    descriptor_a = self._frame_descriptors.get((idx_a, frame_idx_a))
                    if descriptor_a is None:
                        continue
                    distance = float(np.linalg.norm(frame_b.retrieval_vector - descriptor_a))
                    best_distance = min(best_distance, distance)
                    if distance < config.distance_threshold:
                        queue.push(LoopCandidate(
                            submap_idx_a=idx_a,
                            frame_idx_a=frame_idx_a,
                            submap_idx_b=query_submap.idx,
                            frame_idx_b=frame_idx_b,
                            distance=distance,
                        ))

        candidates = queue.get_best()
        # Diagnostic: the best distance seen (even above threshold) tells you
        # where the threshold should sit for the current dataset/domain.
        if eligible:
            best_str = f"{best_distance:.3f}" if np.isfinite(best_distance) else "n/a"
            print(f"[LoopClosure] submap {query_submap.idx}: "
                  f"best descriptor distance {best_str} "
                  f"(threshold {config.distance_threshold:g}), "
                  f"{len(eligible)} eligible submap(s), "
                  f"{len(candidates)} candidate(s)")
        return candidates

    # ── re-inference verification ─────────────────────────────────────────────

    def _verify(self, candidate: LoopCandidate) -> LoopClosure | None:
        """
        Verify a candidate by re-running DA3 on [query_frame, detected_frame]
        and derive the loop constraint from the resulting relative pose.

        The two frames of the fresh inference share one consistent (if
        arbitrary) local world, so the relative cam-to-world transform between
        them is well defined:

            relative_b_to_a = w2c_query_lc @ c2w_detected_lc
                            = lc_frame0.extrinsic @ inv(lc_frame1.extrinsic)

        The translation is in the LC inference's own metric scale; the caller
        rescales it before inserting the factor into the graph.
        """
        submap_a = self._submaps[candidate.submap_idx_a]  # detected
        submap_b = self._submaps[candidate.submap_idx_b]  # query
        frame_a = submap_a.frames[candidate.frame_idx_a]
        frame_b = submap_b.frames[candidate.frame_idx_b]

        tag = (
            f"[LoopClosure] "
            f"{candidate.submap_idx_a}[f{candidate.frame_idx_a}]"
            f"↔{candidate.submap_idx_b}[f{candidate.frame_idx_b}]"
            f"  dist={candidate.distance:.3f}"
        )

        # Run DA3 on [query_frame, detected_frame] — order matters:
        # LC frame 0 = query (b), LC frame 1 = detected (a)
        prediction = self.builder.estimator.infer([frame_b.image, frame_a.image])

        # Quality gate: mean depth confidence across both LC frames
        mean_conf = float(np.mean(prediction.confidence))
        if mean_conf < self.config.min_confidence_ratio:
            print(f"{tag}  REJECTED (confidence={mean_conf:.3f} "
                  f"< {self.config.min_confidence_ratio})")
            return None

        lc_idx = self._next_lc_idx
        self._next_lc_idx -= 1
        lc_submap = self.builder.build_from_prediction(prediction, lc_idx)
        lc_submap.is_lc_submap = True

        lc_extrinsic_query = lc_submap.frames[0].extrinsic.astype(np.float64)
        lc_extrinsic_detected = lc_submap.frames[1].extrinsic.astype(np.float64)
        relative_b_to_a = lc_extrinsic_query @ np.linalg.inv(lc_extrinsic_detected)

        print(f"{tag}  conf={mean_conf:.3f}  lc_idx={lc_idx}  ACCEPTED")
        return LoopClosure(
            candidate=candidate,
            lc_submap=lc_submap,
            relative_b_to_a=relative_b_to_a,
            lc_confidence=mean_conf,
        )
