"""
Loop closure detection and pose estimation.

Detection:
    Every frame of a new submap is described by a DINO-SALAD descriptor and
    matched (L2 distance) against all stored per-frame descriptors of all
    eligible previous submaps.  A fixed-capacity priority queue keeps the
    top-K candidates (lowest distance = most similar).

Verification & transform estimation:
    DA3 is re-run on the matched frames plus up to `context_frames` temporal
    neighbours of each (wide-baseline 2-frame inference is poorly
    conditioned; the neighbours give DA3 multi-view support).  The relative
    pose between the two matched frames in that fresh inference defines the
    loop constraint directly — no ICP or point cloud matching.  A mean
    depth-confidence gate over the matched frames (analogous to VGGT-SLAM's
    image_match_ratio threshold) rejects bad pairs.

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

    `reinference_submap` is the submap built from the DA3 re-inference on
    the matched frames (plus their context neighbours); the query frame sits
    at `query_frame_pos` and the detected frame at `detected_frame_pos`.
    Its depth maps are kept so the caller can estimate the metric scale of
    the constraint (see DA3SLAM._loop_closure_worker).
    """
    candidate: LoopCandidate

    # Submap from re-inference: matched frames + temporal context neighbours
    reinference_submap: Submap

    # (4, 4) float64 — measured relative cam-to-world transform from the
    # query frame (b) to the detected frame (a), expressed in the
    # re-inference submap's own (arbitrary) metric scale.  This is exactly
    # the between-factor measurement for cam-to-world graph nodes:
    # node_b.inverse() ⊗ node_a ≈ relative_b_to_a.
    relative_b_to_a: np.ndarray

    # Mean DA3 depth confidence over the two matched re-inference frames
    # (diagnostic)
    mean_confidence: float

    # Positions of the query / detected frames within reinference_submap
    query_frame_pos: int = 0
    detected_frame_pos: int = 1


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

        # All previously seen (non-loop-closure) submaps, keyed by submap idx
        self._submaps: dict[int, Submap] = {}
        # Per-frame descriptors indexed by (submap_idx, frame_idx)
        self._frame_descriptors: dict[tuple[int, int], np.ndarray] = {}

        # Loop-closure submap indices are negative to avoid collision with
        # regular submaps
        self._next_loop_closure_idx = -1

        self._model = self._load_salad_model()
        # RoMa is loaded on first use so a run with the gate off never pays
        # for it (1 GB checkpoint + a DINOv3 backbone).
        self._roma = None

    def reset(self) -> None:
        """Clear all per-sequence state — seen submaps, the per-frame
        descriptor index, and the loop-closure submap counter — while keeping
        the loaded DINO-SALAD model.

        Lets one detector be reused across sequences (e.g. SharedSLAM in the
        benchmark/sweep drivers) instead of being rebuilt: rebuilding reloads
        DINO-SALAD, which re-validates DINOv2 against GitHub through torch.hub
        and can fail mid-run on a transient network error.
        """
        self._submaps.clear()
        self._frame_descriptors.clear()
        self._next_loop_closure_idx = -1

    def _roma_overlap(self, image_b: np.ndarray, image_a: np.ndarray) -> float | None:
        """RoMa v2 mean predicted overlap between two frames, or None on failure.

        This is spatial verification the global descriptor cannot provide: it
        measures how much of the two views actually correspond, rather than how
        close their whole-image embeddings are.  A failure returns None and the
        candidate is passed through to the normal gates rather than dropped, so
        a broken matcher cannot silently suppress every closure.
        """
        try:
            if self._roma is None:
                from romav2 import RoMaV2
                print("[LoopClosure] Loading RoMa v2 for dense verification...")
                self._roma = RoMaV2()
            import tempfile, cv2, os
            paths = []
            for img in (image_b, image_a):
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                paths.append(path)
            try:
                preds = self._roma.match(paths[0], paths[1])
                _, overlaps, _, _ = self._roma.sample(preds, 5000)
                ov = overlaps.detach().cpu().numpy() if hasattr(overlaps, "detach") \
                    else np.asarray(overlaps)
                return float(np.mean(ov))
            finally:
                for path in paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        except Exception as exc:
            print(f"[LoopClosure] dense gate unavailable ({exc}); "
                  f"candidate passed to the standard gates")
            return None

    def _load_salad_model(self) -> torch.nn.Module:
        """Load DINO-SALAD, downloading the checkpoint on first use."""
        print("[LoopClosure] Loading DINO-SALAD...")
        from salad.eval import load_model
        checkpoint_path = os.path.join(
            torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
        if not os.path.exists(checkpoint_path):
            print(f"[LoopClosure] Checkpoint not found — downloading to {checkpoint_path}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.hub.download_url_to_file(self._SALAD_CHECKPOINT_URL, checkpoint_path)
        model = load_model(checkpoint_path).to(self.device).eval()
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

        descriptors = self._extract_per_frame_descriptors(submap)
        submap.set_all_retrieval_vectors(descriptors)
        for frame_idx, descriptor in enumerate(descriptors):
            self._frame_descriptors[(submap.idx, frame_idx)] = descriptor

        closures = []
        for candidate in self._dedupe_candidates(self._find_candidates(submap)):
            closure = self._verify(candidate)
            if closure is not None:
                closures.append(closure)
        return closures

    def verify_boundary(
        self,
        prev_submap_idx: int,
        frame_idx_a: int,
        curr_submap_idx: int,
        frame_idx_b: int,
    ) -> LoopClosure | None:
        """Verify a known-adjacent frame pair spanning a submap boundary.

        Used for boundary *repair*: when the two batches disagree on a
        boundary (scale break / pose break), a fresh DA3 inference of a frame
        from each side gives a third, independent measurement of the relative
        pose that arbitrates the disagreement.  Same verification path as a
        retrieval match (context frames, confidence gate), but there is no
        descriptor distance — the frames are adjacent by construction.
        """
        candidate = LoopCandidate(
            submap_idx_a=prev_submap_idx,
            frame_idx_a=frame_idx_a,
            submap_idx_b=curr_submap_idx,
            frame_idx_b=frame_idx_b,
            distance=0.0,
        )
        return self._verify(candidate)

    @staticmethod
    def _dedupe_candidates(candidates: list[LoopCandidate]) -> list[LoopCandidate]:
        """Keep only the best candidate per detected submap.

        Near-duplicate matches (same submap pair, neighbouring frames) each
        cost a full DA3 re-inference while adding little independent evidence;
        one verification per detected submap keeps the top-K slots diverse —
        which also makes downstream corroboration more meaningful — and bounds
        the GPU load of a match-heavy submap.  `candidates` is sorted
        best-first, so the first hit per submap wins.
        """
        best: dict[int, LoopCandidate] = {}
        for candidate in candidates:
            best.setdefault(candidate.submap_idx_a, candidate)
        return list(best.values())

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
            self._TRANSFORM(PILImage.fromarray(frame.image))
            for frame in submap.frames
        ]).to(self.device)                               # (N, 3, 224, 224)

        features = self._model(tensors).cpu().numpy().astype(np.float32)  # (N, D)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return [features[i] for i in range(len(submap.frames))]

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
            and not submap.is_loop_closure_submap
        ]

        best_distance = float("inf")
        best_match = None  # (idx_a, frame_idx_a, frame_idx_b) of best_distance
        for frame_idx_b, frame_b in enumerate(query_submap.frames):
            if frame_b.retrieval_vector is None:
                continue
            for idx_a in eligible:
                for frame_idx_a in range(len(self._submaps[idx_a].frames)):
                    descriptor_a = self._frame_descriptors.get((idx_a, frame_idx_a))
                    if descriptor_a is None:
                        continue
                    distance = float(np.linalg.norm(frame_b.retrieval_vector - descriptor_a))
                    if distance < best_distance:
                        best_distance = distance
                        best_match = (idx_a, frame_idx_a, frame_idx_b)
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
            where = (f" [f{best_match[2]}] → submap {best_match[0]}"
                     f"[f{best_match[1]}]" if best_match else "")
            print(f"[LoopClosure] submap {query_submap.idx}: "
                  f"best descriptor distance {best_str}{where} "
                  f"(threshold {config.distance_threshold:g}), "
                  f"{len(eligible)} eligible submap(s), "
                  f"{len(candidates)} candidate(s)")
        return candidates

    # ── re-inference verification ─────────────────────────────────────────────

    @staticmethod
    def _context_window(
        submap: Submap, frame_idx: int, context: int,
    ) -> tuple[list[np.ndarray], int]:
        """Images of frame_idx ± context (clipped) and frame_idx's position within."""
        lo = max(0, frame_idx - context)
        hi = min(len(submap.frames) - 1, frame_idx + context)
        images = [submap.frames[i].image for i in range(lo, hi + 1)]
        return images, frame_idx - lo

    def _verify(self, candidate: LoopCandidate) -> LoopClosure | None:
        """
        Verify a candidate by re-running DA3 on the matched frames (each with
        up to `context_frames` temporal neighbours for multi-view support) and
        derive the loop constraint from the resulting relative pose.

        All frames of the fresh inference share one consistent (if arbitrary)
        local world, so the relative cam-to-world transform between the two
        matched frames is well defined:

            relative_b_to_a = world_to_cam(query) @ cam_to_world(detected)

        The translation is in the re-inference's own metric scale; the caller
        rescales it before inserting the factor into the graph.
        """
        submap_a = self._submaps[candidate.submap_idx_a]  # detected
        submap_b = self._submaps[candidate.submap_idx_b]  # query

        # Dense-matching gate runs BEFORE the DA3 re-inference: rejecting here
        # saves the more expensive multi-view forward on a bad candidate.
        if self.config.roma_gate:
            overlap = self._roma_overlap(
                submap_b.frames[candidate.frame_idx_b].image,
                submap_a.frames[candidate.frame_idx_a].image)
            if overlap is not None and overlap < self.config.roma_min_overlap:
                print(f"[LoopClosure] {candidate.submap_idx_a}"
                      f"[f{candidate.frame_idx_a}]"
                      f"↔{candidate.submap_idx_b}[f{candidate.frame_idx_b}] "
                      f"rejected by dense gate (overlap {overlap:.3f} < "
                      f"{self.config.roma_min_overlap:.3f})")
                return None

        tag = (
            f"[LoopClosure] "
            f"{candidate.submap_idx_a}[f{candidate.frame_idx_a}]"
            f"↔{candidate.submap_idx_b}[f{candidate.frame_idx_b}]"
            f"  dist={candidate.distance:.3f}"
        )

        # Batch = query window then detected window; the matched frames'
        # positions inside the batch are tracked explicitly.
        context = max(0, int(self.config.context_frames))
        images_b, query_pos = self._context_window(
            submap_b, candidate.frame_idx_b, context)
        images_a, pos_in_window_a = self._context_window(
            submap_a, candidate.frame_idx_a, context)
        detected_pos = len(images_b) + pos_in_window_a

        prediction = self.builder.estimator.infer(images_b + images_a)

        # Quality gate: mean depth confidence over the two *matched* frames
        # (the context frames only serve to condition the inference).
        mean_confidence = float(np.mean(
            prediction.confidence[[query_pos, detected_pos]]))
        if mean_confidence < self.config.min_confidence_ratio:
            print(f"{tag}  REJECTED (confidence={mean_confidence:.3f} "
                  f"< {self.config.min_confidence_ratio})")
            return None

        submap_idx = self._next_loop_closure_idx
        self._next_loop_closure_idx -= 1
        reinference_submap = self.builder.build_from_prediction(prediction, submap_idx)
        reinference_submap.is_loop_closure_submap = True

        extrinsic_query = (
            reinference_submap.frames[query_pos].extrinsic.astype(np.float64))
        extrinsic_detected = (
            reinference_submap.frames[detected_pos].extrinsic.astype(np.float64))
        relative_b_to_a = extrinsic_query @ np.linalg.inv(extrinsic_detected)

        print(f"{tag}  conf={mean_confidence:.3f}  idx={submap_idx}  "
              f"({len(images_b) + len(images_a)} frames)  ACCEPTED")
        return LoopClosure(
            candidate=candidate,
            reinference_submap=reinference_submap,
            relative_b_to_a=relative_b_to_a,
            mean_confidence=mean_confidence,
            query_frame_pos=query_pos,
            detected_frame_pos=detected_pos,
        )
