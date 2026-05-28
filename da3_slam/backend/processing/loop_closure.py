"""
Loop closure detection and pose estimation.

Detection:
    DINOv2 (ViT-B/14) per-frame L2-norm matching against all stored
    per-frame descriptors across all eligible previous submaps.
    A priority queue keeps the top-K candidates (lowest L2 = most similar).

Verification & transform estimation:
    DA3 is re-run on the matched image pair [query_frame, detected_frame].
    The relative pose from DA3 defines the loop constraint directly,
    replacing the old ICP-based alignment.  A mean depth-confidence gate
    (analogous to VGGT's image_match_ratio threshold) rejects bad pairs.

Graph integration:
    A 2-frame LC submap is built from the re-inference result and wired
    into the pose graph with two factors:
      - sequential factor:    query_submap  →  lc_submap
      - loop closure factor:  detected_submap  ←  lc_submap
    This mirrors VGGT-SLAM's mechanism exactly.
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
from da3_slam.backend.processing.alignment import AlignmentResult


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class LoopClosureConfig:
    # L2 distance threshold for frame-level candidate detection.
    # Lower distance = more similar.
    distance_threshold: float

    # Minimum submap index gap between the query and any candidate
    min_submaps_apart: int

    # Maximum loop closures accepted per submap (priority queue capacity)
    max_loop_closures: int

    # Minimum mean DA3 depth confidence [0, 1] required to accept a closure.
    # Analogous to VGGT's image_match_ratio >= 0.85 gate.
    min_confidence_ratio: float


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class LoopCandidate:
    """Frame-level loop closure candidate from descriptor matching."""
    submap_idx_a: int   # detected (older) submap
    frame_idx_a:  int   # frame index within the detected submap
    submap_idx_b: int   # query (current) submap
    frame_idx_b:  int   # frame index within the query submap
    distance:     float # L2 norm of descriptor difference (lower = better)


@dataclass
class LoopClosure:
    """
    Verified loop closure with a 2-frame LC submap and two graph alignments.

    The LC submap contains [query_frame, detected_frame] as processed by a
    fresh DA3 inference.  Two AlignmentResults express the LC submap's world
    frame relative to both the query and the detected submap's world frames.
    """
    candidate:            LoopCandidate
    lc_submap:            Submap          # 2-frame LC submap from re-inference
    alignment_to_query:   AlignmentResult # world_lc → world_query   (B=LC, A=query)
    alignment_to_detected: AlignmentResult # world_lc → world_detected (B=LC, A=detected)
    lc_confidence:        float           # mean depth confidence (diagnostic)

    @property
    def submap_idx_a(self) -> int:
        return self.candidate.submap_idx_a

    @property
    def submap_idx_b(self) -> int:
        return self.candidate.submap_idx_b


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
    Detects and verifies loop closures using frame-level DINOv2 matching
    followed by DA3 re-inference on the matched image pair.

    Usage:
        detector = LoopClosureDetector(config, builder)
        for submap in submaps:
            closures = detector.process(submap, graph_pose)
            for closure in closures:
                pose_graph.add_lc_submap(
                    closure.lc_submap,
                    closure.candidate.submap_idx_b,
                    closure.alignment_to_query,
                )
                pose_graph.add_loop_closure(
                    closure.candidate.submap_idx_a,
                    closure.lc_submap.idx,
                    closure.alignment_to_detected,
                )
    """

    # SALAD input size (224×224) and ImageNet normalisation — mirrors VGGT-SLAM
    _INPUT_SIZE = 224
    _TRANSFORM  = T.Compose([
        T.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        config: LoopClosureConfig | None = None,
        builder: SubmapBuilder | None = None,
        device: torch.device | None = None,
    ):
        self.config  = config or LoopClosureConfig()
        self.builder = builder
        self.device  = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._submaps:            dict[int, Submap]              = {}
        # Per-frame descriptors indexed by (submap_idx, frame_idx)
        self._frame_descriptors:  dict[tuple[int, int], np.ndarray] = {}
        # Submap-level aggregated descriptor (mean of all frames, for diagnostics)
        self._descriptors:        dict[int, np.ndarray]          = {}

        # LC submap indices are negative to avoid collision with regular submaps
        self._next_lc_idx = -1

        print("[LoopClosure] Loading DINO-SALAD...")
        from salad.eval import load_model
        ckpt_path = os.path.join(torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[LoopClosure] Checkpoint not found — downloading to {ckpt_path}")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.hub.download_url_to_file(
                "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
                ckpt_path,
            )
        self._model: torch.nn.Module = load_model(ckpt_path).to(self.device).eval()
        print("[LoopClosure] Ready.")

    def update_optimized_poses(self, poses: dict[int, np.ndarray]) -> None:
        """No-op: optimized poses are no longer used (ICP replaced by re-inference)."""
        pass

    @torch.no_grad()
    def process(
        self,
        submap: Submap,
        graph_pose: np.ndarray,
    ) -> list[LoopClosure]:
        """
        Register a new submap and return verified loop closures.

        Populates submap.retrieval_vectors with per-frame DINOv2 descriptors,
        then runs frame-level L2 matching against all previous submaps.
        Each candidate is verified by DA3 re-inference on the matched pair.

        Args:
            submap:     newly built submap
            graph_pose: (4, 4) accumulated pose estimate (retained for API compat)

        Returns:
            list of LoopClosure objects, each carrying a 2-frame LC submap
        """
        self._submaps[submap.idx] = submap

        # Extract per-frame descriptors; store on submap and in local flat index
        per_frame = self._extract_per_frame_descriptors(submap)
        submap.set_all_retrieval_vectors(per_frame)
        for frame_idx, desc in enumerate(per_frame):
            self._frame_descriptors[(submap.idx, frame_idx)] = desc

        # Submap-level aggregated descriptor (for diagnostics / future use)
        mean_desc = np.mean(per_frame, axis=0)
        mean_desc /= np.linalg.norm(mean_desc) + 1e-8
        self._descriptors[submap.idx] = mean_desc

        candidates = self._find_candidates(submap)
        closures   = []
        for candidate in candidates:
            closure = self._verify(candidate)
            if closure is not None:
                closures.append(closure)

        return closures

    # ── descriptor extraction ─────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_per_frame_descriptors(self, submap: Submap) -> list[np.ndarray]:
        """
        Extract one L2-normalised DINO-SALAD descriptor per frame in the submap.

        All frames are processed as a single batch for efficiency.
        Each (H, W, 3) uint8 numpy image is resized to 224×224, normalised
        with ImageNet statistics, then fed to the SALAD model — identical
        to VGGT-SLAM's ImageRetrieval.get_batch_descriptors().
        """
        tensors = torch.stack([
            self._TRANSFORM(PILImage.fromarray(f.image))
            for f in submap.frames
        ]).to(self.device)                               # (N, 3, 224, 224)

        feats = self._model(tensors)                     # (N, D) on device
        feats_np = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats_np, axis=1, keepdims=True)
        feats_np /= norms + 1e-8
        return [feats_np[i] for i in range(len(submap.frames))]

    # ── frame-level candidate detection ──────────────────────────────────────

    def _find_candidates(self, query_submap: Submap) -> list[LoopCandidate]:
        """
        Compare every frame of query_submap against every stored frame of all
        eligible previous submaps using L2 distance on DINOv2 descriptors.

        Returns the top-K candidates from the priority queue (K = max_loop_closures).
        """
        config  = self.config
        queue   = LoopMatchQueue(config.max_loop_closures)

        eligible = [
            idx for idx in self._submaps
            if abs(idx - query_submap.idx) > config.min_submaps_apart
            and not self._submaps[idx].is_lc_submap
        ]

        for frame_idx_b, frame_b in enumerate(query_submap.frames):
            if frame_b.retrieval_vector is None:
                continue
            q_vec = frame_b.retrieval_vector

            for a_idx in eligible:
                a_submap = self._submaps[a_idx]
                for frame_idx_a in range(len(a_submap.frames)):
                    desc_a = self._frame_descriptors.get((a_idx, frame_idx_a))
                    if desc_a is None:
                        continue
                    dist = float(np.linalg.norm(q_vec - desc_a))
                    if dist < config.distance_threshold:
                        queue.push(LoopCandidate(
                            submap_idx_a=a_idx,
                            frame_idx_a=frame_idx_a,
                            submap_idx_b=query_submap.idx,
                            frame_idx_b=frame_idx_b,
                            distance=dist,
                        ))

        return queue.get_best()

    # ── re-inference verification ─────────────────────────────────────────────

    def _verify(self, candidate: LoopCandidate) -> LoopClosure | None:
        """
        Verify a candidate by re-running DA3 on [query_frame, detected_frame].

        Alignment derivation (anchor frame principle):
            Both frames share a physical camera viewpoint with the corresponding
            frame in each original submap.  The world-to-cam extrinsic must be
            identical for the same physical camera, so:

                world_lc_to_world_X = inv(frame_X.extrinsic) @ lc_frameN.extrinsic

            This gives an exact transform without any ICP or point cloud matching.
        """
        if self.builder is None:
            raise RuntimeError(
                "LoopClosureDetector requires a SubmapBuilder for re-inference. "
                "Pass builder= to the constructor."
            )

        submap_a = self._submaps[candidate.submap_idx_a]  # detected
        submap_b = self._submaps[candidate.submap_idx_b]  # query
        frame_a  = submap_a.frames[candidate.frame_idx_a]
        frame_b  = submap_b.frames[candidate.frame_idx_b]

        tag = (
            f"[LoopClosure] "
            f"{candidate.submap_idx_a}[f{candidate.frame_idx_a}]"
            f"↔{candidate.submap_idx_b}[f{candidate.frame_idx_b}]"
            f"  dist={candidate.distance:.3f}"
        )

        # Run DA3 on [query_frame, detected_frame] — order matters:
        # LC frame 0 = query, LC frame 1 = detected
        prediction = self.builder.estimator.infer([frame_b.image, frame_a.image])

        # Quality gate: mean depth confidence across both LC frames
        mean_conf = float(np.mean(prediction.confidence))
        if mean_conf < self.config.min_confidence_ratio:
            print(f"{tag}  REJECTED (confidence={mean_conf:.3f} < {self.config.min_confidence_ratio})")
            return None

        # Build 2-frame LC submap from the re-inference result
        lc_idx    = self._next_lc_idx
        self._next_lc_idx -= 1
        lc_submap = self.builder.build_from_prediction(prediction, lc_idx)
        lc_submap.set_lc_status(True)
        lc_submap.set_last_non_loop_frame_index(1)

        # Anchor frame alignment: derive world_lc_to_world_X for both submaps.
        # lc_submap.frames[0].extrinsic: world_lc → cam  (same cam as frame_b)
        # frame_b.extrinsic:             world_query → cam
        # => world_lc_to_world_query = inv(frame_b.extrinsic) @ lc_frame0.extrinsic
        world_lc_to_world_query = (
            np.linalg.inv(frame_b.extrinsic.astype(np.float64))
            @ lc_submap.frames[0].extrinsic.astype(np.float64)
        )
        world_lc_to_world_detected = (
            np.linalg.inv(frame_a.extrinsic.astype(np.float64))
            @ lc_submap.frames[1].extrinsic.astype(np.float64)
        )

        alignment_to_query = AlignmentResult(
            world_b_to_world_a=world_lc_to_world_query.astype(np.float32),
            method="lc_inference",
            scale=1.0,
        )
        alignment_to_detected = AlignmentResult(
            world_b_to_world_a=world_lc_to_world_detected.astype(np.float32),
            method="lc_inference",
            scale=1.0,
        )

        print(f"{tag}  conf={mean_conf:.3f}  lc_idx={lc_idx}  ACCEPTED")
        return LoopClosure(
            candidate=candidate,
            lc_submap=lc_submap,
            alignment_to_query=alignment_to_query,
            alignment_to_detected=alignment_to_detected,
            lc_confidence=mean_conf,
        )
