"""
CLIP-based semantic embedding for open-vocabulary scene queries.

Wraps a HuggingFace CLIP model to produce per-frame image embeddings
and text embeddings. Embeddings are stored on each Submap so the full
trajectory can be searched with a text query.

Requires: pip install transformers
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class SemanticEmbedder:
    """
    Produces L2-normalised CLIP embeddings for images and text.

    Usage:
        embedder = SemanticEmbedder()
        for submap in submaps:
            vectors = embedder.encode_frames(submap)
            submap.set_all_semantic_vectors(vectors)

        text_vec = embedder.encode_text("coffee machine")
        best = result.retrieve_best_semantic_frame(text_vec)
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: torch.device | None = None,
    ):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for semantic embeddings. "
                "Install it with: pip install transformers"
            ) from exc

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[SemanticEmbedder] Loading {model_id} on {self.device}...")
        self._model     = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)
        print("[SemanticEmbedder] Ready.")

    @torch.no_grad()
    def encode_frames(self, submap) -> list[np.ndarray]:
        """
        Encode every frame in a submap as an L2-normalised CLIP image embedding.

        Returns a list of (D,) float32 arrays, one per frame, in the same
        order as submap.frames.
        """
        vectors = []
        for frame in submap.frames:
            pil_image = Image.fromarray(frame.image)
            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self._model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            vectors.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
        return vectors

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query as an L2-normalised CLIP text embedding.

        Returns (D,) float32.
        """
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self._model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)
