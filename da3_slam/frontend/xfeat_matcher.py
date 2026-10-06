"""
XFeat matching for the tracker's relocalisation path.

ORB is what relocalisation ships with, and the measurements say correspondence
quality — not map access — is what limits the live trajectory: relocalisation
succeeded on 287 of 287 attempts while PnP's RANSAC rejected 1150 frames, i.e.
the matches were there and geometrically wrong.  XFeat is a learned detector
and descriptor built for real-time CPU inference, which matters here because
the GPU is the contended resource (DA3 runs at ~0.73 duty, 1.71 when loop
closure is hot).  It therefore defaults to CPU and stays out of DA3's way.

The model is loaded through `torch.hub` from the upstream repository, cached
under TORCH_HOME (a persisted volume in the Docker setup), so nothing is
vendored and nothing is added to requirements.txt.  Import and load are both
lazy: a run with `matcher: orb` never touches torch on this path.
"""

from __future__ import annotations

import threading

import numpy as np

_REPO = "verlab/accelerated_features"
_lock = threading.Lock()
_model = None
_device = None


def available() -> bool:
    """Can XFeat be loaded at all (torch present)?"""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def load(device: str = "cpu"):
    """Load XFeat once per process; returns None if it cannot be obtained.

    Failure is not fatal anywhere: the tracker falls back to ORB, which needs
    no weights and no network.
    """
    global _model, _device
    with _lock:
        if _model is not None and _device == device:
            return _model
        try:
            import torch
            model = torch.hub.load(_REPO, "XFeat", pretrained=True,
                                   trust_repo=True, top_k=4096)
            model = model.to(device).eval()
            # XFeat moves its own inputs to `self.dev`, which it picks at
            # construction (cuda when visible).  Moving only the weights leaves
            # the input on the GPU and the model on the CPU, which fails inside
            # the first conv — so the choice has to be told to both.
            model.dev = torch.device(device)
            _model, _device = model, device
            return _model
        except Exception as exc:
            print(f"[tracker] XFeat unavailable ({exc!r}); falling back to ORB")
            _model, _device = None, device
            return None


def detect(gray: np.ndarray, device: str = "cpu", top_k: int = 2048):
    """(keypoints (N,2) float32, feature dict) for one grayscale image."""
    model = load(device)
    if model is None:
        return None, None
    import torch
    # XFeat wants (B, C, H, W) float; a grayscale frame is replicated to 3
    # channels because the network was trained on RGB.
    tensor = torch.from_numpy(np.repeat(gray[None, None], 3, axis=1)).float()
    with torch.inference_mode():
        output = model.detectAndCompute(tensor.to(device), top_k=top_k)[0]
    points = output["keypoints"].cpu().numpy().astype(np.float32)
    return points, output


def match(features_a, features_b, min_cossim: float = 0.82):
    """Mutual-nearest-neighbour match between two detect() outputs.

    Returns (indices into a, indices into b).  `min_cossim` is XFeat's own
    confidence gate and plays the role Lowe's ratio plays for ORB.
    """
    model = load(_device or "cpu")
    if model is None or features_a is None or features_b is None:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    import torch
    with torch.inference_mode():
        index_a, index_b = model.match(features_a["descriptors"],
                                       features_b["descriptors"],
                                       min_cossim=min_cossim)
    return (np.asarray(index_a.cpu()).astype(int),
            np.asarray(index_b.cpu()).astype(int))
