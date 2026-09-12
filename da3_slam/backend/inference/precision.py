"""
Backbone weight precision for the DA3 model.

WHY THIS EXISTS
---------------
DA3 ships its checkpoints in fp32, and `profile_da3_memory.py` showed that on
nested-giant @504 the parameters are **55% of peak GPU memory** (6.5 GB of an
11.8 GB peak at 16 frames) — a frame-independent floor under every submap size.

But the two ViT backbones already run under `torch.autocast` in bf16: autocast
converts each fp32 weight to bf16 on every matmul and throws the copy away.  So
for those modules the fp32 master copies are storage the forward never benefits
from.  Storing them as bf16 removes both the storage and the per-matmul copies,
which is why the measured peak drops by ~4.7 GB when the weights themselves
only account for ~2.8 GB.

    fp32 -> bf16 backbones, nested-giant @504:  peak 11.8 GB -> 7.1 GB at 16
    frames, and submap 64 runs where fp32 OOMs (measured wall moves from
    48-64 frames to 128-192).

THE DTYPE BOUNDARY (why this is not a one-line `.to(bfloat16)`)
---------------------------------------------------------------
`da3.py` runs the depth/camera heads inside
`torch.autocast(device_type=..., enabled=False)` — deliberately, so the DPT
decoder and pose head compute in fp32.  With autocast disabled there is nothing
to reconcile a bf16 backbone output against fp32 head weights, and the forward
dies with

    RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float

inside `cam_dec`.  So the cast is paired with a forward hook that converts the
backbone's output tree back to fp32 at exactly that boundary: backbone in bf16,
heads unchanged in fp32.

ACCURACY — THIS IS A TRADE, NOT A FREE WIN
------------------------------------------
bf16 keeps fp32's exponent range but only ~3 significant decimal digits instead
of ~7.  Measured against the bf16 run-to-run noise floor, casting the backbones
moves poses by ~300x that floor.  Whether the pose graph absorbs it is an
empirical question about ATE, not something to assume — benchmark it before
adopting (scripts/sweep_merging.py --backbone_dtype).

ONE-WAY
-------
Casting down loses the fp32 mantissa bits; there is no way back without
reloading the checkpoint.  `cast_backbones` therefore refuses to widen.
"""

from __future__ import annotations

import torch
from torch import nn

DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,                 # leave the checkpoint as loaded
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str | None) -> torch.dtype | None:
    """Map a config string to a torch dtype (None = leave as loaded)."""
    if name is None:
        return None
    try:
        return DTYPES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"unknown backbone precision {name!r}; expected one of "
            f"{sorted(DTYPES)}") from None


def _to_float32(value):
    """Recursively cast a tensor / tuple / list tree back to fp32."""
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return type(value)(_to_float32(item) for item in value)
    return value


def cast_backbones(model: nn.Module, dtype: torch.dtype | None) -> list[str]:
    """Cast every DA3 branch's ViT backbone to `dtype`, fixing the head boundary.

    Args:
        model: the `depth_anything_3.api.DepthAnything3` wrapper, or a bare net.
        dtype: target dtype, or None to leave the model untouched.

    Returns:
        The branch names that were cast (empty when dtype is None).
    """
    if dtype is None:
        return []
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"cast_backbones only narrows precision; {dtype} is not supported")

    from da3_slam.backend.inference.token_tap import _resolve_net

    cast: list[str] = []
    for branch in ("anyview", "metric"):
        try:
            net = _resolve_net(model, branch)
        except (ValueError, TypeError):
            continue                      # not a nested model — no metric branch
        backbone = getattr(net, "backbone", None)
        if backbone is None:
            continue
        current = next(backbone.parameters(), None)
        if current is not None and current.dtype == dtype:
            continue                      # already cast (idempotent)
        backbone.to(dtype)
        # Restore fp32 at the head boundary — the heads run with autocast
        # disabled and would otherwise see a bf16 input against fp32 weights.
        backbone.register_forward_hook(lambda m, args, out: _to_float32(out))
        cast.append(branch)
    return cast
