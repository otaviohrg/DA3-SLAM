"""
Cross-view token merging for the DA3 backbone (Branch C — the FastVGGT port).

WHAT THIS DOES
--------------
FastVGGT (ICLR 2026) observes that a visual-geometry transformer's *global*
(cross-view) attention maps are highly redundant, and accelerates inference —
training-free, checkpoint unchanged — by merging most tokens together before the
global attention, running attention on the short sequence, and scattering the
result back to full length.  This module ports that to DA3.

WHERE IT HOOKS
--------------
DA3's backbone is a DINOv2 ViT whose block loop
(`depth_anything_3/model/dinov2/vision_transformer.py`,
`_get_intermediate_layers_not_chunked`) runs block `i` as *global* attention
when `i >= alt_start and i % 2 == 1`, and as *local* (per-frame) attention
otherwise.  Each block is therefore permanently one or the other, so the port is
just: find the global blocks and wrap their attention.

  nested-giant anyview branch: alt_start=13, depth=40 -> global blocks are the
  odd indices 13..39, i.e. 14 of 40.  DA3-LARGE: alt_start=8, depth=24 -> 8 of
  24.  Both are read off the loaded model; nothing here is hardcoded.

The metric branch of a nested model has `alt_start = -1` — no cross-view
attention at all — so there is nothing to merge there and it is left alone.

WHY A RUNTIME WRAPPER AND NOT A FORK PATCH
------------------------------------------
The Depth-Anything-3 checkout is gitignored and re-cloned by `setup.sh`, so
edits there are unversioned and vanish on rebuild.  Same reasoning, same
approach, and the same model-introspection helpers as `token_tap.py` — which
hooks the *other* half of the same block loop (the frame-independent prefix,
blocks `[0, alt_start)`).  The two never touch the same block and compose.

MECHANISM (M1: module swap)
---------------------------
Each global block's `.attn` is replaced by a `MergingAttention` that holds the
original module by reference and reimplements DA3's ~15-line attention forward
with the merge spliced in at FastVGGT's exact position.  The class it mirrors is
`depth_anything_3/model/dinov2/layers/attention.py` — note that DA3 has a second,
near-identical `Attention` in `model/utils/attention.py` which the ViT does NOT
use; they differ in whether `head_dim` is an attribute and whether a `fused_attn`
fallback exists, so copying the wrong one silently breaks.  The pipeline is:

    norm1(x) -> qkv -> qk_norm -> rope -> MERGE(q,k,v) -> SDPA -> proj -> UNMERGE

Merging after the projection and the RoPE (rather than merging the input) is
what makes this a faithful port: DA3 sets `qk_norm=True` on exactly the blocks
that are global (`qknorm_start == alt_start` in every config), and LayerNorm
does not commute with averaging.

ON / OFF
--------
Three levels, in increasing strength:

  * `configure(enable=False)` — the wrapper stays installed but delegates
    straight to the original module, so the forward is bit-identical to the
    unmerged baseline.  This is the cheap switch for a sweep that keeps one
    model loaded across configs.
  * `detach()` — restores the original `.attn` modules; nothing of this module
    is left in the graph.
  * per-thread `disabled()` — force off for the calling thread only.

There is also an automatic gate: `min_frames`.  Batches smaller than that are
never merged, which is what keeps the loop-closure worker's ~4-frame
re-inference (2 matched frames + `context_frames` neighbours per side) on the
exact path — merging there would save nothing and would degrade the inference
that decides whether a closure is accepted.  Because the gate is a property of
the *batch*, not of the calling thread, it needs no coordination between the
inference thread and the loop-closure worker.

THREADING
---------
The pipeline shares one `DepthEstimator` between `_inference` and
`_loop_closure_worker`, so both threads' forwards pass through these wrappers.
The per-call geometry (patch grid, frame count) is therefore held in a
`threading.local`, exactly as `token_tap.py` holds its arming state.

ONE CAVEAT
----------
Wrapping re-parents the attention submodules, so while attached the model's
`state_dict()` keys for global blocks gain an `.inner` level
(`blocks.13.attn.qkv.weight` -> `blocks.13.attn.inner.qkv.weight`).  Nothing in
DA3-SLAM saves backbone weights, and `detach()` restores the original keys, but
do not attach around a checkpoint save.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from da3_slam.backend.inference._tome import token_merge_bipartite2d
from da3_slam.backend.inference.token_tap import _resolve_backbone, _resolve_vit


@dataclass
class MergeSettings:
    """Runtime knobs.  Mirrors `da3_slam.config.TokenMergingConfig`, which is
    the YAML-facing (torch-free) version of the same thing."""

    enable: bool = False

    # First global block that merges, expressed as a POSITION AMONG THE GLOBAL
    # BLOCKS (0 = all of them).  Deliberately not a raw block index: 14 global
    # blocks in giant vs 8 in large means a raw index does not transfer across
    # model sizes, but "the last 60% of the cross-view stack" does.
    start: int = 0

    # Fraction of tokens to absorb.  FastVGGT's 0.9 in practice absorbs every
    # source token (the source set is only ~75% of the sequence to begin with).
    merge_ratio: float = 0.9

    # Destination stride over the patch grid.  One token per (sx, sy) cell is
    # held out as a merge destination.
    sx: int = 2
    sy: int = 2

    # Hold a uniform stride of tokens out of the merge entirely.
    protect: bool = True
    protect_ratio: float = 0.1

    # Seed for the destination choice.  Fixed so a config is reproducible; it is
    # re-seeded per attention call, so every global block uses the same pattern.
    seed: int = 33

    # Batches with fewer frames than this are never merged (see ON / OFF).
    min_frames: int = 8


@dataclass
class MergeStats:
    """What actually happened, for the report tables."""

    calls_merged: int = 0
    calls_passthrough: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def token_ratio(self) -> float:
        """Realised merged-length / full-length.  Attention cost scales with
        the square of this."""
        return self.tokens_out / self.tokens_in if self.tokens_in else 1.0

    def describe(self) -> str:
        if not self.calls_merged:
            return f"no merged calls ({self.calls_passthrough} passthrough)"
        return (
            f"{self.calls_merged} merged / {self.calls_passthrough} passthrough "
            f"attention calls, tokens {self.token_ratio:.3f}x "
            f"(attention ~{self.token_ratio ** 2:.3f}x)"
        )


@dataclass
class _Plan:
    """Per-call geometry needed to build the merge."""

    w: int
    h: int
    n_special: int
    sx: int
    sy: int


class _CallState(threading.local):
    """Per-thread state for the forward pass currently in flight."""

    def __init__(self) -> None:
        self.w: int | None = None
        self.h: int | None = None
        self.n_frames: int = 0
        self.force_off: bool = False


class MergingAttention(nn.Module):
    """DA3 `Attention` with FastVGGT token merging spliced into the global path.

    Holds the original module (`.inner`) by reference — no weights are copied,
    so installing and removing this is free.
    """

    def __init__(self, inner: nn.Module, merger: "TokenMerger", slot: int):
        super().__init__()
        self.inner = inner
        self.slot = slot                 # position among the global blocks
        object.__setattr__(self, "_merger", merger)   # not a submodule

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        merger: TokenMerger = self._merger
        plan = merger.plan_for(self.slot, x)
        if plan is None:
            return self.inner(x, pos=pos, attn_mask=attn_mask)

        if attn_mask is not None:
            # Merging changes the sequence length, so a mask built for the full
            # sequence would be silently wrong.  DA3-SLAM never passes one.
            raise RuntimeError(
                "token merging does not support attn_mask (the merged sequence "
                "no longer matches the mask's shape)"
            )

        inner = self.inner
        B, N, C = x.shape
        # NOTE: mirrors depth_anything_3/model/dinov2/layers/attention.py — the
        # class the ViT blocks actually use.  (There is a second, similar
        # `Attention` in model/utils/attention.py used elsewhere in DA3; it
        # exposes `head_dim` as an attribute and has no `fused_attn`.  This one
        # keeps head_dim local, so derive it the same way it does.)
        n_heads = inner.num_heads
        head_dim = C // n_heads

        qkv = (
            inner.qkv(x)
            .reshape(B, N, 3, n_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = inner.q_norm(q), inner.k_norm(k)
        if inner.rope is not None and pos is not None:
            q = inner.rope(q, pos)
            k = inner.rope(k, pos)

        settings = merger.settings
        generator = torch.Generator(device=x.device)
        generator.manual_seed(settings.seed)

        merge, unmerge = token_merge_bipartite2d(
            x,
            w=plan.w,
            h=plan.h,
            sx=plan.sx,
            sy=plan.sy,
            r=int(N * settings.merge_ratio),
            no_rand=False,
            generator=generator,
            enable_protection=settings.protect,
            n_special=plan.n_special,
            protect_ratio=settings.protect_ratio,
        )

        # (B, heads, N, head_dim) -> (B, N, heads*head_dim) for the merge, and
        # back again afterwards.
        def flatten_heads(t: Tensor) -> Tensor:
            return t.permute(0, 2, 1, 3).reshape(B, N, n_heads * head_dim)

        q_m, k_m, v_m = merge(
            flatten_heads(q),
            mode="mean",
            extra_tensors=flatten_heads(k),
            extra_tensors_2=flatten_heads(v),
        )
        del q, k, v, qkv

        n_merged = q_m.shape[1]

        def unflatten_heads(t: Tensor) -> Tensor:
            return t.reshape(B, n_merged, n_heads, head_dim).permute(0, 2, 1, 3)

        q_m, k_m, v_m = (
            unflatten_heads(q_m),
            unflatten_heads(k_m),
            unflatten_heads(v_m),
        )

        if getattr(inner, "fused_attn", True):
            out = F.scaled_dot_product_attention(
                q_m,
                k_m,
                v_m,
                dropout_p=inner.attn_drop.p if inner.training else 0.0,
            )
        else:
            attn = (q_m * inner.scale) @ k_m.transpose(-2, -1)
            attn = inner.attn_drop(attn.softmax(dim=-1))
            out = attn @ v_m
            del attn
        del q_m, k_m, v_m

        out = out.transpose(1, 2).reshape(B, n_merged, C)
        out = inner.proj(out)
        out = inner.proj_drop(out)
        out = unmerge(out)

        merger.record_merged(N, n_merged)
        return out


class TokenMerger:
    """Installs / removes / configures cross-view token merging on a DA3 model.

    Usage:

        merger = TokenMerger(estimator.model).attach()
        merger.configure(MergeSettings(enable=True, start=7))
        ...                                   # merged inference
        merger.configure(enable=False)        # bit-identical to baseline
        merger.detach()                       # nothing left installed
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        branch: str = "anyview",
        settings: MergeSettings | None = None,
    ):
        self._vit = _resolve_vit(model, branch)
        self._backbone = _resolve_backbone(model, branch)
        self.branch = branch

        alt_start = int(getattr(self._vit, "alt_start", -1))
        if alt_start < 0:
            raise ValueError(
                f"backbone '{branch}' has alt_start={alt_start}: it has no "
                "cross-view attention, so there is nothing to merge.  (The "
                "metric branch of a nested model is configured this way; merge "
                "the anyview branch instead.)"
            )

        self.global_blocks = [
            i
            for i in range(len(self._vit.blocks))
            if i >= alt_start and i % 2 == 1
        ]
        if not self.global_blocks:
            raise ValueError(
                f"backbone '{branch}' (alt_start={alt_start}, "
                f"{len(self._vit.blocks)} blocks) exposes no global blocks"
            )

        self.patch_size = int(getattr(self._vit, "patch_size", 14))
        self.settings = settings or MergeSettings()
        self.stats = MergeStats()

        self._call = _CallState()
        self._originals: dict[int, nn.Module] = {}
        self._handle = None
        self._stats_lock = threading.Lock()
        self._warned: set[str] = set()

        self._verify_upstream()

    # ── description ───────────────────────────────────────────────────────────

    def describe(self) -> str:
        return (
            f"{self.branch}: {len(self.global_blocks)}/{len(self._vit.blocks)} "
            f"blocks are cross-view (indices {self.global_blocks[0]}.."
            f"{self.global_blocks[-1]} odd), patch {self.patch_size}"
        )

    @property
    def attached(self) -> bool:
        return bool(self._originals)

    # ── attach / detach ───────────────────────────────────────────────────────

    def attach(self) -> "TokenMerger":
        """Swap in the merging attention (idempotent).  Inert while disabled."""
        if self.attached:
            return self
        self._handle = self._backbone.register_forward_pre_hook(
            self._on_backbone_start
        )
        for slot, block_idx in enumerate(self.global_blocks):
            block = self._vit.blocks[block_idx]
            self._originals[block_idx] = block.attn
            block.attn = MergingAttention(block.attn, self, slot)
        return self

    def detach(self) -> None:
        """Restore the original attention modules and remove the hook."""
        for block_idx, original in self._originals.items():
            self._vit.blocks[block_idx].attn = original
        self._originals.clear()
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "TokenMerger":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    # ── configuration ─────────────────────────────────────────────────────────

    def configure(self, settings: MergeSettings | None = None, **overrides) -> None:
        """Replace the settings wholesale and/or override individual fields.

        Cheap: no model reload, no re-attach.  A sweep can flip merging on and
        off between runs on a model that stays resident.
        """
        if settings is not None:
            self.settings = settings
        for key, value in overrides.items():
            if not hasattr(self.settings, key):
                raise AttributeError(f"unknown merge setting '{key}'")
            if value is not None:
                setattr(self.settings, key, value)
        self._validate_settings()

    def reset_stats(self) -> None:
        with self._stats_lock:
            self.stats = MergeStats()

    @contextmanager
    def disabled(self):
        """Force merging off for the calling thread only."""
        previous = self._call.force_off
        self._call.force_off = True
        try:
            yield
        finally:
            self._call.force_off = previous

    # ── the per-call decision ─────────────────────────────────────────────────

    def plan_for(self, slot: int, x: Tensor) -> _Plan | None:
        """Geometry for this call, or None to pass through unmerged."""
        state = self._call
        settings = self.settings

        if state.force_off or not settings.enable:
            return None
        if slot < settings.start:
            return None
        if state.w is None or state.n_frames < settings.min_frames:
            self._count_passthrough()
            return None

        w, h = state.w, state.h
        tokens_per_img, remainder = divmod(int(x.shape[1]), state.n_frames)
        n_special = tokens_per_img - w * h
        if remainder or n_special < 0:
            # Geometry does not match the tensor — refuse rather than corrupt.
            self._warn(
                "geometry",
                f"token count {x.shape[1]} does not factor as "
                f"{state.n_frames} frames x ({w}*{h} + special); "
                "merging is off for this call",
            )
            self._count_passthrough()
            return None

        # The destination scatter assumes the covered block spans a full row, so
        # a width not divisible by sx must fall back to sx=1.  An indivisible
        # height is fine: the uncovered bottom rows just stay source tokens.
        sx = settings.sx if w % settings.sx == 0 else 1
        if sx != settings.sx:
            self._warn(
                "sx",
                f"patch grid width {w} is not divisible by sx={settings.sx}; "
                "using sx=1 for this resolution",
            )
        if h % settings.sy:
            self._warn(
                "sy",
                f"patch grid height {h} is not divisible by sy={settings.sy}; "
                f"the bottom {h % settings.sy} patch row(s) stay source tokens",
            )
        return _Plan(w=w, h=h, n_special=n_special, sx=sx, sy=settings.sy)

    def record_merged(self, tokens_in: int, tokens_out: int) -> None:
        with self._stats_lock:
            self.stats.calls_merged += 1
            self.stats.tokens_in += tokens_in
            self.stats.tokens_out += tokens_out

    # ── internals ─────────────────────────────────────────────────────────────

    def _count_passthrough(self) -> None:
        with self._stats_lock:
            self.stats.calls_passthrough += 1

    def _warn(self, key: str, message: str) -> None:
        """Warn once per distinct condition (these fire per block per call)."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[TokenMerger] warning: {message}")

    def _on_backbone_start(self, module, args):
        """Capture this call's patch grid and frame count.

        The DinoV2 wrapper sees the raw (B, S, 3, H, W) batch, which is the only
        place the patch grid is knowable — by the time a global block runs, the
        tensor is (B, S*n, C) and the frame/patch split is ambiguous.  This is
        DA3's equivalent of FastVGGT's per-scene update_patch_dimensions(), and
        doing it per call is what keeps the module correct across the resolution
        sweep instead of baking in one grid.
        """
        state = self._call
        state.w = state.h = None
        state.n_frames = 0

        x = args[0]
        if not torch.is_tensor(x) or x.dim() != 5:
            return None
        n_batches, n_frames, _, height, width = x.shape
        if n_batches != 1:
            self._warn(
                "batch",
                f"expected a single submap per forward (B=1), got B={n_batches}; "
                "merging is off",
            )
            return None
        state.n_frames = int(n_frames)
        state.w = int(width) // self.patch_size
        state.h = int(height) // self.patch_size
        return None

    def _validate_settings(self) -> None:
        s = self.settings
        n_global = len(self.global_blocks)
        if not 0 <= s.start < n_global:
            raise ValueError(
                f"merge start {s.start} is out of range: this backbone has "
                f"{n_global} global blocks, so start must be in [0, {n_global})"
            )
        if not 0.0 < s.merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be in (0, 1], got {s.merge_ratio}")
        if s.sx < 1 or s.sy < 1:
            raise ValueError(f"sx/sy must be >= 1, got sx={s.sx} sy={s.sy}")
        if not 0.0 <= s.protect_ratio < 1.0:
            raise ValueError(
                f"protect_ratio must be in [0, 1), got {s.protect_ratio}"
            )

    def _verify_upstream(self) -> None:
        """Fail loudly if upstream DA3 no longer looks like what we reimplement.

        `setup.sh` re-clones Depth-Anything-3, so the ~15 lines of attention
        forward that MergingAttention duplicates can change under us.  Checking
        the attributes is cheap insurance against a silent wrong answer.
        """
        required = (
            "qkv", "q_norm", "k_norm", "proj", "proj_drop",
            "attn_drop", "rope", "num_heads", "scale",
        )
        probe = self._vit.blocks[self.global_blocks[0]].attn
        missing = [name for name in required if not hasattr(probe, name)]
        if missing:
            raise RuntimeError(
                "upstream DA3 attention has changed: "
                f"{type(probe).__name__} is missing {missing}.  "
                "MergingAttention.forward reimplements that module and must be "
                "re-checked against the new upstream before it can be trusted."
            )
