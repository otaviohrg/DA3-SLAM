"""
Encoder-token tap for the DA3 backbone (Branch C, Step 0).

WHY THIS EXISTS
---------------
The temporal-reuse study needs two things the DA3 API does not expose:
  (a) the per-frame ENCODER tokens — the backbone's output *before* any
      cross-frame attention has run, and
  (b) a way to INJECT precomputed tokens for a chosen frame so that frame's
      encoder work is skipped entirely.

WHERE THE FRAME-INDEPENDENT PREFIX IS
-------------------------------------
DA3's backbone is a DINOv2 ViT whose block loop
(`depth_anything_3/model/dinov2/vision_transformer.py`,
`_get_intermediate_layers_not_chunked`) runs every block with index
`i < alt_start` as *local* attention: the token tensor is reshaped to
`(b·s) n c`, so those blocks see one frame at a time and cannot mix frames.
Cross-view ("global") attention, the reference-view reordering and the
camera-token injection all start at `alt_start`.

  => blocks[0 : alt_start] are a pure per-frame ENCODER whose output depends
     only on (image, resolution) — not on which other frames share the batch.

That is the cacheable unit.  For the default `nested-giant`, the anyview
branch is a 40-block vitg with `alt_start = 13` (blocks 0-12 are the prefix).
The metric branch (`da3metric-large`) is configured with `alt_start = -1`,
i.e. it has *no* cross-view attention at all and is frame-independent end to
end — a much larger reuse opportunity, but it feeds the DPT head from four
intermediate layers, so caching it needs four tap points instead of one.  This
module deliberately taps the anyview branch only (see `branch=`); the metric
branch is left for a follow-up.

WHY HOOKS AND NOT A FORK PATCH
------------------------------
The Depth-Anything-3 checkout is gitignored and re-cloned by `setup.sh`, so
edits there are unversioned and vanish on rebuild.  Everything here is
attached at runtime to an already-loaded model and can be detached again.

HOW THE SKIP WORKS
------------------
`process_attention` re-expands `(b s) n c -> b s n c` using the frame count it
captured *before* calling the block, so a block may not return fewer rows than
it was given.  Each prefix block therefore gets a pair of hooks:

  pre-hook  : drop the rows of frames whose tokens are cached, so attention and
              the MLP only run on the frames that actually need encoding.
  post-hook : scatter the computed rows back into a full-size tensor whose
              remaining rows are passed through unchanged (they are never read
              — the tap block overwrites them with the cached tokens).

Capture-only mode registers no slicing work at all: the pre-hooks return None
and only the tap block's post-hook does anything.

CACHE VALIDITY (one non-obvious condition)
------------------------------------------
The prefix is frame-independent, but DA3's *preprocessing* is not:
`InputProcessor._unify_batch_shapes` centre-crops every image in a batch to the
batch's smallest (H, W) when the batch mixes sizes.  A frame's pixels — and so
its tokens — therefore depend on the batch whenever sizes are mixed.  Cached
tokens are only reusable across batches of the same processed resolution; for a
single sequence (every frame the same size) that always holds, and
`scripts/test_token_hook.py` check 2 verifies it empirically.

THREADING
---------
The pipeline shares one `DepthEstimator` between the inference thread and the
loop-closure worker, and these hooks live on the module, so every thread's
forward passes through them.  Arming is therefore stored in a `threading.local`
and hooks are inert for any thread that has not armed them — an untapped
loop-closure re-inference running concurrently is unaffected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TapInfo:
    """Static description of the tapped backbone (for logs / reports)."""

    branch: str
    n_blocks: int          # total transformer blocks in the tapped ViT
    alt_start: int         # first block index that may use cross-view attention
    prefix_len: int        # blocks in the frame-independent encoder prefix
    embed_dim: int

    @property
    def prefix_fraction(self) -> float:
        """Share of the tapped backbone's blocks that are cacheable per frame."""
        return self.prefix_len / self.n_blocks

    def describe(self) -> str:
        return (f"{self.branch}: {self.prefix_len}/{self.n_blocks} blocks "
                f"({100 * self.prefix_fraction:.0f}%) are frame-independent, "
                f"dim {self.embed_dim}")


class _ArmState(threading.local):
    """Per-thread arming state (see THREADING in the module docstring)."""

    def __init__(self) -> None:
        self.armed = False
        self.capture = False
        self.inject: dict[int, torch.Tensor] | None = None
        self.captured: list[torch.Tensor] | None = None
        self.n_frames = 0
        self.keep_rows: torch.Tensor | None = None
        self.saved_input: torch.Tensor | None = None


class EncoderTokenTap:
    """
    Captures and/or injects the per-frame encoder tokens of a DA3 backbone.

    Usage (single inference call):

        tap = EncoderTokenTap(estimator.model)   # the DepthAnything3 API object
        tap.attach()
        with tap.armed(capture=True):
            prediction = estimator.infer(images)
        tokens = tap.captured                    # list[(n_tokens, dim)], one per frame

        with tap.armed(inject={0: tokens[0]}):   # frame 0 is not re-encoded
            prediction = estimator.infer(images)
    """

    def __init__(self, model: nn.Module, *, branch: str = "anyview"):
        self._vit = _resolve_vit(model, branch)
        self._backbone = _resolve_backbone(model, branch)

        alt_start = int(getattr(self._vit, "alt_start", -1))
        if alt_start <= 0:
            raise ValueError(
                f"backbone '{branch}' has alt_start={alt_start}: it has no "
                "cross-view attention, so its whole stack is frame-independent "
                "and a single tap point cannot describe it (the DPT head reads "
                "four intermediate layers).  Tap the anyview branch instead."
            )

        self.info = TapInfo(
            branch=branch,
            n_blocks=len(self._vit.blocks),
            alt_start=alt_start,
            prefix_len=alt_start,
            embed_dim=int(self._vit.embed_dim),
        )
        # Last block of the frame-independent prefix: its output is what we
        # capture, and what an injected cache replaces.
        self._tap_index = alt_start - 1
        self._state = _ArmState()
        self._handles: list = []

    # ── attach / detach ───────────────────────────────────────────────────────

    def attach(self) -> "EncoderTokenTap":
        """Register the hooks (idempotent).  Inert until a thread arms them."""
        if self._handles:
            return self

        self._handles.append(
            self._backbone.register_forward_pre_hook(self._on_backbone_start))
        for i in range(self.info.prefix_len):
            block = self._vit.blocks[i]
            self._handles.append(
                block.register_forward_pre_hook(self._make_pre_hook(),
                                                with_kwargs=True))
            self._handles.append(
                block.register_forward_hook(self._make_post_hook(i),
                                            with_kwargs=True))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "EncoderTokenTap":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    # ── arming ────────────────────────────────────────────────────────────────

    def armed(self, *, capture: bool = False,
              inject: dict[int, torch.Tensor] | None = None) -> "_ArmedScope":
        """
        Context manager arming the tap for the calling thread only.

        Args:
            capture: store the prefix output of every frame in `captured`.
            inject:  {frame index within the batch: (n_tokens, dim) tensor} —
                     those frames skip the encoder prefix and receive the given
                     tokens instead.
        """
        return _ArmedScope(self, capture, inject)

    @property
    def captured(self) -> list[torch.Tensor] | None:
        """Per-frame prefix tokens from the last armed call in this thread."""
        return self._state.captured

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _on_backbone_start(self, module, args):
        """Reset per-call state and work out which rows still need encoding."""
        state = self._state
        if not state.armed:
            return None

        x = args[0]
        if x.dim() != 5:
            raise RuntimeError(
                f"expected backbone input (B, S, 3, H, W), got {tuple(x.shape)}")
        n_batches, n_frames = int(x.shape[0]), int(x.shape[1])
        if n_batches != 1:
            raise NotImplementedError(
                "the token tap assumes a single submap per forward (B=1); "
                f"got B={n_batches}")

        state.n_frames = n_frames
        state.captured = [] if state.capture else None
        state.saved_input = None

        keep = rows_to_encode(n_frames, state.inject)
        state.keep_rows = (
            None if len(keep) == n_frames
            else torch.tensor(keep, dtype=torch.long, device=x.device))
        return None

    def _make_pre_hook(self):
        def pre_hook(module, args, kwargs):
            state = self._state
            if not state.armed or state.keep_rows is None:
                return None

            x = args[0]
            state.saved_input = x
            keep = state.keep_rows
            new_args = (x.index_select(0, keep),) + tuple(args[1:])
            new_kwargs = dict(kwargs)
            pos = new_kwargs.get("pos")
            if pos is not None:
                new_kwargs["pos"] = pos.index_select(0, keep)
            return new_args, new_kwargs
        return pre_hook

    def _make_post_hook(self, block_index: int):
        is_tap = block_index == self._tap_index

        def post_hook(module, args, kwargs, output):
            state = self._state
            if not state.armed:
                return None

            out = output
            if state.keep_rows is not None:
                # Rebuild the full row set: computed rows from `out`, the rest
                # passed through unchanged (never read — see the docstring).
                full = state.saved_input.clone()
                full.index_copy_(0, state.keep_rows, out.to(full.dtype))
                out = full
                state.saved_input = None

            if is_tap:
                if state.inject:
                    out = out.clone() if out is output else out
                    for frame, tokens in state.inject.items():
                        out[frame] = tokens.to(device=out.device, dtype=out.dtype)
                if state.capture:
                    # Clone, don't view: right after this block the loop does
                    # `x[:, :, 0] = cam_token` (at i == alt_start), an in-place
                    # write into this very storage — a view would silently show
                    # the camera token in place of the encoder's cls token.
                    state.captured = [out[i].detach().clone()
                                      for i in range(out.shape[0])]

            return out if out is not output else None
        return post_hook


class _ArmedScope:
    """Arms an `EncoderTokenTap` for the calling thread inside a `with` block."""

    def __init__(self, tap: EncoderTokenTap, capture: bool,
                 inject: dict[int, torch.Tensor] | None):
        self._tap = tap
        self._capture = capture
        self._inject = inject

    def __enter__(self) -> EncoderTokenTap:
        if not self._tap._handles:
            raise RuntimeError("EncoderTokenTap.attach() must be called first")
        state = self._tap._state
        if state.armed:
            raise RuntimeError("EncoderTokenTap is already armed in this thread")
        state.armed = True
        state.capture = self._capture
        state.inject = self._inject
        state.captured = None
        return self._tap

    def __exit__(self, *exc) -> None:
        state = self._tap._state
        state.armed = False
        state.capture = False
        state.inject = None
        state.keep_rows = None
        state.saved_input = None


# ── row bookkeeping ───────────────────────────────────────────────────────────

def rows_to_encode(n_frames: int,
                   inject: dict[int, torch.Tensor] | None) -> list[int]:
    """
    Frame rows the encoder prefix still has to compute.

    Pure bookkeeping, kept out of the hooks so it can be tested without a GPU.
    """
    inject = inject or {}
    for frame in inject:
        if not 0 <= frame < n_frames:
            raise IndexError(
                f"injection frame {frame} out of range for a {n_frames}-frame batch")

    keep = [f for f in range(n_frames) if f not in inject]
    # An all-cached batch would leave the block with zero rows, which not every
    # attention kernel tolerates — keep one row and throw its result away (the
    # tap block overwrites every row from the cache anyway).
    return keep or [0]


# ── model introspection ───────────────────────────────────────────────────────

def _resolve_net(model: nn.Module, branch: str) -> nn.Module:
    """
    Find the DepthAnything3Net inside whatever was passed.

    Accepts the `depth_anything_3.api.DepthAnything3` wrapper (`.model`), a
    `NestedDepthAnything3Net` (`.da3` / `.da3_metric`) or a bare
    `DepthAnything3Net`.
    """
    net = getattr(model, "model", model)      # api wrapper → net
    if hasattr(net, "da3"):                   # nested → chosen branch
        if branch == "anyview":
            net = net.da3
        elif branch == "metric":
            net = net.da3_metric
        else:
            raise ValueError(f"unknown branch '{branch}' (use 'anyview' or 'metric')")
    elif branch != "anyview":
        raise ValueError(f"model has no '{branch}' branch — it is not a nested model")
    if not hasattr(net, "backbone"):
        raise TypeError(f"could not find a DA3 backbone on {type(net).__name__}")
    return net


def _resolve_backbone(model: nn.Module, branch: str) -> nn.Module:
    """The `DinoV2` wrapper module — it sees the (B, S, 3, H, W) input."""
    return _resolve_net(model, branch).backbone


def _resolve_vit(model: nn.Module, branch: str) -> nn.Module:
    """The `DinoVisionTransformer` — it owns `blocks` and `alt_start`."""
    backbone = _resolve_backbone(model, branch)
    return getattr(backbone, "pretrained", backbone)
