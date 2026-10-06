"""
Bipartite soft token matching (ToMe) — vendored from FastVGGT.

SOURCE / LICENCE
----------------
Adapted from https://github.com/mystorm16/FastVGGT (`merging/merge.py`), which
is distributed under Meta's **VGGT License** (research-use terms) — *not* the
MIT licence this repository ships under.  The file is kept as a recognisable
derivative for that reason: do not relabel it, and check the VGGT License terms
before any non-research distribution.  The underlying algorithm is bipartite
soft matching from ToMe (Bolya et al.) via ToMeSD.

WHAT CHANGED vs THE ORIGINAL
----------------------------
1. `tokens_per_img` was hardcoded as `w * h + 5` (VGGT has 1 camera + 4 register
   tokens per frame).  DA3 has a different layout — `patch_start_idx = 1` and
   `num_register_tokens = 0`, so one special token per frame — so the count is
   now the `n_special` parameter.  Every `+ 5` offset follows from it.
2. `w % sx == 0` is now asserted.  The original silently mis-writes the
   destination pattern when the patch grid width is not divisible by `sx`: the
   per-image scatter writes a 2-D block of width `effective_w` into a
   *contiguous* run of `effective_grid_size` indices, which only aligns with the
   row-major token layout when `effective_w == w`.  A odd/indivisible *height*
   is fine and is handled the way the original handles it (the uncovered bottom
   rows simply stay source tokens), which matters because DA3's grids are
   36x27 at process_res 504 on 4:3 input.
3. The protection ratio is a parameter rather than a hardcoded 10%.

A NOTE ON `enable_protection` (kept as-is, deliberately)
--------------------------------------------------------
`protected_indices` is a uniform stride over *all* token indices, so it overlaps
the destination set and the per-frame special tokens.  Those tokens then appear
twice in the merged sequence — once merge-averaged, once untouched — which
double-counts them as attention keys; `unmerge` scatters the protected copy
last, so it wins.  This ships in FastVGGT and evidently works, but it is an
accident of their token layout rather than a designed property.  It is
reproduced faithfully so that an accuracy result is attributable to the method;
`protect=False` is the ablation.
"""

from typing import Callable, Optional, Tuple, Union

import torch


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked argmax-similarity of every src token against every dst token.

    Chunked so the (num_src x num_dst) score matrix is never materialised in
    full, and computed in bf16 because only the argmax is used.
    """
    B, num_src, C = a.shape
    original_dtype = a.dtype

    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        node_max[:, i:end_i] = chunk_max_bf16.to(original_dtype)
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    n_special: int = 1,
    protect_ratio: float = 0.1,
) -> Tuple[Callable, Callable]:
    """
    Split tokens into source (src) and destination (dst) groups and merge r src
    tokens into their most similar dst token.

    dst tokens are one randomly chosen token per (sx, sy) cell of each frame's
    patch grid, plus **all** of frame 0 and **all** per-frame special tokens.
    Frame 0 being held out whole matters in DA3: the reference-view selector has
    already moved the chosen reference view to index 0 by the time the first
    cross-view block runs, so "frame 0" is the anchor, not an arbitrary frame.

    Args:
        metric: [B, N, C] tensor the similarity is computed on (the attention
            input, i.e. post-norm1 tokens).
        w, h: patch grid width/height in tokens, for ONE frame.
        sx, sy: dst stride in x / y.  `w` must be divisible by `sx`; an `h` not
            divisible by `sy` leaves the bottom `h % sy` rows as src.
        r: number of src tokens to absorb (capped at the src count).
        no_rand: pick the top-left token of each cell instead of a random one.
        generator: RNG for the dst choice (seed it for reproducibility).
        enable_protection: hold `protect_ratio` of tokens out of the merge.
        n_special: non-patch tokens per frame (DA3: 1 camera/cls token).
        protect_ratio: fraction of tokens protected when enabled.

    Returns:
        (merge, unmerge): merge shortens [B, N, C] -> [B, N_merged, C]; unmerge
        scatters a merged tensor back to full length.
    """
    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    assert w % sx == 0, (
        f"patch grid width {w} must be divisible by sx={sx}; the dst pattern "
        "scatter assumes the covered block spans the full row stride"
    )

    gather = torch.gather

    tokens_per_img = w * h + n_special
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, (
        f"token count {N} is not (w*h + n_special) * num_imgs "
        f"= ({w}*{h} + {n_special}) * {num_imgs}"
    )

    with torch.no_grad():
        if enable_protection:
            num_protected = int(N * protect_ratio)
            step = max(1, N // max(num_protected, 1))
            protected_indices = torch.arange(0, N, step, device=metric.device)[
                :num_protected
            ]
        else:
            protected_indices = None
            num_protected = 0

        # Global marker of length N: -1 = dst, 0 = src.
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx

        # Frame 0 is entirely dst (the reference/anchor view).
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        if num_imgs > 1:
            # Every other frame's special tokens are dst too (in DA3 that is the
            # camera token, which the pose head reads directly).
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(
                n_special, device=metric.device
            )
            idx_buffer_seq[cls_indices.flatten()] = -1

            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)  # == w, asserted above
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
                    + n_special
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )
                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )
                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )
                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )
                for i in range(total_other_imgs):
                    grid_start = (i + 1) * tokens_per_img + n_special
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        a_idx = rand_idx[:, num_dst_orig:, :]   # src
        b_idx = rand_idx[:, :num_dst_orig, :]   # dst

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            if enable_protection:
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            return src, dst

        # Cosine similarity: normalise, then dot product.
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, _protected = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        chunk_size = min(5000, max(num_src_actual, 1))

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            r_actual = min(r, valid_edges.shape[0])
            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r

        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual

    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)
            protected = None

        n, _t1, c = src.shape
        unm_len = unm_idx.shape[1]
        src_len = src_idx.shape[1]

        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        merged_extra_1 = None
        merged_extra_2 = None

        for extra, slot in ((extra_tensors, 1), (extra_tensors_2, 2)):
            if extra is None:
                continue
            e_dim = extra.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra)
            else:
                src_e, dst_e = split(extra)
                protected_e = None
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, e_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, e_dim))
            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, e_dim), src_e_r, reduce=mode
            )
            parts = [unm_e, dst_e] + ([protected_e] if enable_protection else [])
            if slot == 1:
                merged_extra_1 = torch.cat(parts, dim=1)
            else:
                merged_extra_2 = torch.cat(parts, dim=1)

        parts = [unm, dst] + ([protected] if enable_protection else [])
        main_result = torch.cat(parts, dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        return main_result

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )
        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )
        return out

    return merge, unmerge
