# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Brute-force SpecInfer multi-round rejection reference (the oracle).

The GPU ``TreeRejectionSampler`` (and any future kernel) must reproduce this
function bit-for-bit; the divergence-free KL gate is judged against it. It
walks the accepted tree path one round at a time: at each accepted node one
target token is drawn and compared against the expanded children; a miss is
rejected and the round is terminated with a token resampled from the
truncation-corrected residual ``max(p - q, 0) / Z`` over the full vocabulary.
The token ``q`` here is the drafter's proposal over the K successor candidates
of the current node (a full-vocab vector that is zero outside the candidates),
so the residual is the standard SpecInfer multi-round correction.

Exact conventions, documented so Task 10 can mirror them:

* ``target_logits`` is ``[1 + B, vocab]`` in verify-layout order: row 0 is the
  implicit anchor (the already-generated context), row ``1 + arena[i]`` is
  tree node ``i``. This matches ``prepare_tree_inputs``/the verify block.
* ``q_cand``/``q_logp`` are ``[L, K, K]`` in the tree-builder's best-first
  order: row ``(depth, predecessor_rank, :)`` holds the successor token ids
  (``q_cand``) and logits (``q_logp``). ``predecessor_rank`` of the anchor is
  0; the walk tracks each accepted node's rank (the slot its token was found
  in) and uses it as the next round's predecessor rank.
* The target distribution at a row is the softmax of ``logits / temperature``
  after the vLLM ``apply_top_k_top_p`` masks (top-k keeps all logits at least
  the k-th largest; top-p keeps the smallest descending-probability prefix
  whose cumulative mass reaches ``top_p``, at least one token, applied after
  top-k). ``greedy`` draws ``argmax``; sampling draws one categorical sample.
* All stochastic draws share a single ``torch.Generator`` seeded from
  ``seed`` (if given), so a fixed seed is fully reproducible and successive
  draws consume one deterministic stream. ``seed=None`` is non-reproducible.
* ``num_accepted`` is the number of accepted draft tokens; the returned list
  is those tokens plus exactly one terminal token (the residual-recovered
  token on rejection, or a bonus target draw once the depth budget is met).
"""

import torch

from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft


def _mask_target_logits(
    logits: torch.Tensor,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply temperature scaling and top-k/top-p masks to one logits row.

    Mirrors the vLLM production sampler: divide by ``temperature`` (skip when
    None/<=0), then keep the top-``top_k`` logits by value (ties at the
    boundary kept, ``logit < kth-large -> -inf``), then keep the smallest
    descending-softmax-prefix reaching ``top_p`` (at least one token).
    """
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and int(top_k) < logits.shape[-1]:
        k = max(int(top_k), 1)
        thr = torch.topk(logits, k).values[-1]
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
        logits = torch.where(logits < thr, neg_inf, logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1)
        n_keep = int((cum < top_p).sum().item()) + 1
        kept = torch.full_like(srt, float("-inf"))
        kept[:n_keep] = srt[:n_keep]
        logits = kept.scatter(-1, idx, kept)
    return logits


def _target_dist(
    target_logits: torch.Tensor,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Full-vocab target distribution at one row."""
    masked = _mask_target_logits(target_logits, temperature, top_k, top_p)
    return torch.softmax(masked, dim=-1)


def _draw(p: torch.Tensor, greedy: bool, generator: torch.Generator | None) -> int:
    if greedy:
        return int(torch.argmax(p).item())
    return int(torch.multinomial(p, 1, generator=generator).item())


def _match_child(tree: TreeDraft, node: int, x: int) -> int | None:
    """Arena index of the expanded child of ``node`` whose token is ``x``."""
    parents = tree.parents
    tokens = tree.token_ids
    for i in range(parents.shape[0]):
        if int(parents[i].item()) == node and int(tokens[i].item()) == x:
            return i
    return None


def _residual(p: torch.Tensor, q_full: torch.Tensor) -> torch.Tensor:
    """Truncation-corrected residual ``max(p - q, 0) / Z``, full vocabulary.

    Falls back to ``p`` (unnormalized-zero residual) when ``Z == 0``, which is
    only reachable when ``p`` is fully dominated on the candidate support and
    holds no mass outside it.
    """
    r = torch.clamp(p - q_full, min=0.0)
    z = r.sum()
    if z > 0:
        return r / z
    return p.clone()


def specinfer_reference(
    target_logits: torch.Tensor,
    tree: TreeDraft,
    q_cand: torch.Tensor,
    q_logp: torch.Tensor,
    greedy: bool = True,
    temperature: float | None = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> tuple[list[int], int]:
    """Run the multi-round SpecInfer walk for one request.

    Args:
        target_logits: ``[1 + B, vocab]`` target logits per row (see module
            docstring for the anchor/node row convention).
        tree: single-request ``TreeDraft`` (1-D fields, arena order).
        q_cand: ``[L, K, K]`` successor token ids in best-first order.
        q_logp: ``[L, K, K]`` successor logits aligned with ``q_cand``.
        greedy: draw ``argmax`` instead of sampling the target.
        temperature: target temperature scaling.
        top_k: target top-k keep count (kept by value, ties kept).
        top_p: target nucleus mass, applied after top-k.
        seed: optional generator seed making all draws reproducible.

    Returns:
        ``(out_tokens, num_accepted)``: the accepted draft tokens followed by
        exactly one terminal token (residual-recovered on rejection, bonus
        target draw at depth-budget exhaustion), and the number of accepted
        draft tokens.

    """
    if target_logits.dim() == 1:
        target_logits = target_logits.unsqueeze(0)
    device = target_logits.device
    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )
    max_depth = int(tree.depths.max().item()) if tree.depths.shape[0] else 0
    out: list[int] = []
    node = -1  # anchor
    rank = 0  # predecessor rank of the current node (0 for the anchor)
    depth = 0

    while depth < max_depth:
        row = 0 if node == -1 else int(tree.arena[node].item()) + 1
        p = _target_dist(target_logits[row], temperature, top_k, top_p)
        x = _draw(p, greedy, generator)
        cands = q_cand[depth, rank]
        child = _match_child(tree, node, x)
        if child is not None:
            out.append(int(x))
            node = child
            depth += 1
            hits = (cands == x).nonzero()
            if hits.numel():
                rank = int(hits[0, 0].item())
            continue
        q = torch.softmax(q_logp[depth, rank].float(), dim=-1)
        q_full = torch.zeros_like(p)
        q_full.index_copy_(0, cands.to(torch.long), q.to(p.dtype))
        recovered = _draw(_residual(p, q_full), greedy, generator)
        out.append(int(recovered))
        return out, depth

    row = 0 if node == -1 else int(tree.arena[node].item()) + 1
    p = _target_dist(target_logits[row], temperature, top_k, top_p)
    out.append(int(_draw(p, greedy, generator)))
    return out, depth
