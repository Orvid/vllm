# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Brute-force traversal verifier reference (the oracle).

Implements the DF2T lossless verification rule: starting at the
anchor, draw one token per depth from the fully processed target distribution
at the current row; descend iff the drawn token is a child of the current node;
otherwise emit it and stop. At a leaf ``children(u)`` is empty, so the same
step emits the bonus token after a fully accepted path — there is no separate
greedy or bonus path, and drafter probabilities are never read.

Exact conventions, documented so the GPU sampler can mirror them:

* ``target_logits`` is ``[1 + B, vocab]`` in verify-layout order: row 0 is the
  anchor (the already-generated context), row ``1 + arena`` is tree node
  ``arena``. This matches the verify block built by ``prepare_tree_inputs``.
* ``token_ids``/``parents`` are per arena index: ``token_ids[i]`` is the node's
  token and ``parents[i]`` its parent's arena index (``-1`` = child of the
  anchor).
* The processed distribution at a row is the softmax of ``logits /
  temperature`` after the vLLM ``apply_top_k_top_p`` masks (top-k keeps all
  logits at least the k-th largest, boundary ties kept; top-p, applied after
  top-k, keeps the smallest descending-probability prefix whose cumulative
  mass reaches ``top_p``, at least one token).
* ``greedy`` draws ``argmax``, so every emitted token (accepted, mismatch, and
  the leaf bonus on a fully accepted path) equals ``argmax(p)`` exactly at
  T=0 (the argmax-everywhere invariant). Otherwise one categorical sample is drawn per depth
  from a single ``torch.Generator`` seeded from ``seed``, drawn sequentially so
  depths are effectively independent and a fixed seed is reproducible
  (``seed=None`` is non-reproducible).
* One draw per depth decides both acceptance and the emitted token: never two
  draws at one depth.
* ``num_accepted`` is the count of accepted draft tokens (the depth of the
  final node), tracked incrementally. This deliberately deviates from the
  brief sketch's ``parents[u] + 1``, which is only correct for depth-aligned
  arenas; the count must be exact for arbitrary trees.
"""

import torch


def _processed(
    logits: torch.Tensor,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Full-vocab processed distribution at one row (see module docstring)."""
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
    return torch.softmax(logits, dim=-1)


def _child_index(
    token_ids: torch.Tensor, parents: torch.Tensor, u: int, x: int
) -> int | None:
    """Arena index of the child of ``u`` whose token equals ``x``."""
    for i in range(parents.shape[0]):
        if int(parents[i].item()) == u and int(token_ids[i].item()) == x:
            return i
    return None


def _draw(p: torch.Tensor, greedy: bool, generator: torch.Generator | None) -> int:
    if greedy:
        return int(torch.argmax(p).item())
    return int(torch.multinomial(p, 1, generator=generator).item())


def tree_verify_reference(
    target_logits: torch.Tensor,
    token_ids: torch.Tensor,
    parents: torch.Tensor,
    greedy: bool = True,
    temperature: float | None = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> tuple[list[int], int]:
    """Run the multi-round traversal verification for one request.

    Args:
        target_logits: ``[1 + B, vocab]`` target logits per row (see module
            docstring for the anchor/node row convention).
        token_ids: ``[B]`` node token per arena index.
        parents: ``[B]`` parent arena index per node (``-1`` = anchor child).
        greedy: draw ``argmax`` instead of sampling the processed distribution.
        temperature: target temperature scaling.
        top_k: target top-k keep count (kept by value, boundary ties kept).
        top_p: target nucleus mass, applied after top-k.
        seed: optional generator seed making all draws reproducible.

    Returns:
        ``(out_tokens, num_accepted)``: the accepted draft tokens followed by
        exactly one terminal token (the mismatch token, or the leaf bonus after
        a fully accepted path), and the number of accepted draft tokens.

    """
    device = target_logits.device
    if target_logits.dim() == 1:
        target_logits = target_logits.unsqueeze(0)
    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )
    out: list[int] = []
    u = -1  # arena index of the current node; -1 = anchor
    num_accepted = 0

    while True:
        row = 0 if u == -1 else 1 + u
        p = _processed(target_logits[row], temperature, top_k, top_p)
        x = _draw(p, greedy, generator)
        child = _child_index(token_ids, parents, u, x)
        if child is not None:
            out.append(x)
            u = child
            num_accepted += 1
            continue
        out.append(x)
        return out, num_accepted
