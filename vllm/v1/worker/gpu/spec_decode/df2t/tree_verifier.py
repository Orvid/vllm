# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch TreeVerifier for DF2T tree speculative decoding (v5 spec §3.6).

Implements TRAVERSAL (not rejection) verification, vectorized across requests:
at the current node, draw ONE token from the fully processed target distribution
at the current row; descend iff it is a child of the current node in the tree;
otherwise emit it and stop. The SAME rule emits the bonus token at a leaf after
a fully accepted path — there is no separate greedy or bonus rule and no second
draw at any depth, and the drafter's probability buffers are never read.

Conventions mirror ``reference_verifier.tree_verify_reference`` exactly:

* ``target_logits`` is ``[num_reqs, 1 + B, vocab]`` in verify-layout order: row
  0 is the anchor, row ``1 + arena`` is tree node ``arena``.
* ``token_ids``/``parents`` are ``[num_reqs, B]`` per arena index: ``parents
  [r, i]`` is node ``i``'s parent arena index (``-1`` = child of the anchor).
  Nodes at or past ``num_nodes[r]`` are padding and can never be matched.
* ``_processed`` reproduces the reference's per-row convention bit-for-bit:
  logits/temperature when ``0 < t != 1``, then the vLLM ``apply_top_k_top_p``
  mask (top-k keeps every logit at least the k-th largest with boundary ties
  kept; top-p runs after top-k and keeps the smallest descending prefix whose
  cumulative mass reaches ``top_p``, at least one token), then softmax.
* ``greedy`` draws ``argmax``, so at T=0 every emitted token equals the argmax
  at its row (invariant 4); otherwise one categorical sample is drawn per depth.
  With per-request ``seeds``, each request draws through its own sequential
  ``torch.Generator`` exactly as the reference does, so a fixed seed is
  reproducible and bit-identical to ``tree_verify_reference``.

The output contract matches the rejection samplers so their plumbing consumes
it unchanged: ``sampled [num_reqs, 1 + B]`` (int64; ``sampled[r, :num_sampled
[r]]`` is the accepted draft prefix followed by the single terminal token) and
``num_sampled [num_reqs]`` (int32; depth traversed before stopping, i.e. the
number of emitted tokens, ``num_accepted + 1``).
"""

import torch

_NEG_INF = float("-inf")


def _row_param(
    value: float | int | torch.Tensor | None,
    R: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Broadcast a scalar or per-request sampling parameter to an ``[R]`` row."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return torch.full((R,), float(value), dtype=dtype, device=device)
    if value.dim() == 0:
        return value.to(dtype=dtype, device=device).expand(R)
    if value.dim() == 1:
        if value.numel() == 1:
            return torch.full((R,), float(value[0]), dtype=dtype, device=device)
        assert value.numel() == R, f"per-request param {value.numel()} != {R}"
        return value.to(dtype=dtype, device=device)
    raise ValueError(f"cannot broadcast sampling param of shape {value.shape}")


class TreeVerifier:
    """Vectorized tree-traversal verifier (v5 §3.6), a sibling of
    :class:`RejectionSampler` with the same output contract, used only for
    ``method="df2t"``."""

    def __init__(
        self,
        max_num_reqs: int,
        B: int,
        vocab_size: int,
        device: torch.device,
    ):
        self.B = B
        self.vocab_size = vocab_size
        self.sampled = torch.zeros(max_num_reqs, B + 1, dtype=torch.long, device=device)
        self.num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def _processed(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-vocab processed distribution over ``[.., V]`` rows, matching
        ``reference_verifier._processed`` per row (see module docstring)."""
        if temperature is not None:
            mask = (temperature > 0) & (temperature != 1.0)
            if bool(mask.any()):
                logits = torch.where(
                    mask[:, None], logits / temperature[:, None], logits
                )
        if top_k is not None:
            do_k = top_k < logits.shape[-1]
            if bool(do_k.any()):
                v = logits.shape[-1]
                k = top_k.clamp(min=1, max=v)
                thr_val = torch.topk(logits, int(k.max()), dim=-1).values
                n_e = torch.tensor(_NEG_INF, dtype=logits.dtype, device=logits.device)
                thr = torch.gather(thr_val, -1, (k - 1).unsqueeze(-1)).squeeze(-1)
                thr = torch.where(do_k, thr, torch.full_like(thr, _NEG_INF))
                logits = torch.where(logits < thr[:, None], n_e, logits)
        if top_p is not None:
            do_p = (top_p > 0) & (top_p < 1.0)
            if bool(do_p.any()):
                srt, idx = torch.sort(logits, dim=-1, descending=True)
                cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1)
                n_keep = (cum < top_p[:, None]).sum(dim=-1) + 1
                arange = torch.arange(
                    logits.shape[-1], device=logits.device, dtype=torch.long
                )
                n_e = torch.tensor(_NEG_INF, dtype=logits.dtype, device=logits.device)
                kept = torch.where(arange[None, :] < n_keep[:, None], srt, n_e)
                restored = torch.empty_like(logits)
                restored.scatter_(-1, idx, kept)
                logits = torch.where(do_p[:, None], restored, logits)
        return torch.softmax(logits, dim=-1)

    def _draw(
        self,
        p: torch.Tensor,
        greedy_mask: torch.Tensor,
        gens: list[torch.Generator] | None,
        lids: torch.Tensor,
    ) -> torch.Tensor:
        """Draw one token per live row: argmax when greedy, otherwise a single
        categorical sample per depth (never two draws)."""
        x = torch.argmax(p, dim=-1)
        want_sample = ~greedy_mask
        if not bool(want_sample.any()):
            return x
        if gens is not None:
            for j in torch.nonzero(want_sample, as_tuple=False).squeeze(-1).tolist():
                r = int(lids[j].item())
                x[j] = torch.multinomial(p[j], 1, generator=gens[r])
        else:
            x[want_sample] = torch.multinomial(p[want_sample], 1).squeeze(-1)
        return x

    def __call__(
        self,
        target_logits: torch.Tensor,
        token_ids: torch.Tensor,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
        greedy: bool | torch.Tensor = True,
        temperature: float | torch.Tensor | None = None,
        top_k: int | torch.Tensor | None = None,
        top_p: float | torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-round traversal verification for a batch of requests.

        Args:
            target_logits: ``[num_reqs, 1 + B, vocab]`` target logits per row
                (row 0 = anchor, row ``1 + arena`` = node ``arena``).
            token_ids: ``[num_reqs, B]`` node token per arena index.
            parents: ``[num_reqs, B]`` parent arena index per node (``-1`` =
                anchor child; entries at/after ``num_nodes`` are never matched).
            num_nodes: ``[num_reqs]`` real node count per request.
            greedy: draw ``argmax`` (scalar, or per-request bool tensor).
            temperature: target temperature (scalar or ``[num_reqs]``; ``None``
                means no scaling, i.e. temperature 1).
            top_k: target top-k keep per request, kept by value with boundary
                ties kept (scalar or ``[num_reqs]``).
            top_p: target nucleus mass applied after top-k (scalar or
                ``[num_reqs]``).
            seeds: optional ``[num_reqs]`` generator seeds making the per-depth
                draws reproducible and identical to the reference.

        Returns:
            ``(sampled, num_sampled)``: ``sampled [R, 1 + B]`` with each row's
            accepted draft prefix followed by exactly one terminal token, and
            ``num_sampled [R]`` (int32) the number of emitted tokens.

        """
        R, nrows, _ = target_logits.shape
        B = nrows - 1
        assert token_ids.shape == (R, B), token_ids.shape
        assert parents.shape == (R, B), parents.shape
        assert num_nodes.shape == (R,), num_nodes.shape
        assert B <= self.B, f"verifier sized for {self.B} nodes, got {B}"
        device = target_logits.device

        temperature = _row_param(temperature, R, device, torch.float32)
        top_k = _row_param(top_k, R, device, torch.long)
        top_p = _row_param(top_p, R, device, torch.float32)
        if isinstance(greedy, bool):
            greedy_mask = torch.full((R,), greedy, dtype=torch.bool, device=device)
        elif greedy.dim() == 0:
            greedy_mask = greedy.to(torch.bool).expand(R).to(device)
        else:
            greedy_mask = greedy.to(torch.bool).to(device)[:R]

        gens = None
        if seeds is not None:
            seeds = seeds.to(torch.long).to(device)
            gens = [
                torch.Generator(device=device).manual_seed(int(seeds[r].item()))
                for r in range(R)
            ]

        cur = torch.full((R,), -1, dtype=torch.long, device=device)
        done = torch.zeros(R, dtype=torch.bool, device=device)
        self.num_sampled[:R].zero_()
        arena = torch.arange(B, device=device)

        for d in range(B + 1):
            live = ~done
            lids = torch.nonzero(live, as_tuple=False).squeeze(-1)
            if lids.numel() == 0:
                break
            nl = lids.numel()
            rows_l = torch.where(cur[lids] == -1, 0, cur[lids] + 1)
            sub = target_logits[lids]
            p = self._processed(
                sub[torch.arange(nl, device=device), rows_l],
                None if temperature is None else temperature[lids],
                None if top_k is None else top_k[lids],
                None if top_p is None else top_p[lids],
            )
            x_l = self._draw(p, greedy_mask[lids], gens, lids)
            valid = arena[None, :] < num_nodes[lids].unsqueeze(-1)
            m = (
                (parents[lids] == cur[lids].unsqueeze(-1))
                & (token_ids[lids] == x_l.unsqueeze(-1))
                & valid
            )
            child_ok_l = m.any(dim=-1)
            child_arena_l = m.to(torch.long).argmax(dim=-1)
            self.sampled[lids, d] = x_l
            stop_l = ~child_ok_l
            stop_lids = lids[stop_l]
            if stop_lids.numel():
                self.num_sampled[stop_lids] = d + 1
                done[stop_lids] = True
            cur[lids] = torch.where(child_ok_l, child_arena_l, cur[lids])

        return self.sampled[:R, : B + 1], self.num_sampled[:R]
