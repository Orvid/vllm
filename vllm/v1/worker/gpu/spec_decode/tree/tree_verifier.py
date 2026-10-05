# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch TreeVerifier for DF2T tree speculative decoding (v5 spec §3.6).

Implements TRAVERSAL (not rejection) verification, vectorized across requests:
at the current node, draw ONE token from the fully processed target distribution
at the current row; descend iff it is a child of the current node in the tree;
otherwise emit it and stop. The SAME rule emits the bonus token at a leaf after
a fully accepted path — there is no separate greedy or bonus rule and no second
draw at any depth, and the drafter's probability buffers are never read.

Conventions mirror the test-tree oracle ``tests/.../reference_verifier.py``
(``tree_verify_reference``, kept as TEST material per v5 SS9) exactly:

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
  reproducible and bit-identical to ``tree_verify_reference``. A seed of ``-1``
  selects the global RNG (no generator); the model-runner path passes
  per-request ``-1`` for unseeded requests and derives per-step seeds for
  explicitly-seeded ones, so draws are independent across (request, step).

The output contract matches the rejection samplers so their plumbing consumes
it unchanged: ``sampled [num_reqs, 1 + B]`` (int64; ``sampled[r, :num_sampled
[r]]`` is the accepted draft prefix followed by the single terminal token) and
``num_sampled [num_reqs]`` (int32; depth traversed before stopping, i.e. the
number of emitted tokens, ``num_accepted + 1``).
"""

import torch

from vllm.triton_utils import tl, triton

_NEG_INF = float("-inf")


@triton.jit
def _tree_walk_kernel(
    row_tokens_ptr,  # [R, 1 + B] int64 drawn token per verify row
    token_ids_ptr,  # [R, B] int64, row stride ``tree_stride``
    parents_ptr,  # [R, B] int64, row stride ``tree_stride``
    num_nodes_ptr,  # [R] int64
    sampled_ptr,  # [max_num_reqs, persist_B + 1] int64 (persistent)
    num_sampled_ptr,  # [max_num_reqs] int32 (persistent)
    arena_ptr,  # [max_num_reqs, persist_B + 1] int64 (persistent)
    tree_stride,
    B: tl.constexpr,
    num_rounds: tl.constexpr,
    persist_B: tl.constexpr,
    ROW_P2: tl.constexpr,  # next power of two >= persist_B + 1
):
    """One tree walk per request (grid = R) over pre-drawn row tokens.

    Replicates the torch traversal exactly (bit-identical given the same
    draws): at round ``d`` take the token ``x`` drawn at the current row (row
    0 = anchor, row ``1 + arena`` = node), descend to the FIRST arena whose
    parent is the current node and token equals ``x``; else emit ``x`` and
    stop. The bonus is emitted at a leaf in the same round (a full chain never
    finds a child on its last round, so ``num_sampled = accepted + 1``
    always). The request's whole ``accepted_arenas`` row (-1 = none) and
    ``num_sampled`` (0 if it never stops) are written here, so no separate
    reset is needed.
    """
    r = tl.program_id(0)
    nn = tl.minimum(tl.maximum(tl.load(num_nodes_ptr + r), 0), B)
    base_a = r * (B + 1)
    base_b = r * tree_stride
    base_s = r * (persist_B + 1)
    cols = tl.arange(0, ROW_P2)
    arenas = tl.full((ROW_P2,), -1, tl.int64)
    n_sampled = tl.full((), 0, tl.int32)
    cur = tl.full((), -1, tl.int64)
    stopped = 0
    for d in range(num_rounds):
        if stopped == 0:
            row = cur + 1  # -1 -> row 0 (anchor); arena a -> row 1 + a
            x = tl.load(row_tokens_ptr + base_a + row)
            found = 0
            child = tl.full((), B, tl.int64)  # B = "no arena" sentinel (>= nn)
            for i in range(0, nn):
                p = tl.load(parents_ptr + base_b + i)
                t = tl.load(token_ids_ptr + base_b + i)
                if (p == cur) & (t == x) & (found == 0):
                    child = i
                    found = 1
            tl.store(sampled_ptr + base_s + d, x)
            if found == 0:
                n_sampled = d + 1
                stopped = 1
            else:
                arenas = tl.where(cols == tl.minimum(d + 1, B), child, arenas)
                cur = child
    tl.store(arena_ptr + base_s + cols, arenas, mask=cols < persist_B + 1)
    tl.store(num_sampled_ptr + r, n_sampled)


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
        num_rounds: int | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.B = B
        self.vocab_size = vocab_size
        # Number of traversal rounds (one draw per depth): the depth loop bound.
        # Defaults to the node count ``B + 1`` (a full chain to the last node
        # plus its terminal). The runner sizes it to the real chain cap
        # ``max_tree_depth + 1`` so the fixed-shape device loop never wastes
        # rounds past any tree's depth.
        self._num_rounds = B + 1 if num_rounds is None else num_rounds
        self.sampled = torch.zeros(max_num_reqs, B + 1, dtype=torch.long, device=device)
        self.num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Accepted node arena index per depth (chain order), for the KV
        # compaction that relocates accepted-tree KV to canonical slots.
        self.accepted_arenas = torch.full(
            (max_num_reqs, B + 1), -1, dtype=torch.long, device=device
        )

    def _processed(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-vocab processed distribution over ``[.., V]`` rows, matching
        ``reference_verifier._processed`` (the test oracle) per row (see
        module docstring)."""
        n_e = torch.tensor(_NEG_INF, dtype=logits.dtype, device=logits.device)
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
                kept = torch.where(arange[None, :] < n_keep[:, None], srt, n_e)
                restored = torch.empty_like(logits)
                restored.scatter_(-1, idx, kept)
                logits = torch.where(do_p[:, None], restored, logits)
        return torch.softmax(logits, dim=-1)

    def _draw(
        self,
        p: torch.Tensor,
        greedy_mask: torch.Tensor,
        gens: list[torch.Generator | None] | None,
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
                g = gens[r]
                if g is None:
                    # No generator for this request (unseeded in a mixed batch,
                    # or the caller's -1 sentinel): draw from the global RNG so
                    # it stays independent across steps and batches.
                    x[j] = torch.multinomial(p[j], 1)
                else:
                    x[j] = torch.multinomial(p[j], 1, generator=g)
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
        all_greedy: bool = False,
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
                draws reproducible. A value of ``-1`` requests the global RNG
                for that request (no generator), so an unseeded request in a
                mixed batch is never pinned to a fixed per-step stream; the
                caller derives per-step seeds for explicitly-seeded requests
                (e.g. mixing the raw seed with the anchor position) so
                successive steps draw independent streams.
            all_greedy: host-computed batch flag: every request is greedy
                (T=0). The device loop then draws argmax only, skipping the
                per-round softmax and the discarded batched multinomial (no
                global RNG consumed), so a T=0 batch skips ~16 full-vocab
                passes.

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
        if (
            all_greedy
            and seeds is None
            and temperature is None
            and top_k is None
            and top_p is None
        ):
            return self._verify_greedy(target_logits, token_ids, parents, num_nodes)

        temperature = _row_param(temperature, R, device, torch.float32)
        top_k = _row_param(top_k, R, device, torch.long)
        top_p = _row_param(top_p, R, device, torch.float32)
        if isinstance(greedy, bool):
            greedy_mask = torch.full((R,), greedy, dtype=torch.bool, device=device)
        elif greedy.dim() == 0:
            greedy_mask = greedy.to(torch.bool).expand(R).to(device)
        else:
            greedy_mask = greedy.to(torch.bool).to(device)[:R]

        gens: list[torch.Generator | None] | None = None
        if seeds is not None:
            seeds = seeds.to(torch.long).to(device)
            gens = []
            for r in range(R):
                s = int(seeds[r].item())
                if s == -1:
                    gens.append(None)
                else:
                    gens.append(torch.Generator(device=device).manual_seed(s))

        if seeds is None and temperature is None and top_k is None and top_p is None:
            # Production shape (the runner applies temperature/top-k/top-p
            # through the ordinary sampling pipeline and never passes seeds):
            # a fixed-round, host-sync-free device loop. ``all_greedy`` is a
            # host-computed batch flag (temperature == 0 for every request);
            # when set, the loop draws argmax only — no per-round softmax and
            # no batched multinomial that would be discarded by the greedy
            # mask — so a T=0 batch skips ~16 full-vocab softmax+multinomial
            # passes per step (see the fast path in :meth:`_verify_device`).
            return self._verify_device(
                target_logits,
                token_ids,
                parents,
                num_nodes,
                greedy_mask,
                all_greedy=all_greedy,
            )
        # Seeded/parametrized verification (reproducibility and oracle tests):
        # per-request sequential generators. Kept verbatim so a fixed seed is
        # reproducible and bit-identical to the reference.
        return self._verify_loop(
            target_logits,
            token_ids,
            parents,
            num_nodes,
            greedy_mask,
            temperature,
            top_k,
            top_p,
            gens,
        )

    @property
    def device(self) -> torch.device:
        return self.sampled.device

    def _verify_greedy(
        self,
        target_logits: torch.Tensor,
        token_ids: torch.Tensor,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """T=0 traversal: one block-wide argmax plus one fused walk kernel.

        argmax is monotone-invariant (``argmax(x) == argmax(softmax(x))`` per
        row), so the walk only needs each verify row's argmax.
        """
        return self.walk(
            torch.argmax(target_logits, dim=-1), token_ids, parents, num_nodes
        )

    def walk(
        self,
        row_tokens: torch.Tensor,
        token_ids: torch.Tensor,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Traverse every request's tree with pre-drawn per-row tokens.

        ``row_tokens[r, row]`` is the token drawn at verify row ``row`` (0 =
        anchor, ``1 + arena`` = node). The traversal visits at most one row
        per depth and each row at most once, so drawing every row up front
        (independently, from that row's processed distribution) and reading
        only the visited ones is the same as drawing lazily: argmax rows give
        the T=0 walk, independent samples the sampled one. The fused kernel
        walks each request serially (grid = R) and writes its whole
        ``accepted_arenas`` row and ``num_sampled``.

        Args:
            row_tokens: ``[R, 1 + B]`` int64 drawn token per verify row.
            token_ids: ``[R, B]`` node token per arena (unit column stride).
            parents: ``[R, B]`` parent arena per node (same layout).
            num_nodes: ``[R]`` live node count per request.

        Returns:
            ``(sampled, num_sampled)`` as :meth:`__call__`.

        """
        R, nrows = row_tokens.shape
        B = nrows - 1
        assert row_tokens.is_contiguous()
        num_nodes = num_nodes.contiguous()
        assert token_ids.stride(0) == parents.stride(0)
        assert token_ids.stride(1) == parents.stride(1) == 1
        _tree_walk_kernel[(R,)](
            row_tokens,
            token_ids,
            parents,
            num_nodes,
            self.sampled,
            self.num_sampled,
            self.accepted_arenas,
            token_ids.stride(0),
            B=B,
            num_rounds=self._num_rounds,
            persist_B=self.B,
            ROW_P2=triton.next_power_of_2(self.B + 1),
        )
        return self.sampled[:R, : B + 1], self.num_sampled[:R]

    def _verify_device(
        self,
        target_logits: torch.Tensor,
        token_ids: torch.Tensor,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
        greedy_mask: torch.Tensor,
        all_greedy: bool = False,
        fused: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-shape, host-sync-free traversal over ``num_rounds`` depths.

        Every round processes the full batch with device masks, so the
        per-depth ``nonzero``/``numel`` early-break, the ``bool(...any())``
        draw gate and the per-request generator reads of :meth:`_verify_loop`
        never appear here (no host scalar is ever read in the loop). Draws are
        ``argmax`` on greedy rows and a batched global-RNG ``multinomial`` on
        the rest (dummy uniform probability feeds the non-greedy rows of the
        same batched draw and is discarded by the greedy mask), so a T=0 batch
        never samples and is bit-identical to the reference; done requests are
        masked out of every write and later round. The round count is
        ``self._num_rounds`` (the runner sizes it to the chain cap + 1), so no
        data-dependent host loop bound exists.

        The loop writes through full-batch masked updates only (no data-
        dependent ``torch.nonzero`` compaction), so every intermediate is a
        fixed-size ``[R]``/``[R, B]``/``[R, V]`` tensor.

        When ``all_greedy`` is set the per-round softmax and the batched
        ``multinomial`` are skipped entirely: ``argmax`` is monotone-
        equivalent on raw logits (``argmax(x) == argmax(softmax(x))``), so the
        emitted tokens are bit-identical while a T=0 batch avoids the ~16
        full-vocab softmax + multinomial passes (and no global RNG is
        consumed).
        """
        if all_greedy and fused:
            return self._verify_greedy(target_logits, token_ids, parents, num_nodes)
        R, nrows, _ = target_logits.shape
        B = nrows - 1
        device = target_logits.device
        self.num_sampled[:R].zero_()
        self.accepted_arenas[:R].fill_(-1)
        nn = num_nodes.to(torch.long).clamp(0, B)
        # The unfused all_greedy loop is the torch reference oracle for the
        # fused walk (see _verify_greedy).
        argmax_block = torch.argmax(target_logits, dim=-1) if all_greedy else None
        cur = torch.full((R,), -1, dtype=torch.long, device=device)
        done = torch.zeros(R, dtype=torch.bool, device=device)
        arena = torch.arange(B, device=device)
        arange_r = torch.arange(R, device=device)
        want = ~greedy_mask
        # One block-wide log-softmax for the sampled path instead of a
        # per-round softmax over the drawn row: graded once, gathered per
        # round. exp(log_softmax) is the same distribution as the per-row
        # softmax up to fp rounding.
        logp = None
        if not all_greedy:
            logp = torch.log_softmax(target_logits, dim=-1)
        for d in range(self._num_rounds):
            rows = torch.where(cur == -1, torch.zeros_like(cur), cur + 1)
            active = ~done
            if all_greedy:
                assert argmax_block is not None
                x = argmax_block[arange_r, rows]
            else:
                assert logp is not None
                row_logp = logp[arange_r, rows]
                p_draw = torch.where(
                    want[:, None],
                    row_logp.exp(),
                    torch.ones((), dtype=logp.dtype, device=device),
                )
                x = torch.argmax(row_logp, dim=-1)
                x = torch.where(want, torch.multinomial(p_draw, 1).squeeze(-1), x)
            m = (
                (parents == cur[:, None])
                & (token_ids == x[:, None])
                & (arena[None, :] < nn[:, None])
            )
            child_ok = m.any(dim=-1)
            child_arena = m.to(torch.long).argmax(dim=-1)
            stop = active & ~child_ok
            # Full-batch masked writes: inactive rows keep their current
            # (init/final) values so don't-care columns stay 0 / -1 exactly as
            # the pre-rewrite per-active-row writes left them.
            self.sampled[:R, d] = torch.where(active, x, self.sampled[:R, d])
            self.num_sampled[:R] = torch.where(stop, d + 1, self.num_sampled[:R])
            accept = active & child_ok
            self.accepted_arenas[:R, min(d + 1, B)] = torch.where(
                accept, child_arena, self.accepted_arenas[:R, min(d + 1, B)]
            )
            cur = torch.where(accept, child_arena, cur)
            done = done | stop
        return self.sampled[:R, : B + 1], self.num_sampled[:R]

    def _verify_loop(
        self,
        target_logits: torch.Tensor,
        token_ids: torch.Tensor,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
        greedy_mask: torch.Tensor,
        temperature: torch.Tensor | None,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        gens: list[torch.Generator | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-depth live-subset traversal with per-request generators.

        The reference-exact path: gathers the live (not-yet-done) requests each
        depth, processes their rows, and draws through the per-request
        ``torch.Generator``s (or the global RNG) in request order exactly as
        ``tree_verify_reference`` does. Used for explicit-seed batches and for
        the oracle tests that pass temperature/top-k/top-p; it keeps its host
        scalar reads (they are the price of the sequential-generator contract
        and are off the per-step path, which only ever takes the device loop).
        """
        R, nrows, _ = target_logits.shape
        B = nrows - 1
        device = target_logits.device
        cur = torch.full((R,), -1, dtype=torch.long, device=device)
        done = torch.zeros(R, dtype=torch.bool, device=device)
        self.num_sampled[:R].zero_()
        self.accepted_arenas[:R].fill_(-1)
        arena = torch.arange(B, device=device)

        for d in range(self._num_rounds):
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
            # Record each accepted arena for the next depth (compaction input).
            accept_lids = lids[child_ok_l]
            if accept_lids.numel():
                self.accepted_arenas[accept_lids, d + 1] = child_arena_l[child_ok_l]

        return self.sampled[:R, : B + 1], self.num_sampled[:R]
