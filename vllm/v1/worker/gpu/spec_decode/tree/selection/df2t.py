# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton lazy-k-best tree builder.

Runs the same lazy k-best expansion as ``build_tree_naive`` (see
tree_metadata.py) with one Triton program per request, so the tree you get is
bit-identical to the torch reference: token ids, depths, parents, ancestor
words, and the per-request node count. Sibling masses are priced with the same
float64 accumulation as the reference, and the frontier is popped in the
reference heap's ``(-mass, -parent_mass, path)`` order; the frontier lives in
registers and each pop is a lexicographic argmax over it.

DFlash2 ``_score_edges`` returns UNSORTED per-row successor scores, but the
lazy-next-sibling advance needs each (depth, predecessor) row best-first. The
kernel therefore computes a per-row log-softmax and reorders successors
descending (by rank, ties to the lower index) before the expansion runs,
reading tokens back through the row permutation;
``build_tree_naive._best_first_perm`` mirrors the same permutation, so both
agree on raw unsorted rows.
"""

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.tree.proposal import DraftProposal
from vllm.v1.worker.gpu.spec_decode.tree.routing import (
    MoERoutingMemory,
    routing_bucket,
)
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import TreeDraft


@triton.jit
def _predicted_experts(tok, table_ptr, H, NL, KE, NLP: tl.constexpr, KEP: tl.constexpr):
    """``tok``'s last-seen experts per MoE layer: (layers, experts, valid)."""
    b = routing_bucket(tok, H)
    nl = tl.arange(0, NLP)[:, None]
    kj = tl.arange(0, KEP)[None, :]
    m = (nl < NL) & (kj < KE)
    e = tl.load(table_ptr + (nl * H + b) * KE + kj, mask=m, other=-1).to(tl.int32)
    return nl, tl.maximum(e, 0), m & (e >= 0)


@triton.jit
def _new_expert_cost(
    tok,
    table_ptr,
    act_ptr,
    cost_ptr,
    unknown_cost,
    H,
    NL,
    KE,
    E,
    EW,
    NLP: tl.constexpr,
    KEP: tl.constexpr,
):
    """Cost (ms) of the experts ``tok`` is predicted to need that the active
    set does not have yet; ``unknown_cost`` for a token never seen."""
    nl, e, ok = _predicted_experts(tok, table_ptr, H, NL, KE, NLP, KEP)
    word = tl.load(act_ptr + nl * EW + (e >> 5), mask=ok, other=0, cache_modifier=".cg")
    off = ((word >> (e & 31)) & 1) == 0
    w = tl.load(cost_ptr + nl * E + e, mask=ok, other=0.0)
    cost = tl.sum(tl.sum(tl.where(ok & off, w, 0.0), axis=1), axis=0)
    seen = tl.sum(tl.sum(ok.to(tl.int32), axis=1), axis=0) > 0
    return tl.where(seen, cost, unknown_cost)


@triton.jit
def _activate_experts(
    tok, table_ptr, act_ptr, H, NL, KE, EW, NLP: tl.constexpr, KEP: tl.constexpr
):
    nl, e, ok = _predicted_experts(tok, table_ptr, H, NL, KE, NLP, KEP)
    bit = tl.full(e.shape, 1, tl.int32) << (e & 31)
    tl.atomic_or(act_ptr + nl * EW + (e >> 5), bit, mask=ok)


@triton.jit
def _tree_builder_kernel(
    scores_ptr,  # [R, S, K, K] transition logits
    cand_ptr,  # [R, S, K] i64 successor token ids
    out_ptr,  # [R, 4 * B + 1] i64 packed tokens|depths|parents|words|num_nodes
    logp_out_ptr,  # [R, B] f32 node path log-mass (-inf past the node count)
    sorted_logp_ptr,  # [R, L * K, K] f32 best-first per-row logp
    sorted_cand_ptr,  # [R, L * K, K] i64 best-first per-row tokens
    sorted_idx_ptr,  # [R, L * K, K] i32 best-first per-row candidate indices
    UMSB,  # int64 constant 1 << 63 for unsigned path comparison
    PAD,  # token id written to the inert tail past the live node count
    # Expert-aware selection only (EXPERT_AWARE):
    table_ptr,  # [NL, H, KE] i16 last-seen experts per (layer, token bucket)
    active_ptr,  # [R, NL, EW] i32 per-request active-expert bitmask, zeroed
    cost_ptr,  # [NL, E] f32 ms to activate each expert
    anchor_ptr,  # [num_anchors] the batch's anchor tokens (rows always verified)
    lam_ptr,  # f32 tokens per ms: the exchange rate between cost and mass
    num_anchors,
    unknown_cost,  # ms charged for a token with no recorded routing
    H,
    NL,
    KE,
    E,
    EW,
    B: tl.constexpr,  # emitted nodes per request (<= 64)
    K: tl.constexpr,  # successors per row (<= 16)
    L: tl.constexpr,  # depth cap (<= 16)
    S: tl.constexpr,  # score rows per request (drafter block depth)
    KP: tl.constexpr,  # next power of two >= K
    BP: tl.constexpr,  # next power of two >= B
    CAP: tl.constexpr,  # frontier slots: next power of two >= FAN * (B + 1)
    FAN: tl.constexpr,  # children per parent held in the frontier at once
    EXPERT_AWARE: tl.constexpr,
    NLP: tl.constexpr,  # next power of two >= NL
    KEP: tl.constexpr,  # next power of two >= KE
    TOPM: tl.constexpr,  # frontier entries (best by mass) weighed per pop
):
    row = tl.program_id(0).to(tl.int64)

    # Pass 1: log-softmax each (depth, predecessor) row and sort successors
    # best-first so rank r means the r-th largest transition log-probability
    # (ties: lower candidate index first). Each element's sorted position is
    # its rank among the row, computed for a whole depth at once. The drafter
    # emits S rows per request (block depth) while the tree is capped at
    # L <= S rows, so scores/cand stride by S.
    pr = tl.arange(0, KP)[:, None]
    cc = tl.arange(0, KP)[None, :]
    ok2 = (pr < K) & (cc < K)
    ii = tl.arange(0, KP)[None, :, None]
    jj = tl.arange(0, KP)[None, None, :]
    for d in tl.range(L):
        vals = tl.load(
            scores_ptr + (row * S + d) * (K * K) + pr * K + cc,
            mask=ok2,
            other=float("-inf"),
        ).to(tl.float32)
        vals = tl.where(pr < K, vals, 0.0)
        # A NaN score is no candidate (nor is a -inf one: a top-k with fewer
        # valid entries than K returns sentinel ids there).
        vals = tl.where(vals == vals, vals, float("-inf"))
        mx = tl.max(vals, axis=1)[:, None]
        mx = tl.where(mx > float("-inf"), mx, 0.0)
        logp = (vals - mx) - tl.log(tl.sum(tl.exp(vals - mx), axis=1))[:, None]
        logp = tl.where(ok2 & (vals > float("-inf")), logp, float("-inf"))
        li = tl.expand_dims(logp, 2)
        lj = tl.expand_dims(logp, 1)
        beats = (lj > li) | ((lj == li) & (jj < ii))
        rank = tl.sum(beats.to(tl.int32), axis=2)
        cands = tl.load(cand_ptr + (row * S + d) * K + cc, mask=cc < K, other=0)
        cands = tl.broadcast_to(cands, (KP, KP))
        srow = (row * (L * K) + d * K + pr) * K
        tl.store(sorted_logp_ptr + srow + rank, logp, mask=ok2)
        tl.store(sorted_cand_ptr + srow + rank, cands, mask=ok2)
        tl.store(sorted_idx_ptr + srow + rank, tl.broadcast_to(cc, (KP, KP)), mask=ok2)

    tl.debug_barrier()

    # Pass 2: lazy k-best expansion. The frontier lives in registers; each
    # step pops its best entry by (mass desc, parent mass desc, unsigned
    # packed path asc) -- the reference heap order -- emits it (emit index ==
    # arena index), replaces it with its next sibling and appends its best
    # child. Entries are (mass f64, parent mass f64, depth, rank, prank, parent
    # arena, packed path), where rank is the entry's best-first position among
    # its siblings and prank is its parent's candidate index, which picks the
    # transition row it was ranked in (row 0 under the anchor). Paths pack 4
    # bits per depth; a depth-16 path can set the int64 sign bit, so paths
    # compare unsigned (top bit flipped).
    sorted_row = row * (L * K) * K
    slot = tl.arange(0, CAP)
    ar = tl.arange(0, BP)
    # Bootstrap: the anchor's FAN best children (depth 1, ranks 0..FAN-1;
    # FAN = 1 except for expert-aware selection, which weighs siblings
    # against each other); the anchor itself is never emitted. Each parent
    # keeps FAN children in the frontier: popping rank r brings in r + FAN.
    first = (slot < FAN) & (slot < K)
    mass = tl.load(sorted_logp_ptr + sorted_row + slot, mask=first, other=0.0).to(
        tl.float64
    )
    pmass = tl.zeros([CAP], dtype=tl.float64)
    path = tl.where(first, slot, 0).to(tl.int64)
    depth = tl.where(first, 1, 0)
    rank = tl.where(first, slot, 0)
    prank = tl.zeros([CAP], dtype=tl.int32)
    pa = tl.full([CAP], -1, dtype=tl.int32)
    # A frontier entry of zero probability (-inf mass) is never emitted, so a
    # proposal with fewer than B possible nodes yields a smaller tree.
    valid = first & (mass > float("-inf"))
    tok_acc = tl.zeros([BP], dtype=tl.int64)
    dep_acc = tl.zeros([BP], dtype=tl.int64)
    par_acc = tl.zeros([BP], dtype=tl.int64)
    word_acc = tl.zeros([BP], dtype=tl.int64)
    lp_acc = tl.zeros([BP], dtype=tl.float32)
    neg_inf = float("-inf")
    key_max = 0x7FFFFFFFFFFFFFFF

    act_row = active_ptr + row * NL * EW
    lam = 0.0
    if EXPERT_AWARE:
        # Every request's anchor row is verified, so the batch's anchors'
        # experts are active before any node is chosen.
        for a in range(num_anchors):
            _activate_experts(
                tl.load(anchor_ptr + a), table_ptr, act_row, H, NL, KE, EW, NLP, KEP
            )
        tl.debug_barrier()
        lam = tl.load(lam_ptr).to(tl.float32)

    emit = tl.full((), 0, tl.int32)
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    nxt = tl.full((), FAN, tl.int32)
    while (emit < B) & (nvalid > 0):
        if EXPERT_AWARE:
            # Weigh the TOPM most probable frontier entries by acceptance mass
            # gained minus the cost of the experts they would add, in tokens
            # (lam converts ms to tokens, the throughput trade CaDDTree's
            # stopping rule makes). Any frontier entry keeps the tree
            # prefix-closed.
            remaining = valid
            best_score = tl.full((), float("-inf"), tl.float32)
            best_slot = tl.full((), CAP, tl.int32)
            for _ in tl.static_range(TOPM):
                m_mass = tl.max(tl.where(remaining, mass, neg_inf), axis=0)
                m_slot = tl.min(
                    tl.where(remaining & (mass == m_mass), slot, CAP), axis=0
                )
                ms = slot == m_slot
                live_m = m_slot < CAP
                m_row = (tl.sum(tl.where(ms, depth, 0), axis=0) - 1) * K + tl.sum(
                    tl.where(ms, prank, 0), axis=0
                )
                m_tok = tl.load(
                    sorted_cand_ptr
                    + sorted_row
                    + m_row * K
                    + tl.sum(tl.where(ms, rank, 0), axis=0),
                    mask=live_m,
                    other=0,
                )
                m_cost = _new_expert_cost(
                    m_tok,
                    table_ptr,
                    act_row,
                    cost_ptr,
                    unknown_cost,
                    H,
                    NL,
                    KE,
                    E,
                    EW,
                    NLP,
                    KEP,
                )
                score = tl.exp(m_mass.to(tl.float32)) - lam * m_cost
                better = live_m & (score > best_score)
                best_score = tl.where(better, score, best_score)
                best_slot = tl.where(better, m_slot, best_slot)
                remaining = remaining & ~ms
            sel = slot == best_slot
        else:
            top_mass = tl.max(tl.where(valid, mass, neg_inf), axis=0)
            c = valid & (mass == top_mass)
            top_pmass = tl.max(tl.where(c, pmass, neg_inf), axis=0)
            c = c & (pmass == top_pmass)
            key = path ^ UMSB
            c = c & (key == tl.min(tl.where(c, key, key_max), axis=0))
            sel = slot == tl.min(tl.where(c, slot, CAP), axis=0)
        top_mass = tl.sum(tl.where(sel, mass, 0.0), axis=0)
        top_pmass = tl.sum(tl.where(sel, pmass, 0.0), axis=0)
        top_depth = tl.sum(tl.where(sel, depth, 0), axis=0)
        top_rank = tl.sum(tl.where(sel, rank, 0), axis=0)
        top_prank = tl.sum(tl.where(sel, prank, 0), axis=0)
        top_pa = tl.sum(tl.where(sel, pa, 0), axis=0)
        top_path = tl.sum(tl.where(sel, path, 0), axis=0)

        drow = (top_depth - 1) * K + top_prank
        sib_rank = top_rank + FAN
        has_sib = sib_rank < K
        has_child = top_depth < L
        token = tl.load(sorted_cand_ptr + sorted_row + drow * K + top_rank)
        if EXPERT_AWARE:
            _activate_experts(token, table_ptr, act_row, H, NL, KE, EW, NLP, KEP)
            tl.debug_barrier()
        # The popped node's own candidate index picks its children's row.
        top_idx = tl.load(sorted_idx_ptr + sorted_row + drow * K + top_rank)
        crow = top_depth * K + top_idx
        sib_logp = tl.load(
            sorted_logp_ptr + sorted_row + drow * K + sib_rank, mask=has_sib, other=0.0
        )
        # The popped node's FAN best children enter slots nxt .. nxt+FAN-1.
        child_rank = slot - nxt
        is_child = (child_rank >= 0) & (child_rank < FAN) & (child_rank < K) & has_child
        child_logp = tl.load(
            sorted_logp_ptr + sorted_row + crow * K + child_rank,
            mask=is_child,
            other=0.0,
        )

        at = ar == emit
        pword = tl.sum(tl.where(ar == top_pa, word_acc, 0), axis=0)
        tok_acc = tl.where(at, token, tok_acc)
        dep_acc = tl.where(at, top_depth.to(tl.int64), dep_acc)
        par_acc = tl.where(at, top_pa.to(tl.int64), par_acc)
        word_acc = tl.where(at, pword | (tl.full((), 1, tl.int64) << emit), word_acc)
        lp_acc = tl.where(at, top_mass.to(tl.float32), lp_acc)

        # Next sibling of the popped node (priced against the parent's mass)
        # takes over its slot; the rank advance is capped by K, not depth.
        mass = tl.where(sel, top_pmass + sib_logp.to(tl.float64), mass)
        rank = tl.where(sel, sib_rank, rank)
        path = tl.where(sel, (top_path // 16) * 16 + sib_rank, path)
        valid = valid & (~sel | has_sib)

        # The best children keep expanding while the depth cap allows.
        new = is_child
        mass = tl.where(new, top_mass + child_logp.to(tl.float64), mass)
        pmass = tl.where(new, top_mass, pmass)
        depth = tl.where(new, top_depth + 1, depth)
        rank = tl.where(new, child_rank, rank)
        prank = tl.where(new, top_idx, prank)
        pa = tl.where(new, emit, pa)
        path = tl.where(new, top_path * 16 + child_rank, path)
        valid = (valid | new) & (mass > neg_inf)

        nxt = nxt + FAN * has_child.to(tl.int32)
        nvalid = tl.sum(valid.to(tl.int32), axis=0)
        emit = emit + 1

    # Columns past the live node count are written inert (mask token, depth
    # 0, parent -2, no ancestor bits) so no consumer can see a stale node.
    live = ar < emit
    in_b = ar < B
    base = row * (4 * B + 1)
    tl.store(out_ptr + base + ar, tl.where(live, tok_acc, PAD), mask=in_b)
    tl.store(out_ptr + base + B + ar, tl.where(live, dep_acc, 0), mask=in_b)
    tl.store(out_ptr + base + 2 * B + ar, tl.where(live, par_acc, -2), mask=in_b)
    tl.store(out_ptr + base + 3 * B + ar, tl.where(live, word_acc, 0), mask=in_b)
    tl.store(out_ptr + base + 4 * B, emit.to(tl.int64))
    tl.store(logp_out_ptr + row * B + ar, tl.where(live, lp_acc, neg_inf), mask=in_b)


class TreeBuilder:
    """Host wrapper around the lazy-k-best triton tree builder.

    All scratch and output buffers are allocated once (``max_num_reqs`` rows,
    ``B`` nodes each) and reused across ``build`` calls; no host-side heap or
    copy lands in the hot path.

    Args:
        max_num_reqs: maximum concurrent requests whose tree fits the buffers.
        L: tree depth cap (<= 15).
        K: successors per row (<= 16).
        B: nodes emitted per request (<= 64).
        device: device the buffers and kernel run on.
        num_steps: score rows per request (drafter block depth); defaults to L.
        num_warps: warps per request program (a per-GPU tuning knob).
        pad_token_id: token id of the inert columns past a tree's node count.

    """

    def __init__(
        self,
        max_num_reqs: int,
        L: int,
        K: int,
        B: int,
        device: torch.device | str = "cuda",
        num_steps: int | None = None,
        num_warps: int = 2,
        pad_token_id: int = 0,
    ) -> None:
        assert L <= 16, "tree depth cap must be <= 16 for the packed path"
        assert K <= 16, "successor count must be <= 16"
        assert B <= 64, "emitted nodes must be <= 64"
        if num_steps is None:
            num_steps = L
        # The drafter emits num_steps score rows per request; the tree is
        # capped at L <= num_steps of them. The kernel reads scores/cand at
        # stride num_steps so a request never reads another request's rows.
        assert num_steps >= L, "score rows (drafter depth) must cover the depth cap"
        self.max_num_reqs = max_num_reqs
        self.L = L
        self.K = K
        self.B = B
        self.num_steps = num_steps
        # Paths are packed 4 bits per depth; a depth-16 path can set the int64
        # sign bit, so the heap tie-break compares paths unsignedly.
        self._umsb = -9223372036854775808  # 1 << 63 as int64
        self._kp = triton.next_power_of_2(K)
        self._bp = triton.next_power_of_2(B)
        self._cap = triton.next_power_of_2(B + 1)
        self._num_warps = num_warps
        self.pad_token_id = pad_token_id
        dev = torch.device(device)

        self._sorted_logp = torch.zeros(
            max_num_reqs, L * K, K, dtype=torch.float32, device=dev
        )
        self._sorted_cand = torch.zeros(
            max_num_reqs, L * K, K, dtype=torch.int64, device=dev
        )
        self._sorted_idx = torch.zeros(
            max_num_reqs, L * K, K, dtype=torch.int32, device=dev
        )

        # One packed row per request, [tokens | depths | parents | words |
        # num_nodes], so the tree moves (device store, host carry) as a single
        # copy; the named fields are strided views of it.
        self.packed = torch.zeros(
            max_num_reqs, 4 * B + 1, dtype=torch.int64, device=dev
        )
        self.token_ids = self.packed[:, :B]
        self.depths = self.packed[:, B : 2 * B]
        self.parents = self.packed[:, 2 * B : 3 * B]
        self.words = self.packed[:, 3 * B : 4 * B]
        self.num_nodes = self.packed[:, 4 * B]
        # Each node's drafter path log-probability, best-first (non-increasing
        # along a row), for adaptive verify-width selection.
        self.logp = torch.full(
            (max_num_reqs, B), float("-inf"), dtype=torch.float32, device=dev
        )
        self.arena = (
            torch.arange(B, dtype=torch.int64, device=dev)
            .expand(max_num_reqs, B)
            .contiguous()
        )
        # Expert-aware selection (bind_routing); None selects by mass alone.
        self.routing: MoERoutingMemory | None = None
        self.lam: Tensor | None = None
        self._active: Tensor | None = None
        self.top_m = 4

    def bind_routing(self, routing: MoERoutingMemory, lam: Tensor) -> None:
        """Select expert-aware: weigh each node's acceptance mass against the
        cost of the MoE experts it would add to the verify step, converted to
        tokens by ``lam`` (a persistent 1-element f32 tensor, tokens per ms,
        that the caller keeps current)."""
        self.routing = routing
        self.lam = lam
        words = (routing.num_experts + 31) // 32
        self._active = torch.zeros(
            self.max_num_reqs,
            routing.num_layers,
            words,
            dtype=torch.int32,
            device=self.packed.device,
        )

    def select(
        self, proposal: DraftProposal, anchor_ids: Tensor | None = None
    ) -> TreeDraft:
        """The tree over the proposal's top ``K`` candidates per position,
        ranked by their conditional logits (each candidate's own logit when
        the drafter's positions are independent). ``anchor_ids`` are the
        batch's anchor tokens, for expert-aware selection."""
        cand, unary = proposal.candidates(self.K)
        scores = proposal.conditional(cand)
        if scores is None:
            scores = unary.float().unsqueeze(-2).expand(*unary.shape[:2], self.K, -1)
        return self.build(scores, cand, anchor_ids)

    def build(
        self, scores: Tensor, cand: Tensor, anchor_ids: Tensor | None = None
    ) -> TreeDraft:
        """Populate the tree for each request.

        Args:
            scores: [R, L, K, K] transition logits, rows in any order (sorted
                internally best-first).
            cand: [R, L, K] successor token ids per (depth, rank).
            anchor_ids: [R] the batch's anchor tokens; required once
                expert-aware selection is bound.

        Returns:
            A TreeDraft in emit order (emit index == arena index): token_ids,
            depths, parents, arena and words, all [R, B] int64 on device.

        """
        R = scores.shape[0]
        scores = scores.contiguous()
        cand = cand.contiguous()
        routing = self.routing
        if routing is not None:
            assert anchor_ids is not None and self._active is not None
            assert self.lam is not None
            self._active[:R].zero_()
            route_args = (
                routing.table,
                self._active,
                routing.cost_ms,
                anchor_ids,
                self.lam,
                anchor_ids.shape[0],
                routing.unknown_cost_ms,
                routing.buckets,
                routing.num_layers,
                routing.top_k,
                routing.num_experts,
                self._active.shape[2],
            )
            nlp = triton.next_power_of_2(routing.num_layers)
            kep = triton.next_power_of_2(routing.top_k)
            fan = self.top_m
        else:
            # Inert placeholders: the expert-aware code is compiled out.
            route_args = (self.packed,) * 5 + (0, 0.0, 1, 1, 1, 1, 1)
            nlp = kep = fan = 1
        _tree_builder_kernel[(R,)](
            scores,
            cand,
            self.packed,
            self.logp,
            self._sorted_logp,
            self._sorted_cand,
            self._sorted_idx,
            self._umsb,
            self.pad_token_id,
            *route_args,
            B=self.B,
            K=self.K,
            L=self.L,
            S=self.num_steps,
            KP=self._kp,
            BP=self._bp,
            CAP=triton.next_power_of_2(fan * (self.B + 1)),
            FAN=fan,
            EXPERT_AWARE=routing is not None,
            NLP=nlp,
            KEP=kep,
            TOPM=self.top_m,
            num_warps=self._num_warps,
        )
        return TreeDraft(
            token_ids=self.token_ids[:R],
            depths=self.depths[:R],
            parents=self.parents[:R],
            arena=self.arena[:R],
            words=self.words[:R],
            num_nodes=self.num_nodes[:R],
        )
