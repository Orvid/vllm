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
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import TreeDraft


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
    B: tl.constexpr,  # emitted nodes per request (<= 64)
    K: tl.constexpr,  # successors per row (<= 16)
    L: tl.constexpr,  # depth cap (<= 16)
    S: tl.constexpr,  # score rows per request (drafter block depth)
    KP: tl.constexpr,  # next power of two >= K
    BP: tl.constexpr,  # next power of two >= B
    CAP: tl.constexpr,  # frontier slots: next power of two >= B + 1
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
    # Bootstrap: the anchor's best child (depth 1, rank 0); the anchor itself
    # is never emitted.
    m0 = tl.load(sorted_logp_ptr + sorted_row).to(tl.float64)
    first = slot == 0
    mass = tl.where(first, m0, 0.0)
    pmass = tl.zeros([CAP], dtype=tl.float64)
    path = tl.zeros([CAP], dtype=tl.int64)
    depth = tl.where(first, 1, 0)
    rank = tl.zeros([CAP], dtype=tl.int32)
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

    emit = tl.full((), 0, tl.int32)
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    nxt = tl.full((), 1, tl.int32)
    while (emit < B) & (nvalid > 0):
        top_mass = tl.max(tl.where(valid, mass, neg_inf), axis=0)
        c = valid & (mass == top_mass)
        top_pmass = tl.max(tl.where(c, pmass, neg_inf), axis=0)
        c = c & (pmass == top_pmass)
        key = path ^ UMSB
        c = c & (key == tl.min(tl.where(c, key, key_max), axis=0))
        sel = slot == tl.min(tl.where(c, slot, CAP), axis=0)
        top_depth = tl.sum(tl.where(sel, depth, 0), axis=0)
        top_rank = tl.sum(tl.where(sel, rank, 0), axis=0)
        top_prank = tl.sum(tl.where(sel, prank, 0), axis=0)
        top_pa = tl.sum(tl.where(sel, pa, 0), axis=0)
        top_path = tl.sum(tl.where(sel, path, 0), axis=0)

        drow = (top_depth - 1) * K + top_prank
        sib_rank = top_rank + 1
        has_sib = sib_rank < K
        has_child = top_depth < L
        token = tl.load(sorted_cand_ptr + sorted_row + drow * K + top_rank)
        # The popped node's own candidate index picks its children's row.
        top_idx = tl.load(sorted_idx_ptr + sorted_row + drow * K + top_rank)
        crow = top_depth * K + top_idx
        sib_logp = tl.load(
            sorted_logp_ptr + sorted_row + drow * K + sib_rank, mask=has_sib, other=0.0
        )
        child_logp = tl.load(
            sorted_logp_ptr + sorted_row + crow * K, mask=has_child, other=0.0
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

        # The best child keeps expanding while the depth cap allows.
        new = (slot == nxt) & has_child
        mass = tl.where(new, top_mass + child_logp.to(tl.float64), mass)
        pmass = tl.where(new, top_mass, pmass)
        depth = tl.where(new, top_depth + 1, depth)
        rank = tl.where(new, 0, rank)
        prank = tl.where(new, top_idx, prank)
        pa = tl.where(new, emit, pa)
        path = tl.where(new, top_path * 16, path)
        valid = (valid | new) & (mass > neg_inf)

        nxt = nxt + has_child.to(tl.int32)
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

    def select(self, proposal: DraftProposal) -> TreeDraft:
        """The tree over the proposal's top ``K`` candidates per position,
        ranked by their conditional logits (each candidate's own logit when
        the drafter's positions are independent)."""
        cand, unary = proposal.candidates(self.K)
        scores = proposal.conditional(cand)
        if scores is None:
            scores = unary.float().unsqueeze(-2).expand(*unary.shape[:2], self.K, -1)
        return self.build(scores, cand)

    def build(self, scores: Tensor, cand: Tensor) -> TreeDraft:
        """Populate the tree for each request.

        Args:
            scores: [R, L, K, K] transition logits, rows in any order (sorted
                internally best-first).
            cand: [R, L, K] successor token ids per (depth, rank).

        Returns:
            A TreeDraft in emit order (emit index == arena index): token_ids,
            depths, parents, arena and words, all [R, B] int64 on device.

        """
        R = scores.shape[0]
        scores = scores.contiguous()
        cand = cand.contiguous()
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
            B=self.B,
            K=self.K,
            L=self.L,
            S=self.num_steps,
            KP=self._kp,
            BP=self._bp,
            CAP=self._cap,
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
