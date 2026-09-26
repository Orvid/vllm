# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton lazy-k-best tree builder.

Runs the same lazy k-best heap as ``build_tree_naive`` (see tree_metadata.py)
with one Triton program per request, so the tree you get is bit-identical to
the torch reference: token ids, depths, parents, ancestor words, and the
per-request node count. Sibling masses are priced with the same float64
accumulation as the reference, and ties are broken by parent mass then by the
packed path, mirroring the reference's ``(-mass, -parent_mass, path)`` heap.

DFlash2 ``_score_edges`` returns UNSORTED per-row successor scores, but the
lazy-next-sibling advance needs each (depth, predecessor) row best-first. The
kernel therefore computes a per-row log-softmax and reorders successors
descending before the heap runs, reading tokens back through the row
permutation; ``build_tree_naive._best_first_perm`` mirrors the same
permutation, so both agree on raw unsorted rows.
"""

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft


@triton.jit
def _heap_above(mass_ptr, pmass_ptr, path_ptr, hbase, ix, jx):
    im = tl.load(mass_ptr + hbase + ix)
    ip = tl.load(pmass_ptr + hbase + ix)
    it = tl.load(path_ptr + hbase + ix)
    jm = tl.load(mass_ptr + hbase + jx)
    jp = tl.load(pmass_ptr + hbase + jx)
    jt = tl.load(path_ptr + hbase + jx)
    return (im > jm) | ((im == jm) & ((ip > jp) | ((ip == jp) & (it < jt))))


@triton.jit
def _heap_swap(
    mass_ptr, pmass_ptr, depth_ptr, rank_ptr, prank_ptr, pa_ptr, path_ptr, hbase, ix, jx
):
    sm = tl.load(mass_ptr + hbase + ix)
    sp = tl.load(pmass_ptr + hbase + ix)
    sd = tl.load(depth_ptr + hbase + ix)
    sr = tl.load(rank_ptr + hbase + ix)
    sp2 = tl.load(prank_ptr + hbase + ix)
    sa = tl.load(pa_ptr + hbase + ix)
    st = tl.load(path_ptr + hbase + ix)
    tl.store(mass_ptr + hbase + ix, tl.load(mass_ptr + hbase + jx))
    tl.store(pmass_ptr + hbase + ix, tl.load(pmass_ptr + hbase + jx))
    tl.store(depth_ptr + hbase + ix, tl.load(depth_ptr + hbase + jx))
    tl.store(rank_ptr + hbase + ix, tl.load(rank_ptr + hbase + jx))
    tl.store(prank_ptr + hbase + ix, tl.load(prank_ptr + hbase + jx))
    tl.store(pa_ptr + hbase + ix, tl.load(pa_ptr + hbase + jx))
    tl.store(path_ptr + hbase + ix, tl.load(path_ptr + hbase + jx))
    tl.store(mass_ptr + hbase + jx, sm)
    tl.store(pmass_ptr + hbase + jx, sp)
    tl.store(depth_ptr + hbase + jx, sd)
    tl.store(rank_ptr + hbase + jx, sr)
    tl.store(prank_ptr + hbase + jx, sp2)
    tl.store(pa_ptr + hbase + jx, sa)
    tl.store(path_ptr + hbase + jx, st)


@triton.jit
def _tree_builder_kernel(
    scores_ptr,  # [R, L, K, K] f32 transition logits
    cand_ptr,  # [R, L, K] i64 successor token ids
    token_out_ptr,  # [R, B] i64
    depth_out_ptr,  # [R, B] i64
    parent_out_ptr,  # [R, B] i64
    word_out_ptr,  # [R, B] i64
    num_nodes_ptr,  # [R] i32
    sorted_logp_ptr,  # [R, L * K, K] f32 best-first per-row logp
    sorted_cand_ptr,  # [R, L * K, K] i64 best-first per-row tokens
    heap_mass_ptr,  # [R, TWO_B] f64
    heap_pmass_ptr,  # [R, TWO_B] f64
    heap_depth_ptr,  # [R, TWO_B] i32
    heap_rank_ptr,  # [R, TWO_B] i32
    heap_prank_ptr,  # [R, TWO_B] i32
    heap_pa_ptr,  # [R, TWO_B] i32
    heap_path_ptr,  # [R, TWO_B] i64
    B: tl.constexpr,  # emitted nodes per request (<= 64)
    K: tl.constexpr,  # successors per row (<= 16)
    L: tl.constexpr,  # depth cap (<= 15)
    TWO_B: tl.constexpr,
    KP: tl.constexpr,  # next power of two >= K
):
    row = tl.program_id(0).to(tl.int64)
    hbase = row * TWO_B
    col = tl.arange(0, KP)
    ok = col < K

    # Pass 1: log-softmax each (depth, predecessor) row and sort successors
    # best-first so rank r means the r-th largest transition log-probability.
    for d in tl.range(L):
        for p in tl.range(K):
            vals = tl.load(
                scores_ptr + (row * L + d) * (K * K) + p * K + col,
                mask=ok,
                other=float("-inf"),
            ).to(tl.float32)
            mx = tl.max(vals, axis=0)
            logp = (vals - mx) - tl.log(tl.sum(tl.exp(vals - mx), axis=0))
            cands = tl.load(cand_ptr + (row * L + d) * K + col, mask=ok, other=0)

            remaining = tl.where(ok, logp, float("-inf"))
            srow = (row * (L * K) + d * K + p) * K
            for kk in tl.range(K):
                best = tl.argmax(remaining, axis=0)
                tl.store(
                    sorted_logp_ptr + srow + kk,
                    tl.max(tl.where(col == best, remaining, float("-inf")), axis=0),
                )
                tl.store(
                    sorted_cand_ptr + srow + kk,
                    tl.sum(tl.where(col == best, cands, 0), axis=0),
                )
                remaining = tl.where(col == best, float("-inf"), remaining)

    tl.debug_barrier()

    # Heap entries are (mass f64, parent mass f64, depth, rank, prank, parent
    # arena, packed path). Each tree node is emitted exactly once, in pop order,
    # and emit index is the node's arena index.

    # Bootstrap: push the anchor's best child (depth 1, rank 0). The anchor
    # itself is never emitted, matching the reference's empty root path.
    # sorted buffers are [R, L * K, K]: request row stride is L * K * K.
    sorted_row = row * (L * K) * K
    tl.store(
        heap_mass_ptr + hbase + 0, tl.load(sorted_logp_ptr + sorted_row).to(tl.float64)
    )
    tl.store(heap_pmass_ptr + hbase + 0, 0.0)
    tl.store(heap_depth_ptr + hbase + 0, 1)
    tl.store(heap_rank_ptr + hbase + 0, 0)
    tl.store(heap_prank_ptr + hbase + 0, 0)
    tl.store(heap_pa_ptr + hbase + 0, -1)
    tl.store(heap_path_ptr + hbase + 0, 0)

    emit = tl.full((), 0, tl.int32)
    hn = tl.full((), 1, tl.int32)
    while (emit < B) & (hn > 0):
        # The heap is read-modify-written through global memory each iteration;
        # the barrier forces the previous iteration's pushes/sifts to commit so
        # the pop below reads the freshest root.
        tl.debug_barrier()
        top_mass = tl.load(heap_mass_ptr + hbase + 0)
        top_pmass = tl.load(heap_pmass_ptr + hbase + 0)
        top_depth = tl.load(heap_depth_ptr + hbase + 0)
        top_rank = tl.load(heap_rank_ptr + hbase + 0)
        top_prank = tl.load(heap_prank_ptr + hbase + 0)
        top_pa = tl.load(heap_pa_ptr + hbase + 0)
        top_path = tl.load(heap_path_ptr + hbase + 0)
        hn = hn - 1
        tl.store(heap_mass_ptr + hbase + 0, tl.load(heap_mass_ptr + hbase + hn))
        tl.store(heap_pmass_ptr + hbase + 0, tl.load(heap_pmass_ptr + hbase + hn))
        tl.store(heap_depth_ptr + hbase + 0, tl.load(heap_depth_ptr + hbase + hn))
        tl.store(heap_rank_ptr + hbase + 0, tl.load(heap_rank_ptr + hbase + hn))
        tl.store(heap_prank_ptr + hbase + 0, tl.load(heap_prank_ptr + hbase + hn))
        tl.store(heap_pa_ptr + hbase + 0, tl.load(heap_pa_ptr + hbase + hn))
        tl.store(heap_path_ptr + hbase + 0, tl.load(heap_path_ptr + hbase + hn))
        i = tl.full((), 0, tl.int32)
        sift = tl.full((), True, tl.int1)
        while sift:
            li = 2 * i + 1
            ri = 2 * i + 2
            largest = i
            # SIM102: keep the child-index guard short-circuit so the heap
            # reads stay inside the populated entries.
            if li < hn:  # noqa: SIM102
                if _heap_above(
                    heap_mass_ptr, heap_pmass_ptr, heap_path_ptr, hbase, li, largest
                ):
                    largest = li
            if ri < hn:  # noqa: SIM102
                if _heap_above(
                    heap_mass_ptr, heap_pmass_ptr, heap_path_ptr, hbase, ri, largest
                ):
                    largest = ri
            if largest == i:
                sift = tl.full((), False, tl.int1)
            else:
                _heap_swap(
                    heap_mass_ptr,
                    heap_pmass_ptr,
                    heap_depth_ptr,
                    heap_rank_ptr,
                    heap_prank_ptr,
                    heap_pa_ptr,
                    heap_path_ptr,
                    hbase,
                    largest,
                    i,
                )
                i = largest

        # Emit the popped node (emit index == arena index).
        drow = (top_depth - 1) * K + top_prank
        token = tl.load(sorted_cand_ptr + sorted_row + drow * K + top_rank)
        tl.store(token_out_ptr + row * B + emit, token)
        tl.store(depth_out_ptr + row * B + emit, top_depth.to(tl.int64))
        tl.store(parent_out_ptr + row * B + emit, top_pa.to(tl.int64))
        bit = tl.full((), 1, tl.int64) << emit
        pword = tl.load(word_out_ptr + row * B + top_pa, mask=top_pa >= 0, other=0)
        tl.store(word_out_ptr + row * B + emit, pword | bit)

        # Successor of the same parent (next sibling), priced against the
        # parent's mass. Rank advance is capped by K, not depth.
        sib_rank = top_rank + 1
        if sib_rank < K:
            sib_mass = top_pmass + tl.load(
                sorted_logp_ptr + sorted_row + drow * K + sib_rank
            ).to(tl.float64)
            tl.store(heap_mass_ptr + hbase + hn, sib_mass)
            tl.store(heap_pmass_ptr + hbase + hn, top_pmass)
            tl.store(heap_depth_ptr + hbase + hn, top_depth)
            tl.store(heap_rank_ptr + hbase + hn, sib_rank)
            tl.store(heap_prank_ptr + hbase + hn, top_prank)
            tl.store(heap_pa_ptr + hbase + hn, top_pa)
            tl.store(heap_path_ptr + hbase + hn, (top_path // 16) * 16 + sib_rank)
            hn = hn + 1
            # sift the new entry up to its priority slot
            iu = hn - 1
            up_go = tl.full((), True, tl.int1)
            while up_go & (iu > 0):
                pu = (iu - 1) // 2
                if _heap_above(
                    heap_mass_ptr, heap_pmass_ptr, heap_path_ptr, hbase, iu, pu
                ):
                    _heap_swap(
                        heap_mass_ptr,
                        heap_pmass_ptr,
                        heap_depth_ptr,
                        heap_rank_ptr,
                        heap_prank_ptr,
                        heap_pa_ptr,
                        heap_path_ptr,
                        hbase,
                        iu,
                        pu,
                    )
                    iu = pu
                else:
                    up_go = tl.full((), False, tl.int1)

        # Best child keeps expanding while the depth cap allows.
        if top_depth < L:
            crow = top_depth * K + top_rank
            child_mass = top_mass + tl.load(sorted_logp_ptr + sorted_row + crow * K).to(
                tl.float64
            )
            tl.store(heap_mass_ptr + hbase + hn, child_mass)
            tl.store(heap_pmass_ptr + hbase + hn, top_mass)
            tl.store(heap_depth_ptr + hbase + hn, top_depth + 1)
            tl.store(heap_rank_ptr + hbase + hn, 0)
            tl.store(heap_prank_ptr + hbase + hn, top_rank)
            tl.store(heap_pa_ptr + hbase + hn, emit)
            tl.store(heap_path_ptr + hbase + hn, top_path * 16)
            hn = hn + 1
            # sift the new entry up to its priority slot
            iu = hn - 1
            up_go = tl.full((), True, tl.int1)
            while up_go & (iu > 0):
                pu = (iu - 1) // 2
                if _heap_above(
                    heap_mass_ptr, heap_pmass_ptr, heap_path_ptr, hbase, iu, pu
                ):
                    _heap_swap(
                        heap_mass_ptr,
                        heap_pmass_ptr,
                        heap_depth_ptr,
                        heap_rank_ptr,
                        heap_prank_ptr,
                        heap_pa_ptr,
                        heap_path_ptr,
                        hbase,
                        iu,
                        pu,
                    )
                    iu = pu
                else:
                    up_go = tl.full((), False, tl.int1)

        emit = emit + 1

    tl.store(num_nodes_ptr + row, emit)


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

    """

    def __init__(
        self,
        max_num_reqs: int,
        L: int,
        K: int,
        B: int,
        device: torch.device | str = "cuda",
    ) -> None:
        assert L <= 15, "tree depth cap must be <= 15 for the packed path"
        assert K <= 16, "successor count must be <= 16"
        assert B <= 64, "emitted nodes must be <= 64"
        self.max_num_reqs = max_num_reqs
        self.L = L
        self.K = K
        self.B = B
        two_b = 2 * B
        self._kp = triton.next_power_of_2(K)
        dev = torch.device(device)

        self._sorted_logp = torch.zeros(
            max_num_reqs, L * K, K, dtype=torch.float32, device=dev
        )
        self._sorted_cand = torch.zeros(
            max_num_reqs, L * K, K, dtype=torch.int64, device=dev
        )
        self._heap_mass = torch.empty(
            max_num_reqs, two_b, dtype=torch.float64, device=dev
        )
        self._heap_pmass = torch.empty(
            max_num_reqs, two_b, dtype=torch.float64, device=dev
        )
        self._heap_depth = torch.empty(
            max_num_reqs, two_b, dtype=torch.int32, device=dev
        )
        self._heap_rank = torch.empty(
            max_num_reqs, two_b, dtype=torch.int32, device=dev
        )
        self._heap_prank = torch.empty(
            max_num_reqs, two_b, dtype=torch.int32, device=dev
        )
        self._heap_pa = torch.empty(max_num_reqs, two_b, dtype=torch.int32, device=dev)
        self._heap_path = torch.empty(
            max_num_reqs, two_b, dtype=torch.int64, device=dev
        )

        self.token_ids = torch.empty(max_num_reqs, B, dtype=torch.int64, device=dev)
        self.depths = torch.empty(max_num_reqs, B, dtype=torch.int64, device=dev)
        self.parents = torch.empty(max_num_reqs, B, dtype=torch.int64, device=dev)
        self.words = torch.empty(max_num_reqs, B, dtype=torch.int64, device=dev)
        self.num_nodes = torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        self.arena = (
            torch.arange(B, dtype=torch.int64, device=dev)
            .expand(max_num_reqs, B)
            .contiguous()
        )

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
            self.token_ids,
            self.depths,
            self.parents,
            self.words,
            self.num_nodes,
            self._sorted_logp,
            self._sorted_cand,
            self._heap_mass,
            self._heap_pmass,
            self._heap_depth,
            self._heap_rank,
            self._heap_prank,
            self._heap_pa,
            self._heap_path,
            B=self.B,
            K=self.K,
            L=self.L,
            TWO_B=2 * self.B,
            KP=self._kp,
            num_warps=1,
        )
        return TreeDraft(
            token_ids=self.token_ids[:R],
            depths=self.depths[:R],
            parents=self.parents[:R],
            arena=self.arena[:R],
            words=self.words[:R],
            num_nodes=self.num_nodes[:R],
        )
