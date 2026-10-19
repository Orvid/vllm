# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-aware sampling feature drills for the tree verify block.

The additive logits processors (penalties, bad words, thinking budget and
custom processors) treat the verify block as a linear draft chain: a row's
"already emitted" prefix is the arena order of the rows before it. Under a
tree that assumption is wrong — siblings and other branches sit in arena
order and are not on the accepted chain — so a repetition/frequency/presence
penalty, a bad-word mask or a budget force would count/see sibling tokens and
the processed distribution diverges from the chain (and non-speculative)
oracle. These tests pin the tree-correct per-row accepted-chain context
mechanism: the three triton kernels, fed the chain_ prefix tables the model
runner now builds, reproduce an independent reference exactly, and the serving
draws are seeded per (request seed, position).
"""

import pytest
import torch

from tests.v1.worker.gpu.spec_decode.tree.reference_verifier import (
    tree_verify_reference,
)
from vllm.v1.worker.gpu.model_runner import _tree_chain_context
from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words
from vllm.v1.worker.gpu.sample.penalties import apply_penalties
from vllm.v1.worker.gpu.sample.thinking_budget import (
    ThinkingBudgetState,
    apply_thinking_budget,
)
from vllm.v1.worker.gpu.spec_decode.tree.tree_verifier import TreeVerifier

_DEVICES = ["cuda"]


def _make_tree(tokens, parents, depths, device, num_nodes=None):
    return type(
        "TreeDraft",
        (),
        {
            "token_ids": torch.tensor(tokens, dtype=torch.long, device=device),
            "parents": torch.tensor(parents, dtype=torch.long, device=device),
            "depths": torch.tensor(depths, dtype=torch.long, device=device),
            "num_nodes": (
                torch.tensor([num_nodes], dtype=torch.int32, device=device)
                if num_nodes is not None
                else None
            ),
        },
    )()


def _chain_fixture(device):
    """One request, arena order [anchor, 11, 5, 3, 9] with depths [1, 1, 2, 2].

    The depth-2 nodes branch under different depth-1 parents, so the arena
    prefix (the old linear assumption) is NOT any node's accepted chain: node
    arena 2 (depth 2, token 3, parent arena 0/11) accepts [11, 3], and arena 3
    (depth 2, token 9, parent arena 1/5) accepts [5, 9].
    """
    trees = [
        _make_tree([11, 5, 3, 9], [-1, -1, 0, 1], [1, 1, 2, 2], device, num_nodes=4)
    ]
    chain_len, chain = _tree_chain_context(trees, [0, 5], 4, device)
    return trees, chain_len, chain, 16


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_chain_context_layout(device):
    """Per-row accepted-chain tables: node at depth d carries its ancestors
    plus itself in depth order; the anchor carries an empty chain."""
    _, chain_len, chain, _ = _chain_fixture(torch.device(device))
    assert chain_len.tolist() == [0, 1, 1, 2, 2], chain_len.tolist()
    assert chain[1, 0].item() == 11
    assert chain[2, 0].item() == 5
    assert chain[3, :2].tolist() == [11, 3]  # arena order is [11, 5, 3, 9]
    assert chain[4, :2].tolist() == [5, 9]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_penalties_tree_context_matches_reference(device):
    """The IS_TREE penalties kernel reproduces the accepted-chain context.

    A sodium reference counts the committed base plus the per-row chain tokens;
    the arena-order (buggy) path would count sibling tokens and differ.
    """
    dev = torch.device(device)
    _, chain_len, chain, vocab = _chain_fixture(dev)
    num_rows = 5
    logits = torch.randn(num_rows, vocab, device=dev) * 2
    expanded_idx = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    block = torch.tensor([7, 11, 5, 3, 9], dtype=torch.int32, device=dev)
    epos = torch.arange(num_rows, dtype=torch.int32, device=dev)
    rep = torch.full((1,), 1.2, dtype=torch.float32, device=dev)
    freq = torch.full((1,), 0.5, dtype=torch.float32, device=dev)
    pres = torch.full((1,), 0.3, dtype=torch.float32, device=dev)
    prompt_mask = torch.zeros(1, (vocab + 31) // 32, dtype=torch.int32, device=dev)
    out_counts = torch.zeros(1, vocab, dtype=torch.int32, device=dev)
    out_counts[0, 7] = 1
    out_counts[0, 11] = 2

    out = logits.clone()
    apply_penalties(
        out,
        expanded_idx,
        block,
        epos,
        rep,
        freq,
        pres,
        prompt_mask,
        out_counts,
        chain_prefix_len=chain_len,
        chain_prefix=chain,
    )

    def ref(row, chain_toks):
        counts = out_counts[0].clone()
        for t in chain_toks:
            counts[t] += 1
        toks = counts > 0
        scale = torch.where(toks, rep, torch.ones_like(logits[row]))
        res = logits[row] * torch.where(logits[row] > 0, 1.0 / scale, scale)
        return res - freq * counts - pres * toks

    expected = torch.stack(
        [ref(0, []), ref(1, [11]), ref(2, [5]), ref(3, [11, 3]), ref(4, [5, 9])]
    )
    assert (out - expected).abs().max().item() < 1e-6
    # The arena-order path (no chain passed) stays the un-fixed behavior.
    out_arena = logits.clone()
    apply_penalties(
        out_arena,
        expanded_idx,
        block,
        epos,
        rep,
        freq,
        pres,
        prompt_mask,
        out_counts,
    )
    assert not torch.allclose(out_arena, expected, atol=1e-6), (
        "the (buggy) arena-order path must differ from the accepted-chain oracle"
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bad_words_tree_context_matches_reference(device):
    """The IS_TREE bad-words kernel reads the accepted-chain context, not the
    arena-order block, for the within-step part of the emitted prefix."""
    dev = torch.device(device)
    _, chain_len, chain, vocab = _chain_fixture(dev)
    num_rows = 5
    all_token_ids = torch.zeros(2, 16, dtype=torch.int32, device=dev)
    all_token_ids[0, 0], all_token_ids[0, 1] = 12, 1
    all_token_ids[0, 2], all_token_ids[0, 3] = 7, 11
    prompt_len = torch.tensor([2], dtype=torch.int32, device=dev)
    total_len = torch.tensor([4], dtype=torch.int32, device=dev)
    block = torch.tensor([7, 11, 5, 3, 9], dtype=torch.int32, device=dev)
    expanded_idx = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    epos = torch.arange(num_rows, dtype=torch.int32, device=dev)
    output_tokens = all_token_ids[0, 2:4].tolist()
    logits = torch.randn(num_rows, vocab, device=dev) * 2

    def ref_mask(row, bad_word):
        emitted = output_tokens + [
            int(chain[row, k]) for k in range(int(chain_len[row]))
        ]
        if (
            len(emitted) >= len(bad_word) - 1
            and emitted[len(emitted) - (len(bad_word) - 1) :] == bad_word[:-1]
        ):
            return bad_word[-1]
        return None

    for bad_word in ([11, 3], [11, 11], [5, 9], [11, 5], [9]):
        bw = torch.tensor([bad_word + [0]], dtype=torch.int32, device=dev)
        off = torch.tensor(
            [[0, len(bad_word), len(bad_word)]], dtype=torch.int32, device=dev
        )
        num_bw = torch.tensor([1], dtype=torch.int32, device=dev)
        out = logits.clone()
        apply_bad_words(
            out,
            expanded_idx,
            bw,
            off,
            num_bw,
            all_token_ids,
            prompt_len,
            total_len,
            block,
            epos,
            1,
            chain_prefix_len=chain_len,
            chain_prefix=chain,
        )
        exp = {
            r: bad_word[-1]
            for r in range(num_rows)
            if ref_mask(r, bad_word) is not None
        }
        got = {}
        for r, t in (out == -float("inf")).nonzero().tolist():
            got[r] = t
        assert got == exp, (bad_word, got, exp)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_thinking_budget_tree_context_matches_reference(device):
    """The IS_TREE thinking-budget kernel scans the accepted-chain context for
    its markers and forces the same tokens a chain decoder would."""
    dev = torch.device(device)
    _, chain_len, chain, vocab = _chain_fixture(dev)
    num_rows = 5

    class Cfg:
        def __init__(self, start, end, natural):
            self.reasoning_start_token_ids = start
            self.reasoning_end_token_ids = end
            self.natural_reasoning_end_token_ids = natural

    class ReqStates:
        device = dev
        max_num_reqs = 1
        vocab_size = vocab

    start_ids = [2, 3]
    end_ids = [13, 14]
    natural_end_ids = [2, 5]
    st = ThinkingBudgetState(ReqStates(), Cfg(start_ids, end_ids, natural_end_ids))
    assert st.enabled
    st.add_request(0, type("SP", (), {"thinking_token_budget": 0})())
    st.apply_staged_writes()

    all_t2 = torch.zeros(1, 16, dtype=torch.int32, device=dev)
    all_t2[0, 0], all_t2[0, 1] = 12, 1
    all_t2[0, 2], all_t2[0, 3] = 2, 3
    total_len2 = torch.tensor([4], dtype=torch.int32, device=dev)
    budget = torch.tensor([0], dtype=torch.int32, device=dev)
    cached_start = torch.tensor([-1], dtype=torch.int32, device=dev)
    cached_end = torch.tensor([-1], dtype=torch.int32, device=dev)
    scan_pos = torch.tensor([0], dtype=torch.int32, device=dev)
    expanded_idx = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    epos = torch.arange(num_rows, dtype=torch.int32, device=dev)
    block = torch.tensor([7, 11, 5, 3, 9], dtype=torch.int32, device=dev)
    logits = torch.randn(num_rows, vocab, device=dev) * 2

    out = logits.clone()
    apply_thinking_budget(
        out,
        torch.tensor([0], dtype=torch.int32, device=dev),
        expanded_idx,
        budget,
        all_t2,
        total_len2,
        block,
        epos,
        cached_start,
        cached_end,
        scan_pos,
        torch.tensor(start_ids, dtype=torch.int32, device=dev),
        torch.tensor(natural_end_ids, dtype=torch.int32, device=dev),
        torch.tensor(end_ids, dtype=torch.int32, device=dev),
        chain_prefix_len=chain_len,
        chain_prefix=chain,
    )

    def eff_token(row, pos, total_len):
        if pos < total_len:
            return int(all_t2[0, pos])
        return int(chain[row, pos - total_len])

    def ref_force(row):
        total_len = 4
        eff_len = total_len + int(chain_len[row])
        last_start, last_end = -1, -1
        for i in range(0, total_len - len(start_ids) + 1):
            if [int(all_t2[0, i + j]) for j in range(len(start_ids))] == start_ids:
                last_start = i
        for i in range(0, total_len - len(natural_end_ids) + 1):
            if [
                int(all_t2[0, i + j]) for j in range(len(natural_end_ids))
            ] == natural_end_ids:
                last_end = i
        start_lo = total_len - len(start_ids) + 1
        for i in range(max(0, start_lo), eff_len - len(start_ids) + 1):
            if [eff_token(row, i + j, total_len) for j in range(len(start_ids))] == (
                start_ids
            ):
                last_start = i
        end_lo = total_len - len(natural_end_ids) + 1
        for i in range(max(0, end_lo), eff_len - len(natural_end_ids) + 1):
            if [
                eff_token(row, i + j, total_len) for j in range(len(natural_end_ids))
            ] == natural_end_ids:
                last_end = i
        if last_start < 0 or last_start <= last_end:
            return None
        reasoning_start = last_start + len(start_ids)
        if eff_len - reasoning_start < 0:
            return None
        end_prefix_len = 0
        for prefix_len in range(1, len(end_ids)):
            if (
                prefix_len <= eff_len
                and [
                    eff_token(row, eff_len - prefix_len + j, total_len)
                    for j in range(prefix_len)
                ]
                == end_ids[:prefix_len]
            ):
                end_prefix_len = prefix_len
        return end_ids[end_prefix_len]

    exp = {}
    for r in range(num_rows):
        f = ref_force(r)
        if f is not None:
            exp[r] = f
    got = {}
    for r, t in (out == 1.0e9).nonzero().tolist():
        got[r] = t
    assert got == exp, (got, exp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_serving_draws_seeded_per_position():
    """The serving verify draws every row with the ordinary sampler's
    Gumbel-max keyed by (request seed, row position): reproducible for a fixed
    seed, and a later step (anchor further along) draws a fresh stream rather
    than replaying the previous one."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    device = torch.device("cuda")
    R, B, V = 3, 6, 48
    rows = 1 + B
    parents = torch.stack(
        [torch.arange(-1, B - 1, dtype=torch.long, device=device)] * R
    )
    token_ids = torch.randint(0, V, (R, B), device=device)
    num_nodes = torch.full((R,), B, dtype=torch.long, device=device)
    seeds = torch.tensor([9000, 9001, 9002], dtype=torch.long, device=device)
    temperature = torch.ones(R, dtype=torch.float32, device=device)
    # Near-uniform rows, so any repeated stream would show.
    logits = torch.randn(R * rows, V, device=device) * 0.15
    exp_map = torch.arange(R * rows, device=device).div(rows, rounding_mode="floor")
    exp_map = exp_map.to(torch.int32)
    depth = torch.arange(rows, device=device)

    def draw(anchor_pos: int) -> torch.Tensor:
        pos = (anchor_pos + depth).repeat(R)
        tokens = gumbel_sample(
            logits, exp_map, temperature, seeds, pos, False, False
        ).view(R, rows)
        tv = TreeVerifier(R, B, V, device)
        sampled, num_sampled = tv.walk(tokens, token_ids, parents, num_nodes)
        return sampled[:, 0].clone()

    assert torch.equal(draw(10), draw(10))  # reproducible
    firsts = [draw(10 + 7 * k).tolist() for k in range(6)]
    assert any(len({row[i] for row in firsts}) >= 2 for i in range(R)), firsts

    # The reference-oracle verifier's seeded path stays reproducible.
    target = logits.view(R, rows, V)
    s1 = seeds * 31 + 5
    tv = TreeVerifier(R, B, V, device)
    s_a, ns_a = tv(target, token_ids, parents, num_nodes, greedy=False, seeds=s1)
    tv_b = TreeVerifier(R, B, V, device)
    s_b, ns_b = tv_b(target, token_ids, parents, num_nodes, greedy=False, seeds=s1)
    assert torch.equal(s_a, s_b) and torch.equal(ns_a, ns_b)
    raw = seeds

    # Mixed batch: the seeded lane is reproducible; -1 lanes draw from the
    # global RNG (advancing between calls) and never crash.
    mixed = torch.tensor([-1, 777, -1], dtype=torch.long, device=device)
    tv_c = TreeVerifier(R, B, V, device)
    s_c, _ = tv_c(target, token_ids, parents, num_nodes, greedy=False, seeds=mixed)
    tv_d = TreeVerifier(R, B, V, device)
    s_d, _ = tv_d(target, token_ids, parents, num_nodes, greedy=False, seeds=mixed)
    assert 0 <= int(s_c[0, 0].item()) < V  # global-RNG lane samples a token
    assert torch.equal(s_c[1], s_d[1])  # seeded lane deterministic

    # Reference parity of the plain (no -1) path is unchanged.
    tv_e = TreeVerifier(R, B, V, device)
    s_e, ns_e = tv_e(target, token_ids, parents, num_nodes, greedy=False, seeds=raw)
    for r in range(R):
        out, nacc = tree_verify_reference(
            target[r], token_ids[r], parents[r], greedy=False, seed=int(raw[r].item())
        )
        assert int(ns_e[r].item()) - 1 == nacc
        assert s_e[r, : int(ns_e[r].item())].tolist() == out
