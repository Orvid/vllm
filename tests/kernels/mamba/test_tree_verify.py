# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-verify recurrent kernels against a per-path sequential reference.

Each row of a tree verify batch must equal what the ordinary chain recurrence
produces on that row's ancestor path, starting from the request's committed
state, and the committed state must be left untouched.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.tree_verify import (
    commit_gated_delta_rule_paths,
    tree_causal_conv1d,
    tree_gated_delta_rule,
    tree_gated_delta_rule_chunk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")


def _batch():
    """Request 0: anchor + 6 tree nodes (two branches of different depth).
    Request 1: a 3-row chain. Slots 3 and 5 hold their committed states."""
    # Rows 0..6: anchor, a, b, c, d, e, f with a,b under the anchor, c under a,
    # d under c, e under b, f under d.
    parent = [-1, 0, 0, 1, 3, 2, 4, -1, 7, 8]
    qsl = torch.tensor([0, 7, 10], dtype=torch.int32, device=DEVICE)
    slots = torch.tensor([3, 5], dtype=torch.int32, device=DEVICE)
    return torch.tensor(parent, dtype=torch.int32, device=DEVICE), qsl, slots


def _path(parent: list[int], row: int, bos: int) -> list[int]:
    path = []
    while row >= bos:
        path.append(row)
        row = parent[row]
    return path[::-1]


def _request_of(qsl: list[int], row: int) -> int:
    return next(n for n in range(len(qsl) - 1) if qsl[n] <= row < qsl[n + 1])


def test_tree_conv1d_matches_per_path_conv():
    torch.manual_seed(0)
    parent, qsl, slots = _batch()
    dim, width = 96, 4
    x = torch.randn(10, dim, device=DEVICE)
    state = torch.randn(8, dim, width - 1, device=DEVICE)
    state_before = state.clone()
    weight = torch.randn(dim, width, device=DEVICE)
    bias = torch.randn(dim, device=DEVICE)
    out = tree_causal_conv1d(x, state, weight, bias, "silu", slots, qsl, 2, parent)
    torch.testing.assert_close(state, state_before)
    p, q = parent.tolist(), qsl.tolist()
    for row in range(10):
        n = _request_of(q, row)
        seq = torch.cat(
            [state[slots[n]].T, x[_path(p, row, q[n])]], dim=0
        )  # committed inputs then the path, oldest first
        window = seq[-width:]
        ref = F.silu((window.T * weight).sum(-1) + bias)
        torch.testing.assert_close(out[row], ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunked", [False, True])
def test_tree_gated_delta_rule_matches_per_path_recurrence(chunked: bool):
    """Both the per-path replay and the ancestor-masked chunk kernel."""
    torch.manual_seed(0)
    parent, qsl, slots = _batch()
    H, HV, K, V = 2, 4, 32, 64
    T = 10
    q = F.normalize(torch.randn(T, H, K, device=DEVICE), dim=-1)
    k = F.normalize(torch.randn(T, H, K, device=DEVICE), dim=-1)
    v = torch.randn(T, HV, V, device=DEVICE)
    g = -torch.rand(T, HV, device=DEVICE)
    beta = torch.rand(T, HV, device=DEVICE)
    state = torch.randn(8, HV, V, K, device=DEVICE)
    state_before = state.clone()
    if chunked:
        out = tree_gated_delta_rule_chunk(
            q, k, v, g, beta, state, slots, qsl, 2, parent, max_depth=8, max_rows=7
        )
    else:
        out = tree_gated_delta_rule(
            q, k, v, g, beta, state, slots, qsl, 2, parent, max_depth=8
        )
    torch.testing.assert_close(state, state_before)
    p, ql = parent.tolist(), qsl.tolist()
    scale = K**-0.5
    for row in range(T):
        n = _request_of(ql, row)
        for hv in range(HV):
            h = hv // (HV // H)
            S = state[slots[n], hv].clone()  # [V, K]
            for r in _path(p, row, ql[n]):
                S = S * torch.exp(g[r, hv])
                u = beta[r, hv] * (v[r, hv] - S @ k[r, h])
                S = S + torch.outer(u, k[r, h])
            ref = S @ (q[row, h] * scale)
            # The chunk kernel's dots run in TF32.
            tol = 2e-3 if chunked else 1e-4
            torch.testing.assert_close(out[row, hv], ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("block_size", [4, 1 << 20])
def test_commit_stores_block_end_and_running_states(block_size: int):
    """Committing a path replays it from the state the verify read and stores
    the state (and conv window) after the last row of every state block it
    touches -- block-end states for prefix caching, then the running state --
    leaving every other slot alone. With one block per sequence (no prefix
    caching) only the running state is stored."""
    torch.manual_seed(0)
    H, HV, K, V, width = 2, 4, 32, 32, 4
    conv_dim = 2 * H * K + HV * V
    # Request 0 commits 3 rows at positions 6..8 (crossing into block 2 when
    # blocks are 4 long), request 1 none, request 2 two rows at 9..10.
    cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([6, 7, 8, 9, 10], dtype=torch.int64, device=DEVICE)
    num_blocks = 4
    block_table = torch.arange(
        1, 1 + 3 * num_blocks, dtype=torch.int32, device=DEVICE
    ).view(3, num_blocks)
    if block_size > 16:
        block_table[:, 1:] = 0
    # The verify read each request's state from the block of position C - 1.
    init = torch.tensor(
        [int(block_table[n, (c - 1) // block_size]) for n, c in enumerate([6, 9, 9])],
        dtype=torch.int32,
        device=DEVICE,
    )
    rows = torch.randn(5, conv_dim + 2 * HV, device=DEVICE)
    num_slots = 1 + 3 * num_blocks
    conv_state = torch.randn(num_slots, conv_dim, width - 1, device=DEVICE)
    ssm_state = torch.randn(num_slots, HV, V, K, device=DEVICE)
    weight = torch.randn(conv_dim, width, device=DEVICE)
    bias = torch.randn(conv_dim, device=DEVICE)
    A_log = torch.randn(HV, device=DEVICE)
    dt_bias = torch.randn(HV, device=DEVICE)
    conv_before, ssm_before = conv_state.clone(), ssm_state.clone()

    commit_gated_delta_rule_paths(
        rows,
        positions,
        cu,
        3,
        init,
        block_table,
        block_size,
        conv_state,
        weight,
        bias,
        "silu",
        ssm_state,
        A_log,
        dt_bias,
        H,
        K,
        V,
    )

    conv_ref, ssm_ref = conv_before.clone(), ssm_before.clone()
    for n, (lo, hi) in enumerate([(0, 3), (3, 3), (3, 5)]):
        if lo == hi:
            continue
        seq = torch.cat([conv_before[init[n]].T, rows[lo:hi, :conv_dim]], dim=0)
        S = ssm_before[init[n]].clone()
        for t in range(hi - lo):
            window = seq[t : t + width]
            y = F.silu((window.T * weight).sum(-1) + bias)
            q, k, v = y.split([H * K, H * K, HV * V])
            k = F.normalize(k.view(H, K), dim=-1)
            v = v.view(HV, V)
            b = rows[lo + t, conv_dim : conv_dim + HV]
            a = rows[lo + t, conv_dim + HV :]
            g = -torch.exp(A_log) * F.softplus(a + dt_bias)
            beta = torch.sigmoid(b)
            for hv in range(HV):
                h = hv // (HV // H)
                S[hv] = S[hv] * torch.exp(g[hv])
                u = beta[hv] * (v[hv] - S[hv] @ k[h])
                S[hv] = S[hv] + torch.outer(u, k[h])
            pos = int(positions[lo + t])
            last = t == hi - lo - 1
            if last or (pos + 1) // block_size != pos // block_size:
                slot = int(block_table[n, pos // block_size])
                ssm_ref[slot] = S
                conv_ref[slot] = seq[t + 1 : t + width].T
    torch.testing.assert_close(conv_state, conv_ref)
    torch.testing.assert_close(ssm_state, ssm_ref, rtol=1e-3, atol=1e-3)
