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


def test_tree_gated_delta_rule_matches_per_path_recurrence():
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
            torch.testing.assert_close(out[row, hv], ref, rtol=1e-4, atol=1e-4)


def test_commit_advances_state_over_committed_rows():
    """Committing a path leaves the conv state holding the path's newest raw
    inputs and the recurrent state where the chain recurrence over the path
    ends; a request with nothing committed keeps both."""
    torch.manual_seed(0)
    H, HV, K, V, width = 2, 4, 32, 32, 4
    conv_dim = 2 * H * K + HV * V
    # Request 0 commits 3 rows, request 1 none, request 2 two rows.
    cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=DEVICE)
    slots = torch.tensor([2, 4, 6], dtype=torch.int32, device=DEVICE)
    rows = torch.randn(5, conv_dim + 2 * HV, device=DEVICE)
    conv_state = torch.randn(8, conv_dim, width - 1, device=DEVICE)
    ssm_state = torch.randn(8, HV, V, K, device=DEVICE)
    weight = torch.randn(conv_dim, width, device=DEVICE)
    bias = torch.randn(conv_dim, device=DEVICE)
    A_log = torch.randn(HV, device=DEVICE)
    dt_bias = torch.randn(HV, device=DEVICE)
    conv_before, ssm_before = conv_state.clone(), ssm_state.clone()

    commit_gated_delta_rule_paths(
        rows,
        cu,
        3,
        3,
        slots,
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

    torch.testing.assert_close(conv_state[4], conv_before[4])
    torch.testing.assert_close(ssm_state[4], ssm_before[4])
    for n, (lo, hi) in enumerate([(0, 3), (3, 3), (3, 5)]):
        if lo == hi:
            continue
        slot = slots[n]
        x = rows[lo:hi, :conv_dim]
        seq = torch.cat([conv_before[slot].T, x], dim=0)
        torch.testing.assert_close(conv_state[slot], seq[-(width - 1) :].T)
        S = ssm_before[slot].clone()
        for t in range(hi - lo):
            window = seq[t : t + width]
            y = F.silu((window.T * weight).sum(-1) + bias)
            q, k, v = y.split([H * K, H * K, HV * V])
            q = F.normalize(q.view(H, K), dim=-1) * K**-0.5
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
        torch.testing.assert_close(ssm_state[slot], S, rtol=1e-3, atol=1e-3)
