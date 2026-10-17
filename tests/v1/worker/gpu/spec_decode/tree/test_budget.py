# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the tree verify budget policy (``s`` scale + per-request shape).

Covers the pure ``resolve_nodes_max`` / ``shape`` operators and the
``BudgetPolicy`` bundle. All inputs are scalars/int, so the tests run fully
offline on CPU and pin exact outputs.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.spec_decode.tree.budget import (
    BudgetPolicy,
    resolve_nodes_max,
    shape,
)


def test_resolve_nodes_max_scales_policy():
    assert resolve_nodes_max(1.0, 16, 64, 15) == 16
    assert resolve_nodes_max(2.0, 16, 64, 15) == 32
    assert resolve_nodes_max(4.0, 16, 64, 15) == 64


def test_resolve_nodes_max_clamps_at_ceiling():
    assert resolve_nodes_max(8.0, 16, 32, 15) == 32


def test_resolve_nodes_max_clamps_at_floor():
    assert resolve_nodes_max(0.5, 16, 64, 15) == 15
    assert resolve_nodes_max(0.25, 32, 64, 15) == 15


def test_resolve_nodes_max_s_one_matches_policy():
    for policy in (1, 16, 32, 64):
        assert resolve_nodes_max(1.0, policy, 64, 1) == policy


def test_shape_static_full_depth():
    assert shape(64, None, 15) == (15, 64)


def test_shape_static_boundary_full_depth():
    assert shape(15, None, 15) == (15, 15)


def test_shape_static_degrades_to_chain():
    assert shape(8, None, 15) == (8, 8)


def test_shape_head_path_monotone_and_bounded():
    c_ts = [-1.0, 0.0, 1.0, 2.0, 3.0]
    shaped = [shape(64, c, 15, floor=1, ceiling=64) for c in c_ts]
    bs = [b for _, b in shaped]
    assert bs == sorted(bs)
    assert all(1 <= b <= 64 for b in bs)
    assert all(depth == 15 for depth, _ in shaped)


def test_shape_head_path_pinned_buckets():
    assert shape(64, -1.0, 15)[1] == 16
    assert shape(64, 0.0, 15)[1] == 32
    assert shape(64, 1.0, 15)[1] == 32
    assert shape(64, 3.0, 15)[1] == 64


def test_shape_head_path_degrades_below_max_depth():
    assert shape(64, -10.0, 15) == (4, 4)


def test_shape_head_path_respects_floor_and_ceiling():
    assert shape(64, -10.0, 15, floor=8)[1] == 8
    assert shape(64, 10.0, 15, ceiling=32)[1] == 32


def test_budget_policy_resolves_and_shapes():
    policy = BudgetPolicy(policy=16, ceiling=64, max_depth=15, s=1.0, floor=1)
    assert policy.nodes_max == 16
    assert policy.resolve_nodes_max() == 16
    assert policy.shape(None) == (15, 16)


def test_budget_policy_scaled_resolve():
    policy = BudgetPolicy(policy=16, ceiling=64, max_depth=15, s=2.0, floor=1)
    assert policy.resolve_nodes_max() == 32


def test_budget_policy_from_config():
    d = SimpleNamespace(
        tree_s=2.0,
        tree_budget=16,
        nodes_ceiling=64,
        max_tree_depth=15,
    )
    policy = BudgetPolicy.from_config(d)
    assert policy.s == 2.0
    assert policy.policy == 16
    assert policy.ceiling == 64
    assert policy.max_depth == 15
    assert policy.resolve_nodes_max() == 32
    assert policy.shape(None) == (15, 32)


def test_budget_policy_static_matches_get_num_draft_tokens():
    d = SimpleNamespace(
        tree_s=1.0,
        tree_budget=16,
        nodes_ceiling=64,
        max_tree_depth=15,
    )
    policy = BudgetPolicy.from_config(d)
    assert policy.resolve_nodes_max() == 16


def test_budget_policy_floor_must_not_exceed_ceiling():
    with pytest.raises(ValueError, match="nodes_floor must not exceed"):
        BudgetPolicy(policy=16, ceiling=4, max_depth=15, floor=8)


def test_resolve_nodes_max_agrees_with_config_under_defaults():
    """The operative nodes budget (BudgetPolicy) equals the config-level
    ``get_num_draft_tokens`` under the default knobs, so both consumers see
    the same per-deployment node budget."""
    cfg = SimpleNamespace(
        method="dflash",
        tree_selection="df2t",
        tree_s=1.0,
        tree_budget=16,
        nodes_ceiling=64,
        max_tree_depth=15,
    )
    policy = BudgetPolicy.from_config(cfg)
    assert policy.resolve_nodes_max() == int(round(cfg.tree_s * cfg.tree_budget))
    assert policy.resolve_nodes_max() == 16
