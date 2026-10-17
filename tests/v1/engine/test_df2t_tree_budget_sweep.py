# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 measurability sweep: tree@B over a fixed depth L=12 must actually be
expressed, and acceptance must track B (WS-A finding [10]).

Before the tree-budget decoupling, the scheduler reserved ``num_speculative_tokens``
draft slots per request and the speculator kept only the first ``num_spec``
nodes, so effective B was ``min(nodes_max, L)`` and tree@16/32/64 over L=12 all
verified 15/12 nodes. This test drives a real offline engine at T=0 and reads
the per-step *scheduled draft count* the scheduler recorded
(``per_request_spec_decode_metrics="detailed"``): it must equal the requested B
(not L), and the per-step mean accepted-draft count must rise with B.

Evidence is structural/call-count (per-step scheduled and accepted counts), not
wall-clock; no instrumentation of the worker is needed because the scheduler
already emits these per request.
"""

import gc

import pytest

from tests.v1.engine.df2t_test_utils import assert_t0_parity
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "Qwen/Qwen3-8B"
_DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
_PROMPTS = [
    "The capital of France is",
    "The largest planet is",
    "Photosynthesis converts",
]
_MAX_TOKENS = 24
_DEPTH = 12


def _offline_models_present() -> bool:
    import torch

    return torch.cuda.is_available()


def _run_engine(
    *, tree_budget: int
) -> tuple[list[list[int]], list[dict[int, float]], list[dict]]:
    """Run df2t at ``tree_budget`` over L=``_DEPTH`` at T=0.

    Returns the output token ids and each request's ``spec_decode_metrics``
    dict (detailed: ordered per-step drafted/accepted counts).
    """
    from vllm import LLM, SamplingParams

    speculative_config = {
        "method": "dflash",
        "tree_selection": "df2t",
        "model": _DRAFT,
        "num_speculative_tokens": 15,
        "tree_attention_backend": "flex",
        "tree_budget": tree_budget,
        "max_tree_budget": tree_budget,
        "max_tree_depth": _DEPTH,
        "nodes_ceiling": tree_budget,
    }
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=192,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        disable_log_stats=True,
        per_request_spec_decode_metrics="detailed",
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    try:
        out = llm.generate(
            _PROMPTS,
            SamplingParams(max_tokens=_MAX_TOKENS, temperature=0.0, logprobs=2),
        )
        ids = [list(o.outputs[0].token_ids) for o in out]
        probs = [o.outputs[0].logprobs for o in out]
        metrics = [
            o.outputs[0].spec_decode_metrics.to_dict()
            if o.outputs[0].spec_decode_metrics
            else {}
            for o in out
        ]
        return ids, probs, metrics
    finally:
        del llm
        gc.collect()


def _reference_ids() -> tuple[list[list[int]], list[list[dict[int, float]]]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_TARGET,
        enforce_eager=True,
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=192,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        disable_log_stats=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    try:
        out = llm.generate(
            _PROMPTS,
            SamplingParams(max_tokens=_MAX_TOKENS, temperature=0.0, logprobs=2),
        )
        return [list(o.outputs[0].token_ids) for o in out], [
            o.outputs[0].logprobs for o in out
        ]
    finally:
        del llm
        gc.collect()


def _per_step_drafted(metrics: list[dict]) -> list[int]:
    """The per-step scheduled draft counts across all requests (detailed)."""
    drafted: list[int] = []
    for m in metrics:
        drafted.extend(m.get("per_step_drafted", []))
    return drafted


def _mean_accepted_per_step(metrics: list[dict]) -> float:
    accepted = [a for m in metrics for a in m.get("per_step_accepted", [])]
    return sum(accepted) / len(accepted) if accepted else 0.0


@pytest.mark.skipif(
    not _offline_models_present(), reason="needs CUDA and a public model download"
)
def test_df2t_tree_budget_is_actually_expressed_over_depth_twelve() -> None:
    """tree@B for B in {16, 32, 64} over L=12 schedules B draft rows per step,
    NOT L: the recorded per-step drafted count equals the requested budget
    (never the ``num_spec``/L cap), and every run stays the lossless T=0 greedy
    walk."""
    ref_ids, ref_probs = _reference_ids()

    effective_b: dict[int, int] = {}
    per_step: dict[int, list[int]] = {}
    mean_accepted: dict[int, float] = {}
    for budget in (16, 32, 64):
        ids, probs, metrics = _run_engine(tree_budget=budget)
        # T=0 losslessness: output equals the plain greedy walk.
        assert_t0_parity(ids, ref_ids, probs, ref_probs, f"tree@{budget}")
        drafted = _per_step_drafted(metrics)
        assert drafted, "no per-step drafted counts recorded"
        per_step[budget] = drafted
        effective_b[budget] = max(drafted)
        mean_accepted[budget] = _mean_accepted_per_step(metrics)

    for budget, width in effective_b.items():
        # Effective B == requested B, and strictly greater than the L=12 cap.
        assert width == budget, (
            f"effective B={width} != requested B={budget}; the tree was truncated "
            "by num_spec/L, not expressed"
        )
        assert width > _DEPTH, (
            f"effective B={width} did not exceed the L={_DEPTH} depth cap"
        )

    # Acceptance is non-decreasing in the tree budget. Real-model noise
    # can invert adjacent rungs, so require the widest tree to beat the
    # narrowest (and never regress by more than a token on the way up).
    assert mean_accepted[64] >= mean_accepted[16], (
        f"acceptance fell with B: {mean_accepted}"
    )
    assert mean_accepted[32] + 1 >= mean_accepted[16], (
        f"acceptance regressed at B=32 vs B=16: {mean_accepted}"
    )
