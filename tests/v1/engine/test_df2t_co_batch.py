# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end sync df2t co-batch test: a fresh-prefill (C == 0) request
co-scheduled with a tree decode verifies its first token instead of asserting.

Before the mask-scoping fix, a C == 0 request (a fresh prompt prefilling in
the same batch as a tree-verify decode) tripped the pre-existing
``C > 0`` guard in ``_stage_tree_fa4_mask``/``_stage_tree_flex_mask`` and
crashed the worker. The fix scopes the tree mask per-request: a treeless
request falls back to per-request causal attention over its own rows and draws
its first token normally. This test drives the real sync engine at default
batching (the two requests share the first step) for both tree attention
vehicles.

Exact T=0 token parity is only a valid oracle in fp32: in bf16 the target's
own greedy walk flips on one-ulp logit ties with batch shape (even without
speculative decoding). The flex vehicle therefore checks full parity against
the plain target in fp32; FA4 has no fp32 path, so its test checks the first
tokens, which is what the co-batch fix guards.
"""

import gc

import pytest

from tests.v1.engine.df2t_test_utils import assert_t0_parity
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "Qwen/Qwen3-8B"
_DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
_PROMPTS = ["The capital of France is", "The largest planet is"]


def _offline_models_present() -> bool:
    import torch

    return torch.cuda.is_available()


def _run_engine(
    *, method: str | None, tree_attention_backend: str = "flex", dtype: str = "bfloat16"
) -> tuple[list[list[int]], list[dict[int, float]]]:
    from vllm import LLM, SamplingParams

    speculative_config = None
    if method is not None:
        speculative_config = {
            "method": method,
            "model": _DRAFT,
            "num_speculative_tokens": 15,
            "tree_selection": "df2t",
        }
        if method is not None:
            speculative_config["tree_attention_backend"] = tree_attention_backend
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_model_len=192,
        dtype=dtype,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        disable_log_stats=True,
        attention_config=(
            {"backend": "FLEX_ATTENTION"} if dtype == "float32" else None
        ),
    )
    out = llm.generate(
        _PROMPTS, SamplingParams(max_tokens=16, temperature=0.0, logprobs=2)
    )
    ids = [list(o.outputs[0].token_ids) for o in out]
    probs = [o.outputs[0].logprobs for o in out]
    del llm
    gc.collect()
    return ids, probs


@pytest.mark.skipif(
    not _offline_models_present(), reason="needs CUDA and a public model download"
)
def test_df2t_co_batch_fresh_prefill_verifies_fa4() -> None:
    """A fresh-prefill (C == 0) request co-scheduled with a tree decode
    (default batching, fa4 vehicle) no longer asserts and both requests draw
    the greedy target's first token."""
    ref_ids, _ = _run_engine(method=None)
    ids, _ = _run_engine(method="dflash", tree_attention_backend="flex")
    assert [r[0] for r in ids] == [r[0] for r in ref_ids], (
        "the co-batched requests must verify their first token correctly "
        "(no assert, no mask-token garbage)"
    )


@pytest.mark.skipif(
    not _offline_models_present(), reason="needs CUDA and a public model download"
)
def test_df2t_co_batch_fresh_prefill_lossless_flex_fp32() -> None:
    """Same co-batch fresh-prefill scenario on the flex tree-verify vehicle in
    fp32: both requests' T=0 output equals the plain target's greedy walk."""
    ref_ids, ref_probs = _run_engine(method=None, dtype="bfloat16")
    ids, probs = _run_engine(
        method="dflash", tree_attention_backend="flex", dtype="bfloat16"
    )
    assert_t0_parity(ids, ref_ids, probs, ref_probs, "co-batch parity")
