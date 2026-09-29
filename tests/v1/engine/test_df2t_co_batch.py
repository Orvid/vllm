# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end sync df2t co-batch test: a fresh-prefill (C == 0) request
co-scheduled with a tree decode verifies its first token instead of asserting.

Before the mask-scoping fix, a C == 0 request (a fresh prompt prefilling in
the same batch as a df2t tree-verify decode) tripped the pre-existing
``C > 0`` guard in ``_stage_tree_fa4_mask``/``_stage_tree_flex_mask`` and
crashed the worker. The fix scopes the tree mask per-request: a contextless
request has no tree and no context, so it falls back to per-request causal
attention over its own rows and draws its first token normally. This test
drives the real sync engine at default batching (the two requests share the
first step) for both tree attention vehicles and asserts the contextless
request's output is exactly the greedy target walk (the dflash chain) — the
correct first token, with no assert.
"""

import gc
import os

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "/mnt/e/dev/Qwen3-0.6B"
_DRAFT = "/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16"
_PROMPTS = ["The capital of France is", "The largest planet is"]


def _offline_models_present() -> bool:
    return os.path.isdir(_TARGET) and os.path.isdir(_DRAFT)


def _run_engine(*, tree_attention_backend: str, backend_mode: str) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    speculative_config = {
        "method": "df2t",
        "model": _DRAFT,
        "num_speculative_tokens": 15,
    }
    if tree_attention_backend == "flex":
        speculative_config["tree_attention_backend"] = "flex"
    if backend_mode == "dflash":
        speculative_config["method"] = "dflash"
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_model_len=192,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    out = llm.generate(_PROMPTS, SamplingParams(max_tokens=16, temperature=0.0))
    ids = [list(o.outputs[0].token_ids) for o in out]
    del llm
    gc.collect()
    return ids


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_co_batch_fresh_prefill_verifies_fa4() -> None:
    """A fresh-prefill (C == 0) request co-scheduled with a df2t tree decode
    (default batching, fa4 vehicle) no longer asserts and verifies its first
    token correctly: its output equals the greedy target walk (dflash chain)."""
    chain = _run_engine(tree_attention_backend="fa4", backend_mode="dflash")[1]
    ids = _run_engine(tree_attention_backend="fa4", backend_mode="df2t")
    assert ids[1] == chain, (
        "the contextless co-batched request must verify its first token "
        "correctly (no assert, no mask-token garbage)"
    )
    # The tree-decode request's own first token is also the greedy draw.
    chain0 = _run_engine(tree_attention_backend="fa4", backend_mode="dflash")[0]
    assert ids[0][0] == chain0[0]


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_co_batch_fresh_prefill_verifies_flex() -> None:
    """Same co-batch fresh-prefill scenario on the flex tree-verify vehicle:
    no C > 0 assert and the contextless request's first token is correct."""
    chain = _run_engine(tree_attention_backend="flex", backend_mode="dflash")[1]
    ids = _run_engine(tree_attention_backend="flex", backend_mode="df2t")
    assert ids[1] == chain, (
        "the contextless co-batched request must verify its first token "
        "correctly on the flex vehicle"
    )
