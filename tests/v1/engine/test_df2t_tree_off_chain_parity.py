# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-off df2t parity: ``method="df2t"`` + ``enable_tree=False`` must serve
the DFlash2 greedy CHAIN from the same drafter (the "tree disabled reproduces
DFlash2" invariant), token-identical to ``method="dflash"`` at T=0.

Exact T=0 token parity is only a valid oracle in fp32 (in bf16 the target's own
greedy walk flips on one-ulp logit ties with batch shape), so both runs use
fp32 on the flex vehicle, as the other df2t engine suites do. If a tree-off
df2t config ever routed to the tree machinery (verify/compaction/tree mask) the
layout or reservation would diverge and the outputs would not be bit-identical
to the dflash chain.
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


def _run_engine(method: str, *, enable_tree: bool | None = None) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    if method == "none":
        speculative_config = None
    else:
        speculative_config = {
            "method": method,
            "model": _DRAFT,
            "num_speculative_tokens": 15,
            "tree_attention_backend": "flex",
            "tree_budget": 32,
            "max_tree_budget": 32,
            "nodes_ceiling": 32,
        }
        if enable_tree is not None:
            speculative_config["enable_tree"] = enable_tree
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_num_batched_tokens=192,
        max_model_len=192,
        dtype="float32",
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
        disable_log_stats=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    out = llm.generate(_PROMPTS, SamplingParams(max_tokens=16, temperature=0.0))
    ids = [list(o.outputs[0].token_ids) for o in out]
    del llm
    gc.collect()
    return ids


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_tree_off_matches_dflash_and_greedy() -> None:
    """``enable_tree=False`` df2t is token-identical to ``method="dflash"`` and
    to the plain greedy target continuation at T=0 — the chain path, nothing
    tree-shaped. A tree leak (verifier/compaction/tree-mask) would break it."""
    dflash = _run_engine("dflash")
    tree_off = _run_engine("df2t", enable_tree=False)
    plain = _run_engine("none")

    assert len(dflash) == len(tree_off) == len(plain) == len(_PROMPTS)
    for i in range(len(_PROMPTS)):
        assert tree_off[i] == dflash[i], (
            f"request {i}: df2t(enable_tree=False) != dflash chain at T=0"
        )
        assert tree_off[i] == plain[i], (
            f"request {i}: df2t(enable_tree=False) != greedy reference at T=0"
        )
