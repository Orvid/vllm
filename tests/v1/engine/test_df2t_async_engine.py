# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end async df2t engine test: concurrent multi-batch tree carry.

With async scheduling the engine runs ``step_with_batch_queue`` (batch queue
depth = ``max_concurrent_batches`` = 2), so consecutive decode steps of
different requests overlap in the queue and the df2t tree is carried from the
previous worker batch via ``_graft_df2t_async_drafts``. This test drives two
requests through that genuinely concurrent path and asserts the T=0 output is
the plain target's greedy continuation for every request. A dropped or
misrouted tree would skip compaction, leave accepted KV scattered, and the next
step would attend garbage and diverge.

Exact T=0 token parity is only a valid oracle in fp32: in bf16 the target's
own greedy walk flips on one-ulp logit ties with batch shape (even without
speculative decoding), so both runs use fp32 on the flex vehicle.
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


def _run_engine(
    *, async_scheduling: bool, method: str | None, max_tokens: int
) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    speculative_config = None
    if method == "df2t":
        speculative_config = {
            "method": method,
            "model": _DRAFT,
            "num_speculative_tokens": 15,
            "tree_attention_backend": "flex",
        }
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=async_scheduling,
        max_num_batched_tokens=16,
        max_model_len=192,
        dtype="float32",
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
        disable_log_stats=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    out = llm.generate(_PROMPTS, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    ids = [list(o.outputs[0].token_ids) for o in out]
    del llm
    gc.collect()
    return ids


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_async_multi_batch_t0_matches_greedy_reference() -> None:
    """Two requests through the concurrent async batch queue produce, at T=0,
    exactly the greedy target continuation for BOTH requests — the tree must
    be carried to the right request across the overlapping batches, and the
    accepted-KV compaction must leave the next step attending the relocated
    canonical slots."""
    reference = _run_engine(async_scheduling=False, method=None, max_tokens=16)
    async_ids = _run_engine(async_scheduling=True, method="df2t", max_tokens=16)

    assert len(async_ids) == len(_PROMPTS) == len(reference)
    for i, expected in enumerate(reference):
        assert async_ids[i] == expected, (
            f"request {i}: async df2t T=0 diverges from the greedy reference"
        )
