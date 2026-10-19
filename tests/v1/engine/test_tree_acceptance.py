# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tree drafter-context acceptance gate.

The tree drafter must condition its next draft on the accepted chain (the
positions the target actually accepted, relocated to canonical ``C + depth``
slots by the KV compaction), not on the arena-order verify rows. When that was
wrong the drafter proposed from garbage hidden states and acceptance
collapsed to the degenerate floor (~1-3 accepted tokens per 32 drafted, i.e.
``mean accepted / draft`` ~0.05). This test drives the real engine on the
offline Qwen3-0.6B + tree drafter, reads the spec-decode counters, and asserts
the acceptance length is materially above that floor, plus a T=0 losslessness
check (df2t output == the plain target's greedy walk) over the whole batch.

Exact T=0 token parity is checked on the flex vehicle: in bf16 the target's
own greedy walk flips on one-ulp logit ties with batch shape (even without
speculative decoding), so the parity check runs fp32 on the flex vehicle.
"""

import gc

import pytest

from tests.v1.engine.tree_test_utils import assert_t0_parity
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "Qwen/Qwen3-8B"
_DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
_NUM_SPEC = 15
_PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Photosynthesis is the process by which",
    "In 1969, the Apollo 11 mission landed",
    "The mitochondria is known as the powerhouse",
    "Water boils at 100 degrees",
]

# Per-request draft-count floor for the acceptance sample. Acceptance is what
# discriminates the buggy floor (mean accepted/draft ~0.05), not the draft
# count — the degenerate drafter actually produces *more* drafts per request
# because it accepts almost nothing. This is only a sample-size guard, kept low
# enough that it stays green in the high-acceptance regime (acceptance length
# ~5 -> ~12 drafts per 64-token request) while still requiring the run to have
# actually done repeated speculative steps.
_DRAFTS_PER_REQ_FLOOR = 6


def _offline_models_present() -> bool:
    import torch

    return torch.cuda.is_available()


def _run_engine(
    method: str | None, prompts: list[str], max_tokens: int
) -> tuple[list[list[int]], list[dict[int, float]], int, int, int]:
    from vllm import LLM, SamplingParams

    speculative_config = None
    if method is not None:
        speculative_config = {
            "method": method,
            "model": _DRAFT,
            "num_speculative_tokens": _NUM_SPEC,
            "tree_selection": "df2t",
            "tree_attention_backend": "flex",
        }
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=512,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        disable_log_stats=False,
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    outs = llm.generate(
        prompts, SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=2)
    )
    ids = [list(o.outputs[0].token_ids) for o in outs]
    probs = [o.outputs[0].logprobs for o in outs]
    num_drafts = num_draft_tokens = num_accepted = 0
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted += metric.value
    del llm
    gc.collect()
    return ids, probs, num_drafts, num_draft_tokens, num_accepted


@pytest.mark.skipif(
    not _offline_models_present(), reason="needs CUDA and a public model download"
)
def test_df2t_acceptance_above_degenerate_floor_and_lossless() -> None:
    """df2t must accept materially more than the buggy 1-3/32 floor.

    Before the drafter-context fix the tree verify engine measured ~0.05 accepted
    tokens per draft step (acceptance length ~1.05) — no speculative speedup.
    The gate asserts a mean accepted-per-draft well above that floor (measured
    ~1.8-2.5 after the fix, comparable to or above the dflash chain's own
    ~1.9), and that the batched df2t T=0 output matches the plain target's
    greedy walk (output distribution must be unchanged).

    The draft-count child guard is a sample-size check only: the degenerate
    floor produces *more* drafts per request (it accepts almost nothing), so
    the acceptance assert is what discriminates it. The guard is kept low
    enough to stay green in the high-acceptance regime, where each draft spans
    more tokens and the per-request draft count shrinks.
    """
    ids, _, num_drafts, num_draft_tokens, num_accepted = _run_engine(
        "dflash", _PROMPTS, 64
    )
    min_drafts = _DRAFTS_PER_REQ_FLOOR * len(_PROMPTS)
    assert num_drafts >= min_drafts, (
        f"too few drafts to judge acceptance: {num_drafts} < {min_drafts} "
        f"({num_drafts / len(_PROMPTS):.1f} drafts/request)"
    )
    mean_accepted = num_accepted / num_drafts
    print(
        "\n  df2t acceptance:"
        f" drafts={num_drafts} drafted={num_draft_tokens} accepted={num_accepted}"
        f" mean_accepted/draft={mean_accepted:.3f}"
        f" acceptance_length={1 + mean_accepted:.3f}",
        flush=True,
    )
    assert mean_accepted > 1.0, (
        "df2t acceptance collapsed to the degenerate floor "
        f"(mean accepted/draft {mean_accepted:.3f}); the drafter is not "
        "conditioning on the accepted chain."
    )
    ref_ids, ref_probs, *_ = _run_engine(None, _PROMPTS, 64)
    got_ids, got_probs, *_ = _run_engine("dflash", _PROMPTS, 64)
    assert_t0_parity(got_ids, ref_ids, got_probs, ref_probs, "acceptance parity")
