# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tree-aware sampling feature oracle gates.

The additive logits processors (repetition/frequency/presence penalties, bad
words, thinking budget and custom processors) treat the verify block as a
linear draft chain: a row's "already emitted" prefix is the arena order of the
rows before it. Under a tree that is wrong — siblings and other branches
sit in arena order and are not on the accepted chain — so the processed
distribution deviates from the chain they are supposed to reproduce. This suite
pins the fix: a tree request with these features active produces the SAME token
stream as the plain (non-speculative) target run with the same parameters.

Exact token parity is only a valid oracle in fp32 (in bf16 the target's own
decoder flips on one-ulp logit ties with batch shape), so the parity runs use
the fp32 flex vehicle. T=0 removes the RNG from the comparison; T>0
distribution-parity of the processed rows is pinned at the kernel level in
``tests/v1/worker/gpu/spec_decode/tree/test_df2t_tree_sampling.py``.
"""

import gc
import os

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "/mnt/e/dev/Qwen3-0.6B"
_DRAFT = "/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16"
_NUM_SPEC = 15
_PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Photosynthesis is the process by which",
    "In 1969, the Apollo 11 mission landed",
    "Water boils at 100 degrees",
]

_BAD_WORDS = ["the", "France"]


def _offline_models_present() -> bool:
    return os.path.isdir(_TARGET) and os.path.isdir(_DRAFT)


def _run_engine(
    prompts: list[str],
    max_tokens: int,
    method: str | None,
    sampling_params,
    logits_processors: list | None = None,
) -> list[list[int]]:
    from vllm import LLM

    speculative_config = None
    if method is not None:
        speculative_config = {
            "method": method,
            "model": _DRAFT,
            "num_speculative_tokens": _NUM_SPEC,
            "tree_attention_backend": "flex",
        }
    llm = LLM(
        model=_TARGET,
        speculative_config=speculative_config,
        enforce_eager=True,
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=512,
        dtype="float32",
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
        disable_log_stats=True,
        attention_config={"backend": "FLEX_ATTENTION"},
        logits_processors=logits_processors,
    )
    outs = llm.generate(prompts, sampling_params)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    gc.collect()
    return ids


def _assert_parity(df2t_ids: list[list[int]], ref_ids: list[list[int]]) -> None:
    for i, (g, want) in enumerate(zip(df2t_ids, ref_ids)):
        assert g == want, (
            f"prompt {i}: df2t output diverges from the target with the same "
            f"sampling features: got {g[:20]}... want {want[:20]}..."
        )


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_repetition_penalty_matches_target_t0():
    """A tree request with a repetition/frequency/presence penalty at T=0 must
    emit exactly the same token stream as the non-speculative target run.

    At T=0 the penalty changes the argmax, so this is a full end-to-end probe
    of the tree-correct accepted-chain context the penalties kernel is fed:
    before the fix the verify rows counted arena-order siblings, so the
    processed distribution (and the emitted token) diverged from the target.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        repetition_penalty=1.4,
        frequency_penalty=0.5,
        presence_penalty=0.3,
    )
    reference = _run_engine(_PROMPTS, 64, None, params)
    got = _run_engine(_PROMPTS, 64, "df2t", params)
    _assert_parity(got, reference)
    print("\n  df2t repetition-penalty T=0 == target greedy (fp32)", flush=True)


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_bad_words_matches_target_t0():
    """Bad-words masking reads the same "already emitted" prefix a chain would:
    at T=0 a tree request masks the exact same tokens the target masks."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=64, temperature=0.0, bad_words=_BAD_WORDS)
    reference = _run_engine(_PROMPTS, 64, None, params)
    got = _run_engine(_PROMPTS, 64, "df2t", params)
    _assert_parity(got, reference)
    print("\n  df2t bad-words T=0 == target greedy (fp32)", flush=True)


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_custom_logits_processor_matches_target_t0():
    """A custom logits processor that consumes the tree-correct accepted-chain
    context (ctx.chain_prefix / ctx.chain_prefix_len under tree
    verification) must make df2t
    emit the same T=0 stream as the target.

    The processor biases a fixed token by the number of accepted-chain tokens
    seen, which is 0 at a plain single-row target decode and 0 at the tree verify
    anchor row; under the (buggy) arena order the tree verify node rows would see a
    nonzero bias (sibling positions) and diverge from the target.
    """
    import torch

    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.logits_processor import (
        LogitsContext,
        LogitsProcessor,
        LogitsProcRequestState,
    )

    class ChainContextLengthBias(LogitsProcessor):
        """Bias token 32 by the length of this row's emitted context.

        The emitted-context length is ``output_len + accepted-chain``: in a
        plain target decode the per-step context is just the committed output
        (``total_len - prompt_len``); under tree verification it is that plus the row's
        accepted-chain count (``ctx.chain_prefix_len``), so the two runs agree
        per generated token index. Under the buggy arena-order assumption the
        df2t node rows would see sibling positions (the *number* of blocks
        before them) rather than the accepted-chain depth and diverge.
        """

        def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
            self.req_states = req_states
            self.bias_token = 32  # arbitrary fixed token to bias

        def add_request(self, req_idx, sampling_params) -> bool:
            return True

        def apply(self, logits, ctx: LogitsContext) -> torch.Tensor:
            total = self.req_states.total_len.gpu
            prompt = self.req_states.prompt_len.gpu
            slots = ctx.expanded_idx_mapping
            base = total[slots].to(logits.dtype) - prompt[slots].to(logits.dtype)
            if ctx.chain_prefix_len is not None:
                extra = ctx.chain_prefix_len.to(logits.dtype)
            else:
                extra = torch.zeros_like(base)
            logits = logits.clone()
            logits[..., self.bias_token] += (base + extra) * 5.0
            return logits

    params = SamplingParams(max_tokens=64, temperature=0.0)
    reference = _run_engine(
        _PROMPTS, 64, None, params, logits_processors=[ChainContextLengthBias]
    )
    got = _run_engine(
        _PROMPTS, 64, "df2t", params, logits_processors=[ChainContextLengthBias]
    )
    _assert_parity(got, reference)
    print("\n  df2t custom-processor T=0 == target greedy (fp32)", flush=True)
