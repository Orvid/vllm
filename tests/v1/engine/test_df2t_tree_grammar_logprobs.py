# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end df2t tree-aware structured-output and real-logprobs oracle gates.

Two independently-reviewed defects are pinned here.

1. **Structured output masks the tree as a sequence.** The scheduler's grammar
   path validated and masked the arena-ordered draft tokens as a linear chain
   (``structured_output_manager.validate_tokens`` / ``grammar_bitmask``), so a
   tree's siblings advanced the grammar state wrongly for every row past the
   first branch: valid tokens were masked out and invalid ones let through. A
   grammar/JSON-schema df2t request therefore diverged from the non-spec target
   (or emitted schema-invalid output).
2. **Logprobs crashed the engine.** ``_df2t_verify_rejects_logprobs`` raised
   NotImplementedError inside the worker on the first df2t verify step, so a
   single ``logprobs`` request killed the server.

The fix makes both tree-aware (validate/mask along the ACCEPTED CHAIN, and emit
real per-token logprobs from the processed distribution at the accepted rows).
This suite proves the oracle: a df2t request with a JSON-schema grammar or with
``logprobs`` produces EXACTLY the same token stream and per-token logprobs as
the plain (non-speculative) target with the same parameters, at T=0, without
crashing.

Exact token parsity is only a valid oracle in fp32 (in bf16 the target's own
decoder flips on one-ulp logit ties with batch shape), so the parity runs use
the fp32 flex vehicle.
"""

import gc
import json
import math
import os

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="df2t requires CUDA", allow_module_level=True)

_TARGET = "/mnt/e/dev/Qwen3-0.6B"
_DRAFT = "/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16"
_NUM_SPEC = 15
_PROMPTS = [
    (
        "Return a JSON object whose name is a famous scientist and whose score "
        "is a number between 0 and 100. Scientist:"
    ),
    (
        "Return a JSON object with a name and a score between 0 and 100. The "
        "name is the capital of France and the score is its population rank:"
    ),
    (
        "Return a JSON object with a name and a score between 0 and 100. Name "
        "this planet and score its distance rank from the sun:"
    ),
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["name", "score"],
    "additionalProperties": False,
}


def _offline_models_present() -> bool:
    return os.path.isdir(_TARGET) and os.path.isdir(_DRAFT)


def _run_engine(prompts, sampling_params, method=None):
    """Return (token_ids, logprobs) per request for one engine launch (fp32)."""
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
    )
    outs = llm.generate(prompts, sampling_params)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    logprobs = [o.outputs[0].logprobs for o in outs]
    del llm
    gc.collect()
    return ids, logprobs


def _sampled_logprobs(logprobs, token_ids):
    """Per-token (logprob, rank) of the emitted tokens."""
    rows = []
    for pos, tid in enumerate(token_ids):
        entry = logprobs[pos].get(tid)
        rows.append((entry.logprob, entry.rank))
    return rows


def _assert_parity(got_ids, got_logprobs, want_ids, want_logprobs) -> None:
    for i, (g, g_lp, want, w_lp) in enumerate(
        zip(got_ids, got_logprobs, want_ids, want_logprobs)
    ):
        assert g == want, (
            f"prompt {i}: df2t output diverges from the target with the same "
            f"grammar: got {g[:20]}... want {want[:20]}..."
        )
        if g_lp is not None:
            assert w_lp is not None
            assert len(g_lp) == len(w_lp) == len(g)
            for pos, ((g_prob, g_rank), (w_prob, w_rank)) in enumerate(
                zip(
                    _sampled_logprobs(g_lp, g),
                    _sampled_logprobs(w_lp, want),
                )
            ):
                assert g_rank == w_rank, (
                    f"prompt {i} token {pos}: rank {g_rank} != target {w_rank}"
                )
                # The two runs are separate fp32 forwards with different
                # attention shapes (tree verify mask vs linear), so the logprob
                # values agree to fp32 rounding noise (~1e-5), not bit-exactly.
                # A wrong token or a wrong masked distribution differs by ~1.0+.
                assert math.isclose(g_prob, w_prob, rel_tol=0.0, abs_tol=1e-4), (
                    f"prompt {i} token {pos}: logprob {g_prob} != target "
                    f"{w_prob} (diff {abs(g_prob - w_prob)})"
                )


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_guided_json_matches_target_t0():
    """A df2t request constrained by a JSON schema at T=0 emits exactly the
    same token stream as the non-speculative target with the same schema, and
    every output is itself schema-valid JSON.

    Before the tree-aware grammar fix the SDK masked the arena-ordered draft
    tree as a chain, so sibling rows advanced the grammar wrongly and the df2t
    stream diverged (or produced schema-invalid output).
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(json=_SCHEMA),
    )
    want_ids, _ = _run_engine(_PROMPTS, params, method=None)
    got_ids, _ = _run_engine(_PROMPTS, params, method="df2t")
    for i, (g, want) in enumerate(zip(got_ids, want_ids)):
        assert g == want, (
            f"prompt {i}: df2t JSON-schema output diverges from the target: "
            f"got {g[:24]}... want {want[:24]}..."
        )
        assert _valid_json(g), f"prompt {i}: df2t output is not valid JSON"
    assert _valid_json(want_ids[-1]), "reference output must itself be valid"
    print("\n  df2t JSON-schema T=0 == target greedy (fp32)", flush=True)


@pytest.mark.skipif(
    not _offline_models_present(), reason="offline df2t models not present"
)
def test_df2t_logprobs_returned_and_match_target_t0():
    """A df2t request with ``logprobs`` neither crashes the engine nor returns
    empty logprobs, and at T=0 the per-token (token id, logprob, rank) exactly
    matches the non-spec target.

    Before the fix the worker raised NotImplementedError on the first df2t
    verify step, so any logprobs request killed the engine.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=48, temperature=0.0, logprobs=5)
    want_ids, want_lp = _run_engine(_PROMPTS, params, method=None)
    got_ids, got_lp = _run_engine(_PROMPTS, params, method="df2t")
    for i, (g, g_lp, want, w_lp) in enumerate(zip(got_ids, got_lp, want_ids, want_lp)):
        assert g_lp is not None, f"prompt {i}: df2t returned no logprobs"
        assert len(g_lp) == len(g), (
            f"prompt {i}: one logprob row per emitted token "
            f"({len(g_lp)} rows for {len(g)} tokens)"
        )
        assert len(g_lp[0]) >= 2, "df2t must return top-k alternatives"
    _assert_parity(got_ids, got_lp, want_ids, want_lp)
    print("\n  df2t logprobs T=0 return + match target (fp32)", flush=True)


def _valid_json(token_ids) -> bool:
    """Decode token ids and check they form an object matching ``_SCHEMA``."""
    from vllm.tokenizers.registry import get_tokenizer

    tokenizer = get_tokenizer(_TARGET, trust_remote_code=True)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(obj, dict)
        and "name" in obj
        and isinstance(obj["name"], str)
        and "score" in obj
        and isinstance(obj["score"], (int, float))
        and 0 <= obj["score"] <= 100
    )
