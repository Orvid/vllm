# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tree CUDA-graph capture of the uniform 1+B tree-verify decode.

df2t now captures its static tree-verify decode through the V2 uniform FULL
CUDA-graph machinery (decode_query_len = 1 + num_speculative_tokens) with
the FA4 tree mask baked into the captured metadata; ``capture_model`` no
longer raises for the supported configs. These tests drive the real engine:

* the engine with cudagraph mode enabled (not ``enforce_eager``) captures and
  serves df2t without error;
* a single-request bf16/fa4 generation is token-for-token identical between
  the graph path and the eager path (the captured target tree-verify forward
  applies the tree mask identically).

bf16 batched multi-request parity is intentionally not asserted here: the
repository's batched-divergence resolution documents that exact-token parity
is not a valid oracle in bf16 at batched shapes (near-tie flips with batch
shape even with no speculation); fp32 is the exact-parity vehicle (flex), and
its graph capture is tracked separately. A single request has no batch-shape
variation, so bf16 exactness is stable.
"""

import pytest
import torch

from vllm import LLM, SamplingParams

_TARGET = "Qwen/Qwen3-8B"
_DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
_PROMPT = "The ancient library of Alexandria"


def _offline_models_present() -> bool:
    import torch

    return torch.cuda.is_available()


def _run(cudagraph_mode: str) -> list[int]:
    from vllm.config.compilation import CompilationConfig

    llm = LLM(
        model=_TARGET,
        speculative_config={
            "method": "dflash",
            "tree_selection": "df2t",
            "model": _DRAFT,
            "num_speculative_tokens": 15,
            "tree_attention_backend": "fa4",
        },
        compilation_config=CompilationConfig(cudagraph_mode=cudagraph_mode),
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=512,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        disable_log_stats=False,
    )
    out = llm.generate([_PROMPT], SamplingParams(max_tokens=96, temperature=0.0))
    ids = list(out[0].outputs[0].token_ids)
    del llm
    torch.accelerator.empty_cache()
    return ids


def _can_run_fa4_tree_graphs() -> bool:
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    return _offline_models_present() and (cap is None or cap.major != 12)


@pytest.mark.skipif(
    not _can_run_fa4_tree_graphs(),
    reason="FA4 FULL tree graphs are disabled on SM12x (no fork changes)",
)
def test_df2t_serves_with_full_cudagraphs_and_matches_eager_single_request() -> None:
    """df2t captures and serves under FULL CUDA graphs, single-request output
    identical to eager (tree-verify forward bit-identical at equal shapes)."""
    graph_ids = _run("FULL_AND_PIECEWISE")
    eager_ids = _run("NONE")
    assert len(graph_ids) == 96
    assert graph_ids == eager_ids, (
        "tree CUDA-graph decode diverges from eager at a single-request shape:\n"
        f"  graph: {graph_ids[:48]}\n  eager: {eager_ids[:48]}"
    )


@pytest.mark.skipif(
    not _can_run_fa4_tree_graphs(),
    reason="FA4 FULL tree graphs are disabled on SM12x (no fork changes)",
)
def test_df2t_with_piecewise_only_mode_fails_closed() -> None:
    """Tree CUDA-graph capture is the FULL uniform 1+B decode graph only; a
    PIECEWISE-only mode cannot serve it and must fail closed (precise error
    surfaced in the engine core; the API boundary reports init failure)
    instead of capturing graphs that would replay without the tree mask."""
    from vllm.config.compilation import CompilationConfig

    with pytest.raises(RuntimeError, match="Engine core initialization failed"):
        LLM(
            model=_TARGET,
            speculative_config={
                "method": "dflash",
                "tree_selection": "df2t",
                "model": _DRAFT,
                "num_speculative_tokens": 15,
                "tree_attention_backend": "fa4",
            },
            compilation_config=CompilationConfig(cudagraph_mode="PIECEWISE"),
            async_scheduling=False,
            max_num_batched_tokens=256,
            max_model_len=512,
            dtype="bfloat16",
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            disable_log_stats=False,
        )
