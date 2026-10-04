# SPDX-License-Identifier: Apache-2.0
"""Confirm the df2t verifier CUDA micro-graph replay is ENGAGED during serving.
The class method is patched (module-level so the spawned EngineCore re-applies
it) and each stage_and_replay invocation appends to a shared file."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# In-process engine so the class-method patch (applied at import) is visible to
# the engine core; with the default spawn the core re-imports only vllm modules.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import pytest  # noqa: E402

import vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier as tvm  # noqa: E402

_COUNT_FILE = "/tmp/df2t_replay_count.txt"

_orig_stage = tvm.TreeVerifier.stage_and_replay


def _spy_stage(self, token_ids, parents, num_nodes, R):
    try:
        with open(_COUNT_FILE, "a") as f:
            f.write(f"{R}\n")
    except Exception:  # noqa: BLE001
        pass
    return _orig_stage(self, token_ids, parents, num_nodes, R)


tvm.TreeVerifier.stage_and_replay = _spy_stage
_orig_capture = tvm.TreeVerifier.capture_cudagraph


def _spy_capture(self, R, *a, **k):
    try:
        with open(_COUNT_FILE, "a") as f:
            f.write(f"CAPTURE {R}\n")
    except Exception:  # noqa: BLE001
        pass
    return _orig_capture(self, R, *a, **k)


tvm.TreeVerifier.capture_cudagraph = _spy_capture


@pytest.mark.skipif(
    not (os.path.isdir("/mnt/e/dev/Qwen3-0.6B")
         and os.path.isdir("/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16")),
    reason="offline df2t models not present",
)
def test_replay_engaged():
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig

    if os.path.exists(_COUNT_FILE):
        os.remove(_COUNT_FILE)
    llm = LLM(
        model="/mnt/e/dev/Qwen3-0.6B",
        speculative_config={
            "method": "df2t",
            "model": "/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16",
            "num_speculative_tokens": 15,
            "tree_attention_backend": "fa4",
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_AND_PIECEWISE"),
        async_scheduling=False,
        max_num_batched_tokens=256,
        max_model_len=512,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        trust_remote_code=True,
    )
    out = llm.generate(
        ["The ancient library of Alexandria"],
        SamplingParams(max_tokens=48, temperature=0.0),
    )
    ids = list(out[0].outputs[0].token_ids)
    del llm
    lines = []
    if os.path.exists(_COUNT_FILE):
        with open(_COUNT_FILE) as f:
            lines = f.read().splitlines()
    captures = [line for line in lines if line.startswith("CAPTURE")]
    replays = [int(line) for line in lines if not line.startswith("CAPTURE")]
    print(f"STRUCT|captures={captures} replays={len(replays)} "
          f"out_len={len(ids)}")
    assert captures, "verifier micro-graphs were not captured"
    assert replays, "verifier micro-graph replay was never called"
