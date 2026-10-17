# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-aware structured-output grammar tests for tree verification (WS-B-2).

``structured_output_manager`` advanced the grammar state along the ARENA-ORDERED
draft tokens as if they were a linear chain. Under a tree that is wrong:
siblings and other branches sit in arena order and are not on the accepted
chain, so a node's bitmask row was computed against the wrong prefix (valid
tokens masked out, invalid ones let through). These tests pin the fix: the
grammar is advanced/masked along each node's ACCEPTED CHAIN (its ancestors plus
itself, in depth order) and the tree verify draft carry is never chain-validated.

The regex ``[a-z]*[0-9]+`` is chosen so the next-token mask observably depends on
whether a digit has been seen: after letters-only any letter/digit is allowed,
after the first digit only digits are allowed. A crafted tree puts a digit
sibling early in arena order, so the (buggy) chain advance reaches a different
state than the node's real accepted chain.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

# These tests use only local offline model paths; never reach the HF hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig

from vllm.config import (
    DeviceConfig,
    ParallelConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.outputs import StashedTree
from vllm.v1.request import Request
from vllm.v1.structured_output import (
    StructuredOutputManager,
    _tree_node_chains,
)
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

TOKENIZER = "Qwen/Qwen3-0.6B"
DRAFT_ROOT = "z-lab/Qwen3-8B-DFlash-b16"
REGEX = "[a-z]*[0-9]+"
MAX_WAIT_SECONDS = 60


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def letters(tokenizer):
    """Single-token letter/digit ids (skipped if a char is multi-token)."""
    out = {}
    for ch in "a1cdz":
        ids = tokenizer.encode(ch)
        assert len(ids) == 1, (ch, ids)
        out[ch] = ids[0]
    return out


@dataclass(frozen=True)
class _Tree:
    """A 2-level tree in arena (emit) order with a digit sibling first."""

    token_ids: tuple[int, ...]  # node tokens, arena order
    parents: tuple[int, ...]  # parent arena index per node (-1 = anchor child)
    depths: tuple[int, ...]


def _make_tree(letters) -> _Tree:
    a, one, c, d = letters["a"], letters["1"], letters["c"], letters["d"]
    # Arena order: node0 = "a" (depth1), node1 = "1" (depth1 sibling),
    # node2 = "c" (depth2 under node0), node3 = "d" (depth2 under node0).
    # Node2's accepted chain is ["a", "c"]; the chain-advance prefix for its
    # row is arena order ["a", "1", "c"], which crosses the digit "1".
    return _Tree(
        token_ids=(a, one, c, d),
        parents=(-1, -1, 0, 0),
        depths=(1, 1, 2, 2),
    )


def _stashed(tree: _Tree) -> StashedTree:
    return StashedTree(
        token_ids=list(tree.token_ids),
        depths=list(tree.depths),
        parents=list(tree.parents),
        arena=list(range(len(tree.token_ids))),
        words=[0] * len(tree.token_ids),
        num_nodes=len(tree.token_ids),
    )


def _wait_for_grammar(request: Request) -> None:
    structured_req = request.structured_output_request
    assert structured_req is not None
    deadline = time.time() + MAX_WAIT_SECONDS
    while not structured_req._check_grammar_completion():
        if time.time() > deadline:
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)
    assert structured_req.grammar is not None


def _init_request(manager: StructuredOutputManager, tokenizer, req_id: str):
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(regex=REGEX)
    )
    sampling_params.structured_outputs._backend = "xgrammar"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
    request = Request(
        request_id=req_id,
        prompt_token_ids=[],
        sampling_params=sampling_params,
        pooling_params=None,
        reasoning_ended=True,
    )
    manager.grammar_init(request)
    _wait_for_grammar(request)
    return request


def _reference_rows(manager, request, tree: _Tree) -> list[torch.Tensor]:
    """Per verify-row reference mask (torch row): row 0 = current grammar,
    row 1+a = grammar advanced through node a's accepted chain (1-row int32)."""
    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert isinstance(grammar, StructuredOutputGrammar)
    nwords = (grammar.vocab_size + 31) // 32
    chains = _tree_node_chains(_stashed(tree))
    rows = []

    def fill(apply: bool) -> torch.Tensor:
        buf = torch.zeros(1, nwords, dtype=torch.int32)
        if apply and not grammar.is_terminated():
            grammar.fill_bitmask(buf, 0)
        else:
            buf[:].fill_(-1)
        return buf

    rows.append(fill(True))
    for path in chains:
        grammar.accept_tokens(request.request_id, path)
        rows.append(fill(True))
        grammar.rollback(len(path))
    return rows


def _df2t_speculative_config(tmp_path: Path) -> SpeculativeConfig:
    """A real df2t SpeculativeConfig over locally-written Qwen3 configs."""
    common = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=2048,
    )
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    target_path.mkdir()
    draft_path.mkdir()
    (target_path / "config.json").write_text(
        LlamaConfig(architectures=["LlamaForCausalLM"], **common).to_json_string()
    )
    (draft_path / "config.json").write_text(
        LlamaConfig(
            **{
                **common,
                "architectures": ["DFlash2DraftModel"],
                "model_type": "qwen3",
                "num_hidden_layers": 1,
                "layer_types": ["sliding_attention"],
                "sliding_window": 128,
                "dflash_config": dict(
                    block_size=4,
                    mask_token_id=255,
                    target_layer_ids=[0],
                    conv_group_size=2,
                    conv_kernel_size=2,
                    selector_rank=8,
                    selector_top_k=4,
                ),
            }
        ).to_json_string()
    )
    model_config = ModelConfig(
        model=str(target_path), tokenizer_mode="skip", max_model_len=2048
    )
    return SpeculativeConfig(
        method="dflash",
        tree_selection="df2t",
        model=str(draft_path),
        num_speculative_tokens=3,
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(distributed_executor_backend="uni"),
    )


def test_tree_node_chains(tokenizer, letters):
    """The accepted-chain helper must reproduce each node's depth-order path
    (ancestors + itself), independent of the arena order of its siblings."""
    tree = _make_tree(letters)
    chains = _tree_node_chains(_stashed(tree))
    assert chains == [
        [letters["a"]],
        [letters["1"]],
        [letters["a"], letters["c"]],
        [letters["a"], letters["d"]],
    ]


def test_tree_grammar_bitmask_uses_accepted_chain(tokenizer, letters):
    """Each verify row's bitmask must come from the grammar advanced through the
    node's ACCEPTED CHAIN, not the arena-ordered prefix.

    The tree's digit sibling (node1, arena position 1) crosses the chain
    advance for node2 ("a","c"): advancing the grammar along arena order
    ``["a","1","c"]`` reaches a digits-only state, while node2's real chain
    ``["a","c"]`` still allows letters. So row 3 (node2) must equal the
    letters-still-allowed reference and differ from the digits-only chain
    advance — otherwise siblings corrupt the grammar state and valid tokens are
    masked out.
    """
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=TOKENIZER),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=8),
    )
    manager = StructuredOutputManager(vllm_config)
    request = _init_request(manager, tokenizer, "tree-flow")
    tree = _make_tree(letters)
    stashed = _stashed(tree)
    refs = _reference_rows(manager, request, tree)
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: list(stashed.token_ids)},
        scheduled_spec_tree_draft={request.request_id: stashed},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == 5
    for row in range(5):
        assert (torch.from_numpy(bitmask[row]) == refs[row][0]).all(), (
            f"verify row {row}: bitmask does not match the accepted-chain grammar state"
        )
    # The discrimination: the arena-order advance for node2 reaches digits-only
    # (its "c" is rejected once the digit "1" has been seen, so the chain fill
    # would mask row 3 digits-only instead of allowing letters).
    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert grammar is not None
    arena = [letters["a"], letters["1"], letters["c"]]
    accepted = 0
    for t in arena:
        if grammar.is_terminated() or not grammar.accept_tokens(
            request.request_id, [t]
        ):
            break
        accepted += 1
    arena_adv = torch.zeros(1, bitmask.shape[1], dtype=torch.int32)
    grammar.fill_bitmask(arena_adv, 0)
    grammar.rollback(accepted)
    assert not (torch.from_numpy(bitmask[3]) == arena_adv[0]).all(), (
        "node2's row must differ from the arena-order (digits-only) advance"
    )


def test_validate_tokens_skips_chain_check_for_df2t(tmp_path, tokenizer, letters):
    """df2t drafts are arena-ordered tree tokens, not a chain: validating them
    as a sequence would drop valid on-chain tokens past a branch. Under df2t
    the manager must return the full arena list untouched."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=TOKENIZER),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=_df2t_speculative_config(tmp_path),
    )
    manager = StructuredOutputManager(vllm_config)
    assert manager._is_tree()
    request = _init_request(manager, tokenizer, "df2t-validate")
    drafts = [letters["a"], letters["1"], letters["c"]]
    assert manager.validate_tokens(request, drafts) == drafts
