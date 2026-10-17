# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tree verify-block layout (positions/slots/logits).

Builds the target verify block for a scheduled tree: per request the
trailing ``1 + B`` token rows become ``[anchor @ C] + [node i @ C + depth_i]``
with RoPE positions ``C + depth_i`` and decoupled KV slots ``C + arena_i``.
``prepare_tree_inputs`` owns that layout; the block-table kernel then maps the
``slot_positions`` to physical KV slots. These tests pin the layout on CPU and
GPU from raw deterministic tensors.
"""

import pytest
import torch

from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import prepare_tree_inputs
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import TreeDraft


def _tree(depths: list[int], token_ids: list[int], device: str) -> TreeDraft:
    b = len(depths)
    return TreeDraft(
        token_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
        depths=torch.tensor(depths, dtype=torch.long, device=device),
        parents=torch.tensor([-1] * b, dtype=torch.long, device=device),
        arena=torch.arange(b, dtype=torch.long, device=device),
        words=torch.zeros(b, dtype=torch.long, device=device),
    )


def _fake_batch(
    anchor_pos: int,
    trees: list[TreeDraft | None],
    last_sampled: int = 100,
    num_new_sampled_tokens: int = 1,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """A realistic verify batch fragment: consecutive positions and logits rows
    as ``prepare_pos_seq_lens``/``combine_sampled_and_draft_tokens`` would leave
    them, so the test only exercises the tree rewrite."""
    num_reqs = len(trees)
    num_logits_per_req = [
        -1 if t is None else t.token_ids.shape[0] + num_new_sampled_tokens
        for t in trees
    ]
    num_logits = [n for n in num_logits_per_req if n > 0]
    total_logits = sum(num_logits)
    buf_size = max(16, total_logits)

    input_ids = torch.zeros(buf_size, dtype=torch.int32, device=device)
    positions = torch.zeros(buf_size, dtype=torch.int64, device=device)
    slot_positions = torch.zeros(buf_size, dtype=torch.int64, device=device)
    logits_indices = torch.zeros(total_logits, dtype=torch.int64, device=device)

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    # Each tree request owns a contiguous block of (1 + B) token rows.
    query_start = [0]
    for n in num_logits_per_req:
        query_start.append(query_start[-1] + max(n, 1))
    query_start_loc = torch.tensor(query_start, dtype=torch.int32, device=device)

    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    prefill_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    last_sampled_tokens = torch.full(
        (num_reqs,), last_sampled, dtype=torch.int32, device=device
    )

    cu_num_logits = [0]
    for r, t in enumerate(trees):
        start = query_start[r]
        n = num_logits_per_req[r]
        if t is None:
            cu_num_logits.append(cu_num_logits[-1])
            continue
        positions[start : start + n] = anchor_pos + torch.arange(
            n, dtype=torch.int64, device=device
        )
        seq_lens[r] = anchor_pos + n
        prefill_len[r] = anchor_pos
        logits_indices[cu_num_logits[-1] : cu_num_logits[-1] + n] = torch.arange(
            start, start + n, dtype=torch.int64, device=device
        )
        cu_num_logits.append(cu_num_logits[-1] + n)
    cu_num_logits_t = torch.tensor(cu_num_logits, dtype=torch.int32, device=device)

    return {
        "input_ids": input_ids,
        "positions": positions,
        "slot_positions": slot_positions,
        "idx_mapping": idx_mapping,
        "last_sampled_tokens": last_sampled_tokens,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "prefill_len": prefill_len,
        "cu_num_logits": cu_num_logits_t,
        "logits_indices": logits_indices,
        "trees": trees,
        "num_new_sampled_tokens": num_new_sampled_tokens,
    }


def _run(batch: dict[str, torch.Tensor]) -> None:
    prepare_tree_inputs(
        batch["input_ids"],
        batch["positions"],
        batch["slot_positions"],
        batch["idx_mapping"],
        batch["last_sampled_tokens"],
        batch["query_start_loc"],
        batch["seq_lens"],
        batch["prefill_len"],
        batch["cu_num_logits"],
        batch["logits_indices"],
        batch["trees"],
        num_new_sampled_tokens=batch["num_new_sampled_tokens"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tree_verify_layout_positions_slots_logits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    C = 7
    tree = _tree([1, 1, 2], [10, 11, 12], device)
    batch = _fake_batch(C, [tree])
    _run(batch)

    assert batch["input_ids"][:4].tolist() == [100, 10, 11, 12]
    assert batch["positions"][:4].tolist() == [7, 8, 8, 9]
    # Slots decouple from depth: anchor at C, node i at C + arena_i.
    assert batch["slot_positions"][:4].tolist() == [7, 8, 9, 10]
    assert batch["logits_indices"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tree_requests_without_tree_are_untouched(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    C = 3
    tree = _tree([1, 2], [20, 21], device)
    batch = _fake_batch(C, [None, tree])
    _run(batch)

    # Request 0 has no tree: nothing written into its single (unused) row.
    assert batch["positions"][0].item() == 0
    assert batch["input_ids"][0].item() == 0
    # Request 1's lone tree row exists at rows 1..3.
    assert batch["positions"][1:4].tolist() == [3, 4, 5]
    assert batch["slot_positions"][1:4].tolist() == [3, 4, 5]
    assert batch["input_ids"][1:4].tolist() == [100, 20, 21]
    assert batch["logits_indices"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tree_verify_layout_anchor_uses_last_sampled(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    C = 5
    tree = _tree([1], [30], device)
    batch = _fake_batch(C, [tree], last_sampled=777)
    _run(batch)

    assert batch["input_ids"][:2].tolist() == [777, 30]
    assert batch["positions"][:2].tolist() == [5, 6]
    assert batch["slot_positions"][:2].tolist() == [5, 6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_verify_slots_translate_through_block_table():
    """The layout, run through the real slot kernel, gives every sibling its
    own physical KV slot even when RoPE positions collide, and keeps them
    distinct across a block boundary. The un-decoupled chain layout would map
    the depth-1 siblings to the same slot."""
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8],
        device=device,
        kernel_block_sizes=[16],
    )
    block_tables.append_block_ids(0, ([100, 101],), overwrite=False)
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    # C = 14 sits just before a 16-token block boundary: the anchor and first
    # depth-1 node land in block 100, the remaining nodes cross into block 101.
    C = 14
    tree = _tree([1, 1, 2], [10, 11, 12], "cuda")
    batch = _fake_batch(C, [tree], device="cuda")
    _run(batch)
    assert batch["positions"][:4].tolist() == [14, 15, 15, 16]

    num_logits = 4
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
    slot_positions = batch["slot_positions"][:4]
    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        slot_positions,
        num_tokens_padded=num_logits,
    )[0].clone()
    assert slot_mappings.tolist() == [100 * 16 + 14, 1615, 101 * 16, 1617]

    # The un-decoupled chain layout (positions directly as slots) would hand
    # both depth-1 siblings, at positions 15 and 15, the same KV slot.
    chain_slots = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        batch["positions"][:4],
        num_tokens_padded=num_logits,
    )[0]
    assert chain_slots.tolist() == [1614, 1615, 1615, 1616]
    assert len(set(slot_mappings.tolist())) == num_logits
    assert chain_slots[1] == chain_slots[2]
