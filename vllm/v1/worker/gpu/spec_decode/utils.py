# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds, StashedTree
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft

logger = init_logger(__name__)


def get_pp_safe_draft_load_config(load_config: LoadConfig) -> LoadConfig:
    """Avoid collectives that include PP ranks without a draft model."""
    if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
        logger.warning_once(
            "fastsafetensors cannot load a draft model instantiated on only "
            "one pipeline stage; falling back to the standard safetensors "
            "loader for the draft model. The target model still uses "
            "fastsafetensors."
        )
        return replace(load_config, load_format="auto")
    return load_config


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        # Per-request stashed tree (parallel to req_ids), or None for the plain
        # draft-token path. Trees ride along with the token ids into the
        # scheduler for the next-step verify.
        self.trees: list[StashedTree | None] | None = None
        # Batched worker->scheduler tree carry. ``self._tree_arrays_np`` holds
        # one step's whole tree batch (5 x [R, W] int64 + [R] int32 num_nodes)
        # staged by an async host copy on ``self.copy_stream``; the per-request
        # ``StashedTree`` lists are materialized from it in
        # ``get_draft_tokens`` after ``self.copy_event``, so the per-request
        # blocking ``.tolist()``/``.item()`` round-trips stay off the worker's
        # per-step path. Buffers are pinned and grown to the largest batch seen.
        self._tree_arrays_np: list[np.ndarray] | None = None
        self._tree_arr_w: int = 0
        self._tree_pinned: list[torch.Tensor] | None = None

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        trees: "list[TreeDraft | None] | None" = None,
        tree_arrays: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self.trees = None
        self._tree_arrays_np = None
        if tree_arrays is not None:
            self._stash_tree_arrays(tree_arrays)
        elif trees is not None:
            if len(trees) != len(self.req_ids):
                raise ValueError(
                    f"trees length ({len(trees)}) does not match req_ids "
                    f"({len(self.req_ids)}); per-request trees must be ordered in "
                    "parallel to the batch requests."
                )
            self.trees = [
                t.to_lists() if t is not None else None for t in trees
            ]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def _stash_tree_arrays(self, tree_arrays: tuple[torch.Tensor, ...]) -> None:
        """Async-batch the tree-carry D2H and stage it for the next get.

        The df2t tree batch arrives as one ``[num_reqs, B]`` device tensor per
        field (views of the builder buffers). All requests' trees are copied in
        a handful of batched host copies on the side stream (pinned ``H2D``
        destinations targeting only the live rows), so the per-request
        ``.tolist()``/``.item()`` round-trips of ``TreeDraft.to_lists`` never
        block the worker's per-step path; the single event sync happens on the
        scheduler side inside ``get_draft_tokens``, the async-overlap point.

        Args:
            tree_arrays: ``(token_ids, depths, parents, arena, words, num_nodes)``
                ``[R, B]`` (``[R]`` for num_nodes) device views in batch order.

        """
        token_ids, depths, parents, arena, words, num_nodes = tree_arrays
        R = token_ids.shape[0]
        W = token_ids.shape[1]
        if len(self.req_ids) != R:
            raise ValueError(
                f"tree_arrays rows ({R}) do not match req_ids ({len(self.req_ids)}); "
                "batched trees must be parallel to the batch requests."
            )
        pinned = self._tree_pinned
        if pinned is None or pinned[0].shape[0] < R or pinned[0].shape[1] < W:
            self._tree_pinned = pinned = [
                torch.empty((R, W), dtype=torch.int64, device="cpu", pin_memory=True)
                for _ in range(5)
            ] + [
                torch.empty((R,), dtype=torch.int32, device="cpu", pin_memory=True)
            ]
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            for src, dst in zip(tree_arrays[:5], pinned[:5]):
                dst[:R, :W].copy_(src, non_blocking=True)
            pinned[5][:R].copy_(tree_arrays[5], non_blocking=True)
            for t in tree_arrays:
                t.record_stream(self.copy_stream)
            self.copy_event.record()
        self._tree_arrays_np = [p[:R, :W].numpy() for p in pinned[:5]] + [
            pinned[5][:R].numpy()
        ]
        self._tree_arr_w = W

    def _materialize_trees(self) -> list[StashedTree | None] | None:
        """Host-list the stashed tree batch, completing the async D2H."""
        arrays = self._tree_arrays_np
        if arrays is None:
            return self.trees
        self.copy_event.synchronize()
        tok, dep, par, are, wrd, nn = arrays
        W = self._tree_arr_w
        out: list[StashedTree | None] = []
        for i in range(tok.shape[0]):
            out.append(
                StashedTree(
                    token_ids=tok[i, :W].tolist(),
                    depths=dep[i, :W].tolist(),
                    parents=par[i, :W].tolist(),
                    arena=are[i, :W].tolist(),
                    words=wrd[i, :W].tolist(),
                    num_nodes=int(nn[i]),
                )
            )
        return out

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(
            self.req_ids,
            draft_token_ids,
            spec_tree_draft=self._materialize_trees(),
        )


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
