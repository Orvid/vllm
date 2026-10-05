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
    from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import TreeDraft

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
        # Batched worker->scheduler tree carry. ``self._tree_packed_np`` holds
        # one step's whole tree batch (packed ``[R, 4 * W + 1]`` int64 rows:
        # tokens | depths | parents | words | num_nodes) staged by one async
        # host copy on ``self.copy_stream``; the per-request ``StashedTree``
        # lists are materialized from it in ``get_draft_tokens`` after
        # ``self.copy_event``. The pinned buffer grows to the largest batch.
        self._tree_packed_np: np.ndarray | None = None
        self._tree_arr_w: int = 0
        self._tree_pinned: torch.Tensor | None = None
        # True while a stashed tree batch is awaiting ``get_draft_tokens``. The
        # pinned buffers are single-buffered, so re-stashing before the previous
        # batch was taken would silently serve step N+1's trees for step N; the
        # flag makes that abuse fail closed.
        self._tree_pending = False
        # True when the pending stash came from a throwaway phase (kernel
        # warmup / graph capture, which stage step after step with no take):
        # such a stash is abandoned rather than served stale, so the first real
        # serving step after it starts clean instead of tripping the guard.
        self._tree_throwaway = False

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        trees: "list[TreeDraft | None] | None" = None,
        tree_packed: torch.Tensor | None = None,
        allow_pending_overwrite: bool = False,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self.trees = None
        if tree_packed is not None:
            if self._tree_pending and not allow_pending_overwrite:
                # A real serving step staging a second stash before the prior
                # one was taken would silently serve step N+1's trees for step
                # N (wrong tree => wrong mask/verify). A pending stash left by
                # a throwaway warmup/capture phase is never taken, so it is
                # abandoned instead of tripping this.
                if not self._tree_throwaway:
                    raise RuntimeError(
                        "df2t tree stash from the previous step was never "
                        "taken before the next step stashed its trees: the "
                        "scheduler would receive step N+1's trees for step N "
                        "(wrong tree => wrong mask/verify). "
                        "take_draft_token_ids() must consume each stashed "
                        "batch before the next execute_model() stages another "
                        "one."
                    )
                self._tree_pending = False
                self._tree_packed_np = None
                self._tree_throwaway = False
            self._stash_tree_packed(tree_packed)
            if allow_pending_overwrite:
                # Warmup/capture re-stash: mark this stash throwaway so any
                # later real step abandons it rather than failing the guard.
                self._tree_throwaway = True
        elif trees is not None:
            if len(trees) != len(self.req_ids):
                raise ValueError(
                    f"trees length ({len(trees)}) does not match req_ids "
                    f"({len(self.req_ids)}); per-request trees must be ordered in "
                    "parallel to the batch requests."
                )
            self.trees = [t.to_lists() if t is not None else None for t in trees]
            # A direct-list payload has no pending async batch; a leftover stash
            # (step N's trees never taken) is abandoned, never served stale.
            self._tree_pending = False
            self._tree_throwaway = False
            self._tree_packed_np = None
        else:
            # No host tree payload this step.
            self._tree_pending = False
            self._tree_throwaway = False
            self._tree_packed_np = None
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

    def _stash_tree_packed(self, packed: torch.Tensor) -> None:
        """Async-copy the packed tree batch to the host for the next get.

        One pinned D2H on the side stream carries every request's tree; the
        event sync happens on the scheduler side inside ``get_draft_tokens``.

        Args:
            packed: ``[num_reqs, 4 * W + 1]`` int64 rows in batch order
                (tokens | depths | parents | words | num_nodes).

        """
        R, cols = packed.shape
        if len(self.req_ids) != R:
            raise ValueError(
                f"tree rows ({R}) do not match req_ids ({len(self.req_ids)}); "
                "batched trees must be parallel to the batch requests."
            )
        pinned = self._tree_pinned
        if pinned is None or pinned.shape[0] < R or pinned.shape[1] != cols:
            self._tree_pinned = pinned = torch.empty(
                (R, cols), dtype=torch.int64, device="cpu", pin_memory=True
            )
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            pinned[:R].copy_(packed, non_blocking=True)
            packed.record_stream(self.copy_stream)
            self.copy_event.record()
        self._tree_packed_np = pinned[:R].numpy()
        self._tree_arr_w = (cols - 1) // 4
        self._tree_pending = True

    def _materialize_trees(self) -> list[StashedTree | None] | None:
        """Host-list the stashed tree batch, completing the async D2H.

        Consumes (clears) the pending stash so a second take for the same step
        returns nothing and a later step cannot collide with it.
        """
        if not self._tree_pending:
            return self.trees
        self.copy_event.synchronize()
        packed = self._tree_packed_np
        assert packed is not None, "a pending tree stash must have staged rows"
        W = self._tree_arr_w
        arena = list(range(W))
        rows = packed.tolist()
        out: list[StashedTree | None] = [
            StashedTree(
                token_ids=row[:W],
                depths=row[W : 2 * W],
                parents=row[2 * W : 3 * W],
                arena=arena.copy(),
                words=row[3 * W : 4 * W],
                num_nodes=row[4 * W],
            )
            for row in rows
        ]
        self._tree_pending = False
        self._tree_throwaway = False
        self._tree_packed_np = None
        self._tree_arr_w = 0
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
