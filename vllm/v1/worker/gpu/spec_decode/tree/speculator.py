# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree drafting over any candidate-producing drafter.

``TreeSpeculatorMixin`` turns a parallel drafter into a tree drafter. The
drafter runs its step and describes what it knows about the draft positions
(``_draft_proposal`` -> :class:`DraftProposal`); the configured tree selection
method (``SpeculativeConfig.tree_selection``) queries that for candidates and
scores and ranks them into a best-first prefix tree. The drafter itself knows
nothing about trees.

Verification walks the tree with draws from the target alone, so the
candidates only affect acceptance, never the output distribution.
"""

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.tree.budget import BudgetPolicy
from vllm.v1.worker.gpu.spec_decode.tree.selection import make_tree_selector
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import TreeDraft


class TreeSpeculatorMixin:
    """Mixed in ahead of a drafter class that implements ``_draft_proposal``
    (see :func:`tree_speculator_class`)."""

    # Provided by the drafter class.
    max_num_reqs: int
    top_k: int
    num_speculative_steps: int
    parallel_drafting_token_id: int

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)  # type: ignore[call-arg]
        d = vllm_config.speculative_config
        if not hasattr(self, "top_k"):
            # No candidate head fixes the drafter's own K.
            self.top_k = d.tree_candidate_top_k
        self.max_depth = d.max_tree_depth
        self.tree_budget = d.tree_budget
        self.budget = BudgetPolicy.from_config(d)
        # ``num_speculative_tokens`` is the drafter depth (positions drafted per
        # step); the tree budget alone drives the verify width, decode shape
        # and reservation.
        self.tree_decode_width = d.tree_decode_width
        self.tree_lookahead = d.tree_lookahead
        self.nodes_max = self.budget.resolve_nodes_max()
        self._last_tree: TreeDraft | None = None
        # Row count the most recent tree build wrote into the selector's
        # persistent buffers (eager build or a replayed drafter graph), so a
        # consumer asking for more rows fails closed.
        self._last_built_rows = 0
        self.tree_selector = make_tree_selector(
            d,
            max_num_reqs=self.max_num_reqs,
            num_candidates=self.top_k,
            num_steps=self.num_speculative_steps,
            nodes_max=self.nodes_max,
            pad_token_id=self.parallel_drafting_token_id,
            device=device,
        )
        # The draft-token carry (scheduler drain slots) is the static tree
        # width, which can exceed the drafter depth.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.tree_decode_width - 1,
            dtype=torch.int64,
            device=device,
        )

    def _tree_accepted_arenas(self) -> torch.Tensor | None:
        """Accepted node arena per chain depth from the last verify.

        The drafter's next-step context must place the accepted nodes at their
        canonical chain positions ``C + depth``, where the KV compaction
        relocated their KV.
        """
        verifier = getattr(self, "tree_verifier", None)
        return None if verifier is None else verifier.accepted_arenas

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Draft candidates, select the tree, and carry its node tokens (emit
        order) as this request's draft tokens."""
        proposal = self._draft_proposal(  # type: ignore[attr-defined]
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        tree = self.tree_selector.select(proposal)
        self._last_tree = tree
        self._last_built_rows = num_reqs
        # The selector pads columns past a tree's node count with the
        # parallel-drafting mask token, so this overwrites the whole carry row.
        width = self.draft_tokens.shape[1]
        self.draft_tokens[:num_reqs] = tree.token_ids[:num_reqs, :width]

    def get_latest_tree_drafts(self, num_reqs: int) -> list[TreeDraft | None] | None:
        """Per-request ``TreeDraft`` for the current batch, or None before the
        first build.

        Reads the selector's persistent buffers, which every build overwrites
        whether it ran eagerly or inside a replayed drafter graph.

        Raises:
            RuntimeError: if ``num_reqs`` exceeds the rows the most recent build
                wrote (a stale-buffer over-read).

        """
        if self._last_tree is None:
            return None
        self._check_built_rows(num_reqs)
        s = self.tree_selector
        return [
            TreeDraft(
                token_ids=s.token_ids[i],
                depths=s.depths[i],
                parents=s.parents[i],
                arena=s.arena[i],
                words=s.words[i],
                num_nodes=s.num_nodes[i : i + 1],
            )
            for i in range(num_reqs)
        ]

    def _check_built_rows(self, num_reqs: int) -> None:
        if num_reqs > self._last_built_rows:
            raise RuntimeError(
                "tree read would over-read the last tree build: "
                f"{num_reqs} requests but only {self._last_built_rows} rows "
                "were built; draft/tree timing diverged."
            )

    def get_latest_tree_packed(self, num_reqs: int) -> torch.Tensor | None:
        """The batch's trees as packed ``[num_reqs, 4 * nodes_max + 1]`` rows
        (tokens | depths | parents | words | num_nodes), or ``None`` before the
        first build."""
        if self._last_tree is None:
            return None
        self._check_built_rows(num_reqs)
        return self.tree_selector.packed[:num_reqs]

    def _mark_draft_graph_built(self, rows: int) -> None:
        """Record the row count a replayed drafter graph just wrote."""
        self._last_built_rows = rows

    @property
    def latest_tree_drafts(self) -> list[TreeDraft | None] | None:
        """:meth:`get_latest_tree_drafts` over the last build's own width."""
        if self._last_tree is None:
            return None
        return self.get_latest_tree_drafts(self._last_tree.token_ids.shape[0])


def tree_speculator_class(drafter_cls: type) -> type:
    """The tree-drafting variant of a candidate-producing drafter class."""
    if not hasattr(drafter_cls, "_draft_proposal"):
        raise ValueError(
            f"{drafter_cls.__name__} does not propose its draft positions in "
            "parallel, so it cannot drive tree verification."
        )
    return type(
        f"Tree{drafter_cls.__name__}",
        (TreeSpeculatorMixin, drafter_cls),
        {
            "_speculator_name": "Tree"
            + getattr(drafter_cls, "_speculator_name", drafter_cls.__name__)
        },
    )
