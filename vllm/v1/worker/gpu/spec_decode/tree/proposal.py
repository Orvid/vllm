# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a parallel drafter knows about one step's draft positions.

A drafter that proposes its whole block at once (``_draft_proposal``) hands the
tree selection method a :class:`DraftProposal`; the selection method decides
which candidates to take, which scores to ask for, and how to rank them, so
drafters never shape the tree.

Draft position ``s`` of request ``r`` follows position ``s - 1`` (the request's
anchor token at ``s = 0``). Every query runs on the device without host syncs,
inside the drafter's step.
"""

from collections.abc import Callable

import torch


class DraftProposal:
    """Per-position distributions with no interaction between positions
    beyond their shared context (a diffusion drafter's parallel denoising).

    Args:
        logits: ``[R, S, V]`` target-vocab logits per draft position.
        hidden: ``[R, S, H]`` the drafter's hidden state per draft position,
            if it has one.

    """

    def __init__(self, logits: torch.Tensor, hidden: torch.Tensor | None = None):
        self._logits = logits
        self._hidden = hidden

    @property
    def num_positions(self) -> int:
        return self._logits.shape[1]

    def position_hidden(self) -> torch.Tensor | None:
        """``[R, S, H]`` the drafter's hidden state per draft position (what
        its logits are computed from), or ``None``."""
        return self._hidden

    def marginal_logits(self) -> torch.Tensor | None:
        """``[R, S, V]`` target-vocab logits per position, if the drafter
        materializes them."""
        return self._logits

    def candidates(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The drafter's ``k`` most likely tokens per position: target-vocab
        ``ids [R, S, k]`` and their logits ``[R, S, k]``."""
        logits = self.marginal_logits()
        assert logits is not None
        values, ids = logits.topk(k, dim=-1)
        return ids, values

    def conditional(self, ids: torch.Tensor) -> torch.Tensor | None:
        """Logit of ``ids[r, s, j]`` following ``ids[r, s - 1, i]`` (the
        anchor at ``s = 0``, for every ``i``): ``[R, S, C, C]``. ``None`` when
        a position does not depend on its predecessor's token, so each
        candidate's own logit scores every edge into it."""
        return None


class LazyDraftProposal(DraftProposal):
    """A :class:`DraftProposal` whose logits are computed on first query.

    The chain-sampling tail never queries the proposal's logits (the sampler
    recomputes its own), so a drafter split into ``_draft_forward`` +
    ``_draft_sample`` can defer the logits head to the tree selector without
    paying for it on the plain path.
    """

    def __init__(
        self,
        hidden: torch.Tensor,
        logits_fn: Callable[[], torch.Tensor],
    ):
        self._hidden = hidden
        self._logits_fn = logits_fn
        self._logits = None

    @property
    def num_positions(self) -> int:
        hidden = self._hidden
        assert hidden is not None
        return hidden.shape[1]

    def marginal_logits(self) -> torch.Tensor:
        """``[R, S, V]``, computed from ``logits_fn`` on first access."""
        if self._logits is None:
            self._logits = self._logits_fn()
        return self._logits


class CandidateHeadProposal(DraftProposal):
    """A drafter whose candidate head fixes its candidates and scores every
    edge between adjacent positions' candidates.

    Args:
        ids: ``[R, S, K]`` the head's candidates.
        scores: ``[R, S, K, K]`` conditional logits between them (row ``i`` at
            ``s = 0`` follows the anchor).
        hidden: ``[R, S, H]`` the drafter's hidden state per draft position,
            if it has one.

    """

    def __init__(
        self,
        ids: torch.Tensor,
        scores: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        self._ids = ids
        self._scores = scores
        self._hidden = hidden

    @property
    def num_positions(self) -> int:
        return self._ids.shape[1]

    def marginal_logits(self) -> torch.Tensor | None:
        return None

    def candidates(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert k == self._ids.shape[-1], (k, self._ids.shape)
        return self._ids, self._scores[:, :, 0]

    def conditional(self, ids: torch.Tensor) -> torch.Tensor | None:
        """Scores between the head's own candidates; ``ids`` must be those
        :meth:`candidates` returned."""
        assert ids.shape == self._ids.shape
        return self._scores
