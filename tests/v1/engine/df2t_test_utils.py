# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the df2t tree engine tests."""

from collections.abc import Sequence

# Near-tie slack: bf16 log-probability noise from different batch shapes
# (the tree verify batch is 1 + B rows per request wide; logits of order 1e3
# round to ~1 nat at a flip).
_BF16_TIE_LOGPROB = 4.0


def _val(x):
    """A log-probability value, from a float or a Logprob namedtuple."""
    return x.logprob if hasattr(x, "logprob") else x


def assert_t0_parity(
    tree_ids: Sequence[Sequence[int]],
    ref_ids: Sequence[Sequence[int]],
    tree_probs: Sequence[Sequence[dict[int, float]]],
    ref_probs: Sequence[Sequence[dict[int, float]]],
    what: str,
) -> None:
    """T=0 output equals the target's greedy walk, up to bf16 near-ties.

    Exact token equality is only fp32-exact: in bf16 the reduction order of
    the tree verify batch (1 + B rows per request) differs from the chain
    batch, so an argmax can flip between two tokens whose log-probabilities
    are within rounding noise of each other. The lossless guarantee itself is
    pinned bit-exactly in fp32 by ``tests/evals/df2t_losslessness.py``; here
    a flip is accepted only when the two emitted tokens' log-probabilities
    agree within the bf16 slack.

    Args:
        tree_ids: per-request token ids of the tree run.
        ref_ids: per-request token ids of the plain target run.
        tree_probs: per-request per-token ``{token_id: logprob}`` of the tree
            run.
        ref_probs: same, of the reference run.
        what: label for the assertion messages.
    """
    assert len(tree_ids) == len(ref_ids), what
    for r, (g, w) in enumerate(zip(tree_ids, ref_ids)):
        assert len(g) == len(w), f"{what}: request {r} length differs"
        for t, (gt, wt) in enumerate(zip(g, w)):
            if gt == wt:
                continue
            lp_g = _val(tree_probs[r][t][gt])
            lp_w = _val(ref_probs[r][t][wt])
            assert abs(lp_g - lp_w) < _BF16_TIE_LOGPROB, (
                f"{what}: request {r} token {t}: tree {gt} ({lp_g:.4f}) vs "
                f"chain {wt} ({lp_w:.4f}) is not a bf16 near-tie"
            )
