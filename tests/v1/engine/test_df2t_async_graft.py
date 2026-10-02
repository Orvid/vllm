# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async engine df2t draft-graft routing tests.

The async scheduling path (``EngineCore.step_with_batch_queue``) cannot know the
draft tokens for the current step in advance, so for df2t it grafts the
PREVIOUS step's worker drafts + tree into each freshly scheduled output via
``_graft_df2t_async_drafts`` before the verify block is laid out. These tests
pin that engine-level seam: the graft fires only under async scheduling with
df2t, forwards the previous batch's ``DraftTokenIds`` (per-request trees intact)
to the freshly scheduled output, and stays inert when disabled or draft-less.

The end-to-end multi-request async run is covered in
``test_df2t_async_engine.py``; the per-request routing of a grafted tree into
the scheduler output is covered in ``test_scheduler.py``.
"""

from types import SimpleNamespace
from typing import Any

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, StashedTree


def _small_tree(num_nodes: int) -> StashedTree:
    return StashedTree(
        token_ids=list(range(1, num_nodes + 1)),
        depths=list(range(1, num_nodes + 1)),
        parents=[-1] + list(range(0, num_nodes - 1)),
        arena=list(range(num_nodes)),
        words=[1 << i for i in range(num_nodes)],
        num_nodes=num_nodes,
    )


class _RecordingExecutor:
    def __init__(self, sequence: list[DraftTokenIds | None]):
        self._sequence = list(sequence)
        self._i = 0
        self.taken: list[DraftTokenIds | None] = []

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._i < len(self._sequence):
            draft = self._sequence[self._i]
            self._i += 1
            self.taken.append(draft)
            return draft
        return None


class _CoreHarness:
    """A bare EngineCore (object.__new__) exposing only the state the async
    df2t graft touches, plus a recorder for the scheduler graft call."""

    def __init__(self, executor: _RecordingExecutor, **attrs: Any):
        self.recorded: list[tuple[DraftTokenIds, object]] = []
        self.core = object.__new__(EngineCore)
        defaults: dict[str, Any] = dict(
            async_scheduling=True,
            check_for_draft_tokens=True,
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(
                    use_df2t=lambda: True, use_df2t_tree=lambda: True
                )
            ),
            model_executor=executor,
            scheduler=SimpleNamespace(
                update_draft_token_ids_in_output=lambda d, s: self.recorded.append(
                    (d, s)
                )
            ),
        )
        defaults.update(attrs)
        for name, value in defaults.items():
            setattr(self.core, name, value)


def test_df2t_async_graft_forwards_previous_batch_drafts():
    """Async + df2t with a previous batch grafts its drafts + tree onto the
    freshly scheduled output, exactly as the worker take returned them."""
    prev_batch = DraftTokenIds(
        req_ids=["req-a", "req-b"],
        draft_token_ids=[[7, 3], [5]],
        spec_tree_draft=[_small_tree(2), None],
    )
    executor = _RecordingExecutor([prev_batch])
    harness = _CoreHarness(executor)

    out = SimpleNamespace(step=1)
    assert harness.core._graft_df2t_async_drafts(out) is True
    assert harness.recorded == [(prev_batch, out)]
    assert executor.taken == [prev_batch]


def test_df2t_async_graft_routes_per_invocation_across_batches():
    """With a queue depth > 1, each freshly scheduled output receives its OWN
    previous batch's DraftTokenIds, never another batch's."""
    batch1 = DraftTokenIds(
        req_ids=["req-a"],
        draft_token_ids=[[7, 3]],
        spec_tree_draft=[_small_tree(2)],
    )
    batch2 = DraftTokenIds(
        req_ids=["req-b"],
        draft_token_ids=[[9]],
        spec_tree_draft=[None],
    )
    executor = _RecordingExecutor([batch1, batch2])
    harness = _CoreHarness(executor)

    out1 = SimpleNamespace(step=1)
    out2 = SimpleNamespace(step=2)
    assert harness.core._graft_df2t_async_drafts(out1) is True
    assert harness.core._graft_df2t_async_drafts(out2) is True
    assert harness.recorded == [(batch1, out1), (batch2, out2)]
    assert executor.taken == [batch1, batch2]


def test_df2t_async_graft_noop_conditions():
    """No graft when the worker has no drafts, and the graft is fully disabled
    outside async scheduling or outside df2t (take is never called)."""
    harness = _CoreHarness(_RecordingExecutor([None]))
    assert harness.core._graft_df2t_async_drafts(SimpleNamespace(step=0)) is False
    assert harness.recorded == []

    not_async = _RecordingExecutor([DraftTokenIds(["req"], [[1]])])
    harness = _CoreHarness(not_async, async_scheduling=False)
    assert harness.core._graft_df2t_async_drafts(SimpleNamespace(step=0)) is False
    assert not_async.taken == [] and harness.recorded == []

    not_df2t = _RecordingExecutor([DraftTokenIds(["req"], [[1]])])
    harness = _CoreHarness(
        not_df2t, vllm_config=SimpleNamespace(speculative_config=None)
    )
    assert harness.core._graft_df2t_async_drafts(SimpleNamespace(step=0)) is False
    assert not_df2t.taken == [] and harness.recorded == []
