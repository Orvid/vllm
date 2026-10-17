# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive tree verify width: calibration, width choice and batch trimming."""

import types

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.spec_decode.tree.adaptive_width import (
    TreeWidthController,
    build_cost_table,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
# Two confident nodes, then two doubtful ones (best-first path log-probs).
LOGP = [-0.05, -0.1, -2.0, -2.1]
DEPTHS = [1, 2, 1, 2]


def _record(
    ctl: TreeWidthController,
    slots: list[int],
    logp_row: list[float] = LOGP,
    depth_row: list[int] = DEPTHS,
) -> None:
    idx = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    logp = torch.tensor([logp_row] * len(slots), dtype=torch.float32, device=DEVICE)
    depths = torch.tensor([depth_row] * len(slots), dtype=torch.int64, device=DEVICE)
    ctl.record_trees(idx, np.array(slots), logp, depths)
    torch.accelerator.synchronize()


def _observe_first_two_accepted(ctl: TreeWidthController, slots: list[int]) -> None:
    idx = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    num_nodes = torch.full((len(slots),), 4, dtype=torch.int64, device=DEVICE)
    arenas = torch.full((len(slots), 5), -1, dtype=torch.int64, device=DEVICE)
    arenas[:, 1] = 0
    arenas[:, 2] = 1
    ctl.observe(idx, num_nodes, arenas)


def test_calibration_learns_per_node_acceptance():
    """Nodes 0-1 always accepted and 2-3 never: the expected accepted count
    converges to 2 at both widths, whatever the exp(logp) prior said."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    _record(ctl, [1])
    for _ in range(200):
        _observe_first_two_accepted(ctl, [1])
    _record(ctl, [1])
    ctl._poll()
    curve = ctl.host_curves[1]
    assert curve[0] == pytest.approx(2.0, abs=0.05)
    assert curve[1] == pytest.approx(2.0, abs=0.05)


def test_calibration_separates_depths_at_equal_path_mass():
    """A drafter equally sure of a whole chain: the target agrees with the
    first two positions only, so the curve must not credit the deeper nodes
    for sharing their path mass."""
    chain = [-0.05] * 4
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    _record(ctl, [1], chain, [1, 2, 3, 4])
    for _ in range(200):
        _observe_first_two_accepted(ctl, [1])
    _record(ctl, [1], chain, [1, 2, 3, 4])
    ctl._poll()
    curve = ctl.host_curves[1]
    assert curve[0] == pytest.approx(2.0, abs=0.05)
    assert curve[1] == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("narrow_cost, expected", [(1.0, 4), (0.5, 2)])
def test_select_trades_acceptance_against_cost(narrow_cost, expected):
    """Width 4 adds ~0.27 expected tokens per request (the exp(logp) prior on
    the two doubtful nodes): worth it at equal cost, not at double the cost."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): narrow_cost, (4, 1): 1.0}, [2, 4], 4))
    _record(ctl, [0, 1])
    assert ctl.select(np.array([0, 1])) == expected


def test_measured_step_time_moves_the_width_choice():
    """The profile prices both widths at 1 ms, so width 4 wins on expected
    tokens; measured 3 ms steps at width 4 correct its cost until width 2
    is the better trade."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): 1.0, (4, 1): 1.0}, [2, 4], 4))
    _record(ctl, [0])
    assert ctl.select(np.array([0])) == 4
    for _ in range(300):
        ctl.observe_step_time(1, 1, 3.0)
    assert ctl.cost_delta_ms[1] == pytest.approx(2.0, abs=0.05)
    assert ctl.cost_delta_ms[0] == 0.0
    assert ctl.select(np.array([0])) == 2


def test_a_badly_profiled_width_is_corrected_within_a_few_steps():
    """A width's first samples average in fully, so a short run already
    learns that the profile underpriced it (e.g. MoE rows that reach far more
    experts than the profile's dummy tokens)."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): 1.0, (4, 1): 1.0}, [2, 4], 4))
    for ms in (2.5, 2.5, 2.5):
        ctl.observe_step_time(1, 1, ms)
    assert ctl.cost_delta_ms[1] == pytest.approx(1.5)


def test_one_outlier_step_moves_a_settled_cost_a_bounded_amount():
    """Once a width has settled, an interval spanning other work (say a
    prefill) counts as at most 3x the prediction, at the EMA weight."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): 1.0, (4, 1): 1.0}, [2, 4], 4))
    for _ in range(64):
        ctl.observe_step_time(0, 1, 1.0)
    ctl.observe_step_time(0, 1, 1000.0)
    assert ctl.cost_delta_ms[0] == pytest.approx((3.0 - 1.0) / 64)


def test_steps_spanning_other_work_are_not_cost_samples(monkeypatch):
    """An interval covering a prefill or an idle engine (``timed=False``)
    never reaches the width's cost; the next pure tree step does."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    seen: list[int] = []
    monkeypatch.setattr(ctl, "observe_step_time", lambda j, n, ms: seen.append(j))
    ctl._time_step(0, 1)
    ctl._time_step(1, 1, timed=False)  # closes width 0's interval: untimed
    ctl._time_step(0, 1)  # closes width 1's interval: a sample
    for _ in range(2):
        torch.accelerator.synchronize()
        ctl._time_step(0, 1)
    assert seen == [1, 0]


def test_rarely_verified_widths_are_explored():
    """A width ruled out by a misleading early cost still gets verified now
    and then until it has enough samples to be judged."""
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): 1.0, (4, 1): 1.0}, [2, 4], 4))
    ctl.observe_step_time(0, 1, 50.0)  # width 2 looks 50x dearer
    _record(ctl, [0])
    chosen = [ctl.select(np.array([0])) for _ in range(64)]
    assert chosen.count(2) == 2  # every 32nd selection explores it
    assert set(chosen) == {2, 4}


def test_new_request_never_reads_previous_occupant_curve():
    ctl = TreeWidthController([2, 4], max_num_reqs=4, device=DEVICE)
    ctl.set_cost_table(build_cost_table({(2, 1): 0.5, (4, 1): 1.0}, [2, 4], 4))
    _record(ctl, [0])
    ctl.add_request(0)
    # The landed row is the previous occupant's and no curve has been seen
    # before: verify the full drafted tree.
    assert ctl.select(np.array([0])) == 4
    _record(ctl, [0])
    assert ctl.select(np.array([0])) == 2


def _trim(width: int, sched: SchedulerOutput, structured: bool = False):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    sched.has_structured_output_requests = structured
    runner = types.SimpleNamespace(
        _tree_width_ctl=types.SimpleNamespace(select=lambda slots, timed: width),
        req_states=types.SimpleNamespace(req_id_to_index={"a": 0, "b": 1, "c": 2}),
    )
    runner._select_tree_width = types.MethodType(
        GPUModelRunner._select_tree_width, runner
    )
    return GPUModelRunner._tree_trim_verify_width(runner, sched)


def _sched() -> SchedulerOutput:
    sched = SchedulerOutput.make_empty()
    sched.num_scheduled_tokens = {"a": 5, "b": 5, "c": 7}
    sched.total_num_scheduled_tokens = 17
    sched.scheduled_spec_decode_tokens = {"a": [-1] * 4, "b": [-1] * 4}
    return sched


def test_trim_cuts_drafts_without_touching_scheduler_output():
    """Decode requests verify the chosen width; the prefill is untouched, and
    the scheduler's object keeps its full drafts (it rolls the cut ones back
    as rejected)."""
    sched = _sched()
    out = _trim(2, sched)
    assert out.num_scheduled_tokens == {"a": 3, "b": 3, "c": 7}
    assert out.total_num_scheduled_tokens == 13
    assert out.scheduled_spec_decode_tokens == {"a": [-1] * 2, "b": [-1] * 2}
    assert sched.num_scheduled_tokens == {"a": 5, "b": 5, "c": 7}
    assert sched.scheduled_spec_decode_tokens["a"] == [-1] * 4


def test_trim_keeps_full_width_for_structured_output():
    sched = _sched()
    assert _trim(2, sched, structured=True) is sched
