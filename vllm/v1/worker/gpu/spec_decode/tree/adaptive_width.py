# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verify width for df2t tree speculative decoding.

Every decode step verifies the first ``w`` nodes (best-first, so a valid
prefix tree) of every request's tree, for one ``w`` from a captured ladder. The
choice maximizes expected emitted tokens per unit of step cost:

    w* = argmax_w (R + sum_r A_r(w)) / cost(R, w)

``A_r(w)`` is the expected number of accepted nodes among request ``r``'s first
``w``: the sum of each node's acceptance probability, estimated from its depth
and drafter path log-probability through a binned calibration table fitted
online against the verifier's walk. Depth matters on its own: a parallel
drafter can be equally sure of every position of its main chain while the
chance that the target agrees all the way down decays with depth. Trimming a
tree never changes whether a kept node is accepted (the draws along its path
are the same), so every verified node is an unbiased sample for its bin
whatever width ran. ``cost`` is profiled at startup on the captured uniform
decode graphs and corrected online per width from the measured GPU time
between consecutive tree steps, which covers per-step work outside the graphs
and costs that depend on the data or drift over time.

The host reads the per-request curves one step stale through a pinned ring
it polls without blocking, so the choice never syncs the async pipeline.
"""

import math
from collections import deque

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available

# Calibration bins over the node path log-probability, [-NUM_BINS * BIN_W, 0],
# per node depth (deeper nodes share the last depth's bins).
_NUM_BINS = 64
_BIN_W = 0.25
_NUM_DEPTHS = 16
# Pseudo-observations behind the exp(logp) prior in each bin.
_PRIOR_N = 4.0
# Halve the calibration statistics this often (in observed steps), so the fit
# tracks a drifting workload.
_DECAY_INTERVAL = 512
_RING = 4
# Online step-cost correction: EMA weight per measured step once a width has
# that many samples (half-life ~44 steps, so it follows drifting costs), and
# the cap on a measured interval relative to its prediction.
_COST_ALPHA = 1.0 / 64
_MAX_STEP_RATIO = 3.0


@triton.jit
def _calib_bin(
    lp, depth, NUM_BINS: tl.constexpr, BIN_W: tl.constexpr, ND: tl.constexpr
):
    b = tl.maximum(tl.minimum((-lp / BIN_W).to(tl.int32), NUM_BINS - 1), 0)
    d = tl.maximum(tl.minimum(depth.to(tl.int32) - 1, ND - 1), 0)
    return d * NUM_BINS + b


@triton.jit
def _observe_kernel(
    hits_ptr,
    counts_ptr,
    logp_ptr,  # [max_num_reqs, B] f32 per request state
    depth_ptr,  # [max_num_reqs, B] i32 per request state
    idx_mapping_ptr,
    num_nodes_ptr,  # [R] i64 verified node count
    arenas_ptr,  # [*, arena_stride] i64 accepted arena per depth (col 0 unused)
    arena_stride,
    B: tl.constexpr,
    BP: tl.constexpr,
    AP: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BIN_W: tl.constexpr,
    ND: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
    nn = tl.minimum(tl.load(num_nodes_ptr + r), B)
    node = tl.arange(0, BP)
    live = node < nn
    lp = tl.load(logp_ptr + slot * B + node, mask=live, other=0.0)
    depth = tl.load(depth_ptr + slot * B + node, mask=live, other=1)
    cols = tl.arange(0, AP)
    acc = tl.load(
        arenas_ptr + r * arena_stride + cols,
        mask=(cols >= 1) & (cols < arena_stride),
        other=-1,
    )
    hit = tl.sum((node[:, None] == acc[None, :]).to(tl.float32), axis=1)
    b = _calib_bin(lp, depth, NUM_BINS, BIN_W, ND)
    tl.atomic_add(counts_ptr + b, 1.0, mask=live)
    tl.atomic_add(hits_ptr + b, hit, mask=live)


@triton.jit
def _predict_kernel(
    curves_ptr,  # [max_num_reqs, NL] f32 per request state
    logp_ptr,  # [R, B] f32 built this step
    depth_ptr,  # [R, depth_stride] built this step
    depth_stride,
    idx_mapping_ptr,
    hits_ptr,
    counts_ptr,
    widths_ptr,  # [NL] i32 ascending
    NL: tl.constexpr,
    B: tl.constexpr,
    BP: tl.constexpr,
    NLP: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BIN_W: tl.constexpr,
    ND: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
    node = tl.arange(0, BP)
    lp = tl.load(logp_ptr + r * B + node, mask=node < B, other=float("-inf"))
    depth = tl.load(depth_ptr + r * depth_stride + node, mask=node < B, other=1)
    b = _calib_bin(lp, depth, NUM_BINS, BIN_W, ND)
    p = tl.load(hits_ptr + b) / tl.load(counts_ptr + b)
    p = tl.where(lp > float("-inf"), p, 0.0)
    j = tl.arange(0, NLP)
    w = tl.load(widths_ptr + j, mask=j < NL, other=0)
    curve = tl.sum(tl.where(node[None, :] < w[:, None], p[None, :], 0.0), axis=1)
    tl.store(curves_ptr + slot * NL + j, curve, mask=j < NL)


class TreeWidthController:
    """Chooses each decode step's df2t verify width.

    Args:
        widths: Ascending candidate widths; the last is the drafted ``B``.
        max_num_reqs: Request-state slots.
        device: Device of the trees and the calibration state.

    """

    def __init__(self, widths: list[int], max_num_reqs: int, device: torch.device):
        assert widths == sorted(set(widths)) and widths[0] >= 1
        self.widths = widths
        self.B = widths[-1]
        self.max_num_reqs = max_num_reqs
        self.device = device
        nl = len(widths)
        self._widths_t = torch.tensor(widths, dtype=torch.int32, device=device)

        centers = (np.arange(_NUM_BINS) + 0.5) * _BIN_W
        self.counts = torch.full(
            (_NUM_DEPTHS * _NUM_BINS,), _PRIOR_N, dtype=torch.float32, device=device
        )
        self.hits = torch.tensor(
            np.tile(_PRIOR_N * np.exp(-centers), _NUM_DEPTHS),
            dtype=torch.float32,
            device=device,
        )
        self.logp = torch.full(
            (max_num_reqs, self.B), float("-inf"), dtype=torch.float32, device=device
        )
        self.depths = torch.ones(max_num_reqs, self.B, dtype=torch.int32, device=device)
        self.curves = torch.zeros(max_num_reqs, nl, dtype=torch.float32, device=device)

        pin = is_pin_memory_available()
        self._ring = [
            torch.zeros(max_num_reqs, nl, dtype=torch.float32, pin_memory=pin)
            for _ in range(_RING)
        ]
        self._ring_events = [torch.cuda.Event() for _ in range(_RING)]
        # Step at which each ring slot's copy was launched (-1 = free).
        self._ring_step = [-1] * _RING
        self._ring_next = 0
        self._step = 0
        self._observed = 0
        self.host_curves = np.zeros((max_num_reqs, nl), dtype=np.float64)
        # Step of each slot's first build since its request joined (-1 =
        # none). A landed copy launched at or after it holds that request's
        # latest curve.
        self.first_built_step = np.full(max_num_reqs, -1, dtype=np.int64)
        self._applied_step = -1
        # Running mean per-request curve, standing in for requests whose own
        # curve has not reached the host yet.
        self._mean_curve: np.ndarray | None = None
        # cost_ms[j, n]: profiled step time verifying widths[j] for n requests.
        self.cost_ms: np.ndarray | None = None
        # Online correction per width (ms): the startup profile replays graphs
        # on dummy inputs, so it misses per-step work outside them and costs
        # that depend on the real data and drift over time (e.g. which MoE
        # experts the rows reach and whether they are resident).
        self.cost_delta_ms = np.zeros(nl, dtype=np.float64)
        self.throughput = 0.0
        self._cost_samples = np.zeros(nl, dtype=np.int64)
        self._next_event = torch.cuda.Event(enable_timing=True)
        self._open: tuple[torch.cuda.Event, int, int] | None = None
        self._timed: deque[tuple[torch.cuda.Event, int, int, torch.cuda.Event]] = (
            deque()
        )

    def add_request(self, slot: int) -> None:
        self.first_built_step[slot] = -1

    def observe(
        self,
        idx_mapping: torch.Tensor,
        num_nodes: torch.Tensor,
        accepted_arenas: torch.Tensor,
    ) -> None:
        """Fold this step's walk into the calibration (before the next build
        overwrites the verified trees' path masses)."""
        num_reqs = idx_mapping.shape[0]
        if num_reqs == 0:
            return
        _observe_kernel[(num_reqs,)](
            self.hits,
            self.counts,
            self.logp,
            self.depths,
            idx_mapping,
            num_nodes.contiguous(),
            accepted_arenas,
            accepted_arenas.stride(0),
            B=self.B,
            BP=triton.next_power_of_2(self.B),
            AP=triton.next_power_of_2(accepted_arenas.shape[1]),
            NUM_BINS=_NUM_BINS,
            BIN_W=_BIN_W,
            ND=_NUM_DEPTHS,
        )
        self._observed += 1
        if self._observed % _DECAY_INTERVAL == 0:
            self.hits.mul_(0.5)
            self.counts.mul_(0.5)

    def record_trees(
        self,
        idx_mapping: torch.Tensor,
        slots: np.ndarray,
        logp: torch.Tensor,
        depths: torch.Tensor,
    ) -> None:
        """Keep this step's node path masses and depths (``[R, B]`` each) and
        publish the requests' expected acceptance curves to the host
        (asynchronously)."""
        num_reqs = idx_mapping.shape[0]
        self._step += 1
        if num_reqs == 0:
            return
        first = self.first_built_step[slots]
        self.first_built_step[slots] = np.where(first < 0, self._step, first)
        self.logp[idx_mapping] = logp[:num_reqs]
        self.depths[idx_mapping] = depths[:num_reqs, : self.B].to(torch.int32)
        _predict_kernel[(num_reqs,)](
            self.curves,
            logp,
            depths,
            depths.stride(0),
            idx_mapping,
            self.hits,
            self.counts,
            self._widths_t,
            NL=len(self.widths),
            B=self.B,
            BP=triton.next_power_of_2(self.B),
            NLP=triton.next_power_of_2(len(self.widths)),
            NUM_BINS=_NUM_BINS,
            BIN_W=_BIN_W,
            ND=_NUM_DEPTHS,
        )
        k = self._ring_next
        if self._ring_step[k] >= 0 and not self._ring_events[k].query():
            # Every slot still in flight: skip this publish.
            return
        self._ring_next = (k + 1) % _RING
        self._ring[k].copy_(self.curves, non_blocking=True)
        self._ring_events[k].record()
        self._ring_step[k] = self._step

    def _poll(self) -> None:
        newest, ring_idx = -1, -1
        for k in range(_RING):
            s = self._ring_step[k]
            if s > max(self._applied_step, newest) and self._ring_events[k].query():
                newest, ring_idx = s, k
        if newest < 0:
            return
        self._applied_step = newest
        self.host_curves[:] = self._ring[ring_idx].numpy()

    def set_cost_table(self, cost_ms: np.ndarray) -> None:
        assert cost_ms.shape[0] == len(self.widths)
        self.cost_ms = cost_ms

    def step_cost_ms(self, n: int) -> np.ndarray:
        """Expected step time per width for ``n`` requests: the profiled
        table plus each width's online correction."""
        assert self.cost_ms is not None
        return self.cost_ms[:, min(n, self.cost_ms.shape[1] - 1)] + self.cost_delta_ms

    def _time_step(self, j: int, n: int) -> None:
        """Mark this step's start on the stream; the interval from the
        previous mark is the previous tree step's real GPU time, folded into
        its width's correction once the marks have landed."""
        if self._open is not None:
            self._timed.append((*self._open, self._next_event))
        self._open = (self._next_event, j, n)
        self._next_event.record()
        self._next_event = torch.cuda.Event(enable_timing=True)
        while self._timed and self._timed[0][3].query():
            start, pj, pn, end = self._timed.popleft()
            self.observe_step_time(pj, pn, start.elapsed_time(end))

    def observe_step_time(self, j: int, n: int, ms: float) -> None:
        """Fold a measured step time (``widths[j]``, ``n`` requests) into the
        width's cost correction."""
        predicted = self.step_cost_ms(n)[j]
        # An interval spanning other work (a prefill, an idle engine)
        # overstates this width's cost, so one sample moves the estimate by a
        # bounded amount; a real, larger cost still converges.
        ms = min(ms, _MAX_STEP_RATIO * predicted)
        # A running mean over a width's first samples (so a few steps already
        # correct a badly profiled width), then an EMA that follows drift.
        self._cost_samples[j] += 1
        alpha = max(_COST_ALPHA, 1.0 / self._cost_samples[j])
        self.cost_delta_ms[j] += alpha * (ms - predicted)

    def select(self, slots: np.ndarray) -> int:
        """Verify width for a decode batch over request-state ``slots``."""
        n = len(slots)
        if self.cost_ms is None or n == 0 or len(self.widths) == 1:
            return self.B
        self._poll()
        first = self.first_built_step[slots]
        known = self.host_curves[slots[(first >= 0) & (first <= self._applied_step)]]
        if len(known):
            known_mean = known.mean(axis=0)
            if self._mean_curve is None:
                self._mean_curve = known_mean
            else:
                self._mean_curve += 0.05 * (known_mean - self._mean_curve)
        elif self._mean_curve is None:
            self._time_step(len(self.widths) - 1, n)
            return self.B
        expected = known.sum(axis=0) + self._mean_curve * (n - len(known))
        rate = (n + expected) / self.step_cost_ms(n)
        j = int(np.argmax(rate))
        # Expected tokens per ms at the chosen width: what a ms of verify cost
        # is worth in tokens (expert-aware tree selection prices with it).
        self.throughput = float(rate[j])
        self._time_step(j, n)
        return self.widths[j]


def default_profile_request_counts(max_num_reqs: int) -> list[int]:
    """Request counts the startup cost profile samples (dense where batches
    are small, geometric after)."""
    counts = list(range(1, min(8, max_num_reqs) + 1))
    n = 8
    while n < max_num_reqs:
        n = min(max_num_reqs, int(math.ceil(n * 1.5)))
        counts.append(n)
    return sorted(set(counts))


def build_cost_table(
    samples: dict[tuple[int, int], float],
    widths: list[int],
    max_num_reqs: int,
) -> np.ndarray:
    """Dense ``[len(widths), max_num_reqs + 1]`` step-cost table from profiled
    ``(width, num_reqs) -> ms`` samples, linear between sampled counts."""
    table = np.zeros((len(widths), max_num_reqs + 1), dtype=np.float64)
    n_all = np.arange(max_num_reqs + 1)
    for j, w in enumerate(widths):
        pts = sorted((n, ms) for (ww, n), ms in samples.items() if ww == w)
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.maximum.accumulate(np.array([p[1] for p in pts], dtype=np.float64))
        table[j] = np.interp(n_all, xs, ys)
        if len(xs) > 1:
            after = n_all > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            table[j, after] = ys[-1] + slope * (n_all[after] - xs[-1])
    # A wider verify never costs less; profiling noise must not say otherwise.
    return np.maximum(np.maximum.accumulate(table, axis=0), 1e-3)
