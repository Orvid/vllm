# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE routing prediction for expert-aware tree selection.

Verifying a draft tree on a MoE target costs roughly the weights of every
distinct expert its rows route to: a node routed to experts the batch already
uses is nearly free, one that needs new experts is not. Routing is only known
after the verify forward, so the tree builder prices each candidate by the
experts a predictor expects it to route to.

The predictor is a small MLP trained online, never to convergence: every
target forward labels its decode rows with the experts they really routed to
(from the routed-experts capturer), and one optimizer step on them keeps the
predictor tracking the workload. Its inputs are the candidate token's target
embedding and the drafter's hidden state at the candidate's draft position,
which carries the request's context as of that exact position. On
Qwen3.6-35B-A3B with DSpark at eight requests it reaches a 0.52 expert hit
rate and orders rows' marginal expert cost with 0.69 pairwise accuracy within
~300 steps (still rising), against 0.34 and 0.61 for each token's last-seen
routing; a token alone caps the hit rate near 0.46, and the target's hidden
state at the request's anchor row adds less than the drafter's does.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

# Weight (in predicted experts) of the hit-rate prior, and how often (in
# observed forwards) the hit-rate statistics are halved.
_HIT_PRIOR = 1000.0
_HIT_DECAY_EVERY = 256
_HIDDEN = 512
_LR = 1e-3
# Rows trained on per forward (a random subset past it), the pinned staging
# buffers they cycle through, and the padded row counts training runs at.
_MAX_TRAIN_ROWS = 512
_RING = 4
_TRAIN_BUCKETS = (16, 32, 64, 128, 256, _MAX_TRAIN_ROWS)


def _rms(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class MoERoutingPredictor:
    """Predicted experts per candidate token, and what activating an expert
    costs.

    ``predict`` fills a ``[num_layers, slots, top_k]`` table for the tree
    builder: one slot per proposal candidate (request, position, candidate
    index, flattened) followed by one per request anchor. Subclasses supply
    ``_topk``; ``observe_forward`` runs after every target forward.

    Args:
        routes: The routed-experts capturer's device buffer
            ``[max_tokens, num_layers, top_k]``, written by every target
            forward.
        moe_layer_ids: Capturer layer index of each MoE layer, in order.
        num_experts: Routed experts per layer.
        expert_cost_ms: ``[len(moe_layer_ids)]`` default cost of activating one
            expert of each layer (its weight bytes over memory bandwidth).
        device: Device of the predictions.

    """

    def __init__(
        self,
        routes: Tensor,
        moe_layer_ids: list[int],
        num_experts: int,
        expert_cost_ms: Tensor,
        device: torch.device,
    ):
        assert num_experts < 1 << 15
        self.routes = routes
        self.top_k = routes.shape[2]
        self.num_layers = len(moe_layer_ids)
        self.num_experts = num_experts
        self.device = device
        self.layer_ids = torch.tensor(moe_layer_ids, dtype=torch.long, device=device)
        self.pred = torch.zeros(
            self.num_layers, 0, self.top_k, dtype=torch.int16, device=device
        )
        # Cost (ms) of activating each expert; a residency or streaming
        # manager may set it per expert (set_expert_costs).
        self.cost_ms = (
            expert_cost_ms.to(device=device, dtype=torch.float32)[:, None]
            .expand(self.num_layers, num_experts)
            .contiguous()
        )
        self._update_unknown_cost()
        # (predicted experts the row then used, predicted experts): the
        # predictor's hit rate, decayed every _HIT_DECAY_EVERY forwards, from
        # a prior of 0.4.
        self.hit_stats = torch.tensor(
            [0.4 * _HIT_PRIOR, _HIT_PRIOR], dtype=torch.float32, device=device
        )
        self._observed = 0
        # Running total (ms, at the costs above) of the distinct experts each
        # observed forward routed to; the width controller regresses measured
        # step times on it to learn what an expert really costs.
        self._touched = torch.zeros(
            self.num_layers, num_experts, dtype=torch.float32, device=device
        )
        self.forward_cost_total = torch.zeros(1, dtype=torch.float64, device=device)

    def set_expert_costs(self, cost_ms: Tensor) -> None:
        """Per-expert activation cost (ms), ``[num_moe_layers, num_experts]``."""
        assert cost_ms.shape == self.cost_ms.shape
        self.cost_ms.copy_(cost_ms)
        self._update_unknown_cost()

    def _update_unknown_cost(self) -> None:
        # Unpredicted routing activates top_k experts per layer; assume half
        # of them are new (one host sync, only when the costs change).
        self.unknown_cost_ms = float(
            0.5 * self.top_k * self.cost_ms.mean(dim=1).sum().item()
        )

    def reserve_slots(self, slots: int) -> None:
        if self.pred.shape[1] < slots:
            self.pred = torch.zeros(
                self.num_layers,
                slots,
                self.top_k,
                dtype=torch.int16,
                device=self.device,
            )

    def predict(
        self, cand: Tensor, anchor_ids: Tensor, hidden: Tensor | None = None
    ) -> Tensor:
        """Fill and return the slot table for candidates ``cand [R, S, K]``
        and anchors ``anchor_ids [R]`` (slot ``cand.numel() + r``), given the
        drafter's hidden state per draft position (``[R, S, H]``), if any."""
        R, S, K = cand.shape
        ids = torch.cat([cand.reshape(-1), anchor_ids.reshape(-1).to(cand.dtype)])
        pos_hidden = None
        if hidden is not None:
            # Every candidate of a position shares its hidden state; anchors
            # precede the draft and have none.
            pos_hidden = torch.cat(
                [
                    hidden[:, :, None].expand(R, S, K, -1).reshape(R * S * K, -1),
                    hidden.new_zeros(R, hidden.shape[-1]),
                ]
            )
        n = ids.numel()
        self.reserve_slots(n)
        top = self._topk(ids, pos_hidden)  # [n, num_layers, top_k]
        self.pred[:, :n] = top.permute(1, 0, 2).to(torch.int16)
        return self.pred

    def _topk(self, ids: Tensor, hidden: Tensor | None) -> Tensor:
        raise NotImplementedError

    def observe_forward(self, input_batch: Any) -> None:
        """Account the forward's routing (``input_batch.num_tokens`` rows of
        the capturer buffer)."""
        n = input_batch.num_tokens
        if n == 0:
            return
        self._observed += 1
        if self._observed % _HIT_DECAY_EVERY == 0:
            self.hit_stats.mul_(0.5)
        experts = self.routes[:n].index_select(1, self.layer_ids).long()
        self._touched.zero_()
        self._touched.scatter_(
            1, experts.permute(1, 0, 2).reshape(self.num_layers, -1).clamp_(min=0), 1.0
        )
        self.forward_cost_total += (self._touched * self.cost_ms).sum(
            dtype=torch.float64
        )


class _RouterMLP(nn.Module):
    def __init__(self, d_in: int, num_layers: int, num_experts: int, hidden: int):
        super().__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, num_layers * num_experts)
        self.shape = (num_layers, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(nn.functional.gelu(self.l1(x))).view(-1, *self.shape)


class OnlineRoutingPredictor(MoERoutingPredictor):
    """Experts predicted by an MLP over (token embedding, drafter hidden state
    at the token's draft position), trained on every forward's decode rows.

    Args:
        embed: The target's input embedding (token ids to hidden states).
        hidden_size: The target's hidden size.
        draft_hidden_size: The drafter's hidden size (its per-position state
            reads as zeros for a drafter that has none).
        max_num_reqs: Request-state slots.
        **kwargs: ``MoERoutingPredictor`` arguments.

    """

    def __init__(
        self,
        embed: Callable[[Tensor], Tensor],
        hidden_size: int,
        draft_hidden_size: int,
        max_num_reqs: int,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed = embed
        self.draft_hidden_size = draft_hidden_size
        with torch.inference_mode(False):
            self.mlp = _RouterMLP(
                hidden_size + draft_hidden_size,
                self.num_layers,
                self.num_experts,
                _HIDDEN,
            ).to(self.device)
            cuda = self.device.type == "cuda"
            self.opt = torch.optim.Adam(
                self.mlp.parameters(), lr=_LR, fused=cuda, capturable=cuda
            )
        # Training steps by padded row count, each captured at first use.
        self._steps: dict[int, _TrainStep] = {}
        # Training rows reach the device through a ring of pinned buffers: a
        # pageable copy would wait for the forward still running. A slot
        # whose copy is still in flight skips that forward's training.
        pin = is_pin_memory_available()
        self._host_rows = [
            torch.zeros(2, _MAX_TRAIN_ROWS, dtype=torch.int64, pin_memory=pin)
            for _ in range(_RING)
        ]
        self._host_events = [torch.cuda.Event() for _ in range(_RING)]
        self._host_used = [False] * _RING
        self._host_next = 0
        # Each request's latest draft hidden states by request-state slot
        # ([slots, S, H], allocated at the first draft), which label its next
        # verify's rows; and the request-state slot of each row of the batch
        # whose draft comes next.
        self.max_num_reqs = max_num_reqs
        self.draft_hidden: Tensor | None = None
        self._batch_slots = torch.zeros(
            max_num_reqs, dtype=torch.long, device=self.device
        )
        self._rng = np.random.default_rng(0)

    def _features(self, ids: Tensor, hidden: Tensor | None) -> Tensor:
        with torch.no_grad():
            tok = _rms(self.embed(ids).float())
            if hidden is None:
                h = tok.new_zeros(tok.shape[0], self.draft_hidden_size)
            else:
                h = _rms(hidden.float())
            return torch.cat([tok, h], -1)

    def _topk(self, ids: Tensor, hidden: Tensor | None) -> Tensor:
        with torch.no_grad():
            logits = self.mlp(self._features(ids, hidden))
        return logits.topk(self.top_k, dim=-1).indices

    def predict(
        self, cand: Tensor, anchor_ids: Tensor, hidden: Tensor | None = None
    ) -> Tensor:
        if hidden is not None:
            R, S, H = hidden.shape
            if self.draft_hidden is None:
                self.draft_hidden = torch.zeros(
                    self.max_num_reqs, S, H, dtype=hidden.dtype, device=self.device
                )
            self.draft_hidden[self._batch_slots[:R]] = hidden
        return super().predict(cand, anchor_ids, hidden)

    def _train_rows(self, input_batch: Any) -> tuple[np.ndarray, np.ndarray]:
        """Rows of the batch's decode requests and their batch index."""
        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        decode = np.flatnonzero(~input_batch.is_prefilling_np[:num_reqs])
        counts = qsl[decode + 1] - qsl[decode]
        reqs = np.repeat(decode, counts)
        rows = np.arange(len(reqs)) - np.repeat(np.cumsum(counts) - counts, counts)
        rows += np.repeat(qsl[decode], counts)
        if len(rows) > _MAX_TRAIN_ROWS:
            keep = self._rng.choice(len(rows), _MAX_TRAIN_ROWS, replace=False)
            rows, reqs = rows[keep], reqs[keep]
        return rows, reqs

    def observe_forward(self, input_batch: Any) -> None:
        super().observe_forward(input_batch)
        if input_batch.num_tokens == 0:
            return
        rows_np, reqs_np = self._train_rows(input_batch)
        k = self._host_next
        n = len(rows_np)
        if n and (not self._host_used[k] or self._host_events[k].query()):
            self._host_next = (k + 1) % _RING
            host = self._host_rows[k]
            host[0, :n] = torch.from_numpy(rows_np)
            host[1, :n] = torch.from_numpy(reqs_np)
            dev = host[:, :n].to(self.device, non_blocking=True)
            self._host_events[k].record()
            self._host_used[k] = True
            rows, reqs = dev[0], dev[1]
            self._train(input_batch, rows, reqs)
        # The draft that follows this forward is for this batch's requests.
        num_reqs = input_batch.num_reqs
        self._batch_slots[:num_reqs] = input_batch.idx_mapping

    def _row_hidden(
        self, input_batch: Any, rows: Tensor, reqs: Tensor
    ) -> Tensor | None:
        """The draft hidden state each verify row was predicted with: a row
        at depth ``d`` (its position past its request's anchor row) is draft
        position ``d - 1``; the anchor has none."""
        if self.draft_hidden is None:
            return None
        pos = input_batch.positions
        depth = pos[rows] - pos[input_batch.query_start_loc[reqs]]
        S = self.draft_hidden.shape[1]
        slots = input_batch.idx_mapping[reqs].long()
        h = self.draft_hidden[slots, (depth - 1).clamp(0, S - 1)]
        return torch.where(((depth >= 1) & (depth <= S))[:, None], h, 0)

    def _train(self, input_batch: Any, rows: Tensor, reqs: Tensor) -> None:
        n = rows.shape[0]
        bucket = next(b for b in _TRAIN_BUCKETS if b >= n)
        step = self._steps.get(bucket)
        if step is None:
            step = self._steps[bucket] = _TrainStep(self, bucket)
        hidden = self._row_hidden(input_batch, rows, reqs)
        step.load(
            self._features(input_batch.input_ids[rows], hidden),
            self.routes[rows].index_select(1, self.layer_ids),
            n,
        )
        step.run()


class _TrainStep:
    """One optimizer step on up to ``rows`` (padded) rows, replayed as a CUDA
    graph: issued eagerly, the step's few dozen small kernels cost more host
    time than the GPU spends on them."""

    def __init__(self, pred: OnlineRoutingPredictor, rows: int):
        self.pred = pred
        self.rows = rows
        dev = pred.device
        d_in = pred.mlp.l1.in_features
        with torch.inference_mode(False):
            self.feats = torch.zeros(rows, d_in, device=dev)
            self.labels = torch.zeros(
                rows, pred.num_layers, pred.top_k, dtype=torch.long, device=dev
            )
            # 1/n on the n real rows, 0 on the padding.
            self.weight = torch.zeros(rows, device=dev)
        self.graph: torch.cuda.CUDAGraph | None = None

    def load(self, feats: Tensor, labels: Tensor, n: int) -> None:
        self.feats[:n] = feats
        self.feats[n:] = 0
        self.labels[:n] = labels.clamp(min=0)
        self.labels[n:] = 0
        self.weight.fill_(0)
        self.weight[:n] = 1.0 / n

    def _step(self) -> None:
        p = self.pred
        target = torch.zeros(
            self.rows, p.num_layers, p.num_experts, device=self.feats.device
        ).scatter_(2, self.labels, 1.0)
        logits = p.mlp(self.feats)
        per_row = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        ).mean(dim=(1, 2))
        p.opt.zero_grad(set_to_none=True)
        (per_row * self.weight).sum().backward()
        p.opt.step()
        with torch.no_grad():
            top = logits.topk(p.top_k, dim=-1).indices
            hit = (top[..., :, None] == self.labels[..., None, :]).any(-1)
            real = self.weight > 0
            p.hit_stats[0] += (hit.sum(dim=(1, 2)) * real).sum()
            p.hit_stats[1] += real.sum() * (p.num_layers * p.top_k)

    def run(self) -> None:
        with torch.inference_mode(False), torch.enable_grad():
            if self.pred.device.type != "cuda":
                self._step()
                return
            if self.graph is None:
                # Warm up on a side stream (autograd and the optimizer state
                # come up there), then capture.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._step()
                torch.cuda.current_stream().wait_stream(side)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._step()
            self.graph.replay()


def _read_bandwidth_gb_s(device: torch.device) -> float:
    """Streaming read bandwidth (GB/s), the rate expert weights load at."""
    x = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    x.fill_(1)
    v = x.view(torch.float32)
    v.sum()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        v.sum()
    end.record()
    end.synchronize()
    return 5 * x.numel() / 1e9 / (start.elapsed_time(end) / 1e3)


def build_moe_routing_predictor(
    model: torch.nn.Module,
    vllm_config: Any,
    max_num_tokens: int,
    max_num_reqs: int,
    device: torch.device,
) -> OnlineRoutingPredictor:
    """Capture the target's MoE routing every forward, price each layer's
    experts by their weight bytes over the device's read bandwidth, and
    predict routing online.

    Raises:
        ValueError: if the model has no MoE layers, or its routers already
            feed another routing capture (e.g. routing replay).

    """
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
        bind_routed_experts_capturer,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = sorted(
        (m for m in model.modules() if isinstance(m, MoERunner)),
        key=lambda m: m.layer_id,
    )
    if not runners:
        raise ValueError("tree_selection 'df2t_moe' needs a MoE target model.")
    if any(getattr(r.router, "capture_fn", None) is not None for r in runners):
        raise ValueError(
            "tree_selection 'df2t_moe' needs the MoE routers' capture hook, "
            "which another routing capture already uses."
        )
    capturer = RoutedExpertsCapturer(max_num_tokens, vllm_config)
    bind_routed_experts_capturer(model, capturer)

    bandwidth = _read_bandwidth_gb_s(device)
    expert_bytes = [
        sum(p.numel() * p.element_size() for p in r.routed_experts.parameters())
        / r.moe_config.num_local_experts
        for r in runners
    ]
    cost_ms = [b / (bandwidth * 1e9) * 1e3 for b in expert_bytes]
    logger.info(
        "Expert-aware tree selection: %d MoE layers, %.2f MB per expert, "
        "%.0f GB/s read bandwidth (%.1f us per expert activation)",
        len(runners),
        sum(expert_bytes) / len(expert_bytes) / 1e6,
        bandwidth,
        sum(cost_ms) / len(cost_ms) * 1e3,
    )
    return OnlineRoutingPredictor(
        embed=model.embed_input_ids,
        hidden_size=vllm_config.model_config.get_hidden_size(),
        draft_hidden_size=(
            vllm_config.speculative_config.draft_model_config.get_hidden_size()
        ),
        max_num_reqs=max_num_reqs,
        routes=capturer.device_buffer,
        moe_layer_ids=[r.layer_id for r in runners],
        num_experts=runners[0].moe_config.num_experts,
        expert_cost_ms=torch.tensor(cost_ms),
        device=device,
    )
