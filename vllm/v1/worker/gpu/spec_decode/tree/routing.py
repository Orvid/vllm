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
embedding and its request's context, the target's last hidden state at the
request's anchor row. On Qwen3.6-35B-A3B at eight requests it reaches a 0.52
expert hit rate and orders rows' marginal expert cost with 0.74 pairwise
accuracy within ~500 steps, against 0.35 and 0.63 for each token's last-seen
routing; the context is what tells requests apart, since a token alone caps
the hit rate near 0.45.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Weight (in predicted experts) of the hit-rate prior, and how often (in
# observed forwards) the hit-rate statistics are halved.
_HIT_PRIOR = 1000.0
_HIT_DECAY_EVERY = 256
_HIDDEN = 512
_LR = 1e-3
# Rows trained on per forward (a random subset past it).
_MAX_TRAIN_ROWS = 512


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

    def predict(self, cand: Tensor, anchor_ids: Tensor) -> Tensor:
        """Fill and return the slot table for candidates ``cand [R, S, K]``
        and anchors ``anchor_ids [R]`` (slot ``cand.numel() + r``)."""
        R = cand.shape[0]
        per_req = cand[0].numel()
        ids = torch.cat([cand.reshape(-1), anchor_ids.reshape(-1).to(cand.dtype)])
        reqs = torch.arange(R, device=cand.device)
        reqs = torch.cat([reqs.repeat_interleave(per_req), reqs])
        n = ids.numel()
        self.reserve_slots(n)
        top = self._topk(ids, reqs)  # [n, num_layers, top_k]
        self.pred[:, :n] = top.permute(1, 0, 2).to(torch.int16)
        return self.pred

    def _topk(self, ids: Tensor, reqs: Tensor) -> Tensor:
        raise NotImplementedError

    def observe_forward(self, input_batch: Any, hidden_states: Tensor | None) -> None:
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
    """Experts predicted by an MLP over (token embedding, request context),
    trained on every forward's decode rows.

    Args:
        embed: The target's input embedding (token ids to hidden states).
        hidden_size: The target's hidden size.
        max_num_reqs: Request-state slots.
        **kwargs: ``MoERoutingPredictor`` arguments.

    """

    def __init__(
        self,
        embed: Callable[[Tensor], Tensor],
        hidden_size: int,
        max_num_reqs: int,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed = embed
        with torch.inference_mode(False):
            self.mlp = _RouterMLP(
                2 * hidden_size, self.num_layers, self.num_experts, _HIDDEN
            ).to(self.device)
            self.opt = torch.optim.Adam(self.mlp.parameters(), lr=_LR)
        # Each request's context (its last hidden state at its anchor row),
        # by request-state slot, and in this step's batch order for predict.
        self.ctx = torch.zeros(max_num_reqs, hidden_size, device=self.device)
        self.batch_ctx = torch.zeros(max_num_reqs, hidden_size, device=self.device)
        self._rng = np.random.default_rng(0)

    def _features(self, ids: Tensor, ctx: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.cat([_rms(self.embed(ids).float()), _rms(ctx)], -1)

    def _topk(self, ids: Tensor, reqs: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.mlp(self._features(ids, self.batch_ctx[reqs]))
        return logits.topk(self.top_k, dim=-1).indices

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

    def observe_forward(self, input_batch: Any, hidden_states: Tensor | None) -> None:
        super().observe_forward(input_batch, hidden_states)
        if hidden_states is None or input_batch.num_tokens == 0:
            return
        rows_np, reqs_np = self._train_rows(input_batch)
        if len(rows_np):
            rows = torch.from_numpy(rows_np).to(self.device, non_blocking=True)
            reqs = torch.from_numpy(reqs_np).to(self.device, non_blocking=True)
            slots = input_batch.idx_mapping[reqs].long()
            self._train(input_batch.input_ids[rows], self.ctx[slots], rows)
        # The next selection prices this batch's candidates against each
        # request's freshest context: its anchor row of this forward.
        num_reqs = input_batch.num_reqs
        anchor_h = hidden_states[input_batch.query_start_loc[:num_reqs]].float()
        self.ctx[input_batch.idx_mapping] = anchor_h
        self.batch_ctx[:num_reqs] = anchor_h

    def _train(self, ids: Tensor, ctx: Tensor, rows: Tensor) -> None:
        labels = self.routes[rows].index_select(1, self.layer_ids).long().clamp_(min=0)
        with torch.inference_mode(False), torch.enable_grad():
            # The forward ran under inference mode; train on ordinary copies.
            feats = self._features(ids, ctx).clone()
            labels = labels.clone()
            target = torch.zeros(
                labels.shape[0], self.num_layers, self.num_experts, device=self.device
            )
            target.scatter_(2, labels, 1.0)
            logits = self.mlp(feats)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                top = logits.topk(self.top_k, dim=-1).indices
                hits = (top[..., :, None] == labels[..., None, :]).any(-1).sum()
                self.hit_stats[0] += hits
                self.hit_stats[1] += top.numel()


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
        max_num_reqs=max_num_reqs,
        routes=capturer.device_buffer,
        moe_layer_ids=[r.layer_id for r in runners],
        num_experts=runners[0].moe_config.num_experts,
        expert_cost_ms=torch.tensor(cost_ms),
        device=device,
    )
