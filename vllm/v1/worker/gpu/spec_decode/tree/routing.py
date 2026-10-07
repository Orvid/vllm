# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-keyed MoE routing memory for expert-aware tree selection.

Verifying a draft tree on a MoE target costs roughly the weights of every
distinct expert its rows route to: a node routed to experts the batch already
uses is nearly free, one that needs new experts is not. Routing is only known
after the verify forward, so each node's experts are predicted from the routing
its token got the last time it was seen: every target forward records its
rows' top-k experts per MoE layer (from the routed-experts capturer) into a
table keyed by a hash of the token id.

On Qwen3.6-35B-A3B the last-seen routing orders verify rows' marginal expert
cost with 0.70 pairwise accuracy at one request (0.61 at eight; chance is
0.5). Hashing tokens into 4096 buckets loses almost none of that, while
keeping fewer layers or fewer experts per token loses much of it, so the table
keeps every MoE layer and the full top-k: ``layers * buckets * top_k`` int16.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

DEFAULT_BUCKETS = 4096
# Weight (in predicted experts) of the hit-rate prior, and how often (in
# recorded forwards) the hit-rate statistics are halved.
_HIT_PRIOR = 1000.0
_HIT_DECAY_EVERY = 256
# Knuth's multiplicative hash; buckets are a power of two.
_HASH_MULT = 2654435761


@triton.jit
def routing_bucket(tok, H: tl.constexpr):
    return (tok.to(tl.int64) * 2654435761) & (H - 1)


@triton.jit
def _record_kernel(
    table_ptr,  # [NL, H, KE] i16
    ids_ptr,  # [N] token ids
    routes_ptr,  # capturer buffer [max_tokens, num_layers, KE]
    route_stride_tok,
    route_stride_layer,
    layer_ids_ptr,  # [NL] capturer layer index of each MoE layer
    stats_ptr,  # [2] f32 (predicted experts used, predicted experts)
    H: tl.constexpr,
    NL: tl.constexpr,
    KE: tl.constexpr,
    NLP: tl.constexpr,
    KEP: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    b = routing_bucket(tl.load(ids_ptr + i), H)
    nl = tl.arange(0, NLP)
    kj = tl.arange(0, KEP)
    mask = (nl[:, None] < NL) & (kj[None, :] < KE)
    src_layer = tl.load(layer_ids_ptr + nl, mask=nl < NL, other=0).to(tl.int64)
    experts = tl.load(
        routes_ptr
        + i * route_stride_tok
        + src_layer[:, None] * route_stride_layer
        + kj[None, :],
        mask=mask,
        other=-1,
    )
    slot = table_ptr + (nl[:, None] * H + b) * KE + kj[None, :]
    # Score the entry being replaced as a prediction of this routing: how many
    # of the experts it predicted the token really used.
    old = tl.load(slot, mask=mask, other=-1).to(tl.int32)
    predicted = mask & (old >= 0)
    used = tl.zeros([NLP, KEP], dtype=tl.int32)
    for j in tl.static_range(KE):
        e_j = tl.sum(tl.where(kj[None, :] == j, experts, 0), axis=1)
        used += (old == e_j[:, None]).to(tl.int32)
    hits = tl.sum(tl.sum(tl.where(predicted, used, 0), axis=1), axis=0)
    total = tl.sum(tl.sum(predicted.to(tl.int32), axis=1), axis=0)
    if total > 0:
        tl.atomic_add(stats_ptr, hits.to(tl.float32))
        tl.atomic_add(stats_ptr + 1, total.to(tl.float32))
    tl.store(slot, experts.to(tl.int16), mask=mask)


class MoERoutingMemory:
    """Last-seen routing per (MoE layer, token bucket), and what activating an
    expert costs.

    Args:
        routes: The routed-experts capturer's device buffer
            ``[max_tokens, num_layers, top_k]``, written by every target
            forward.
        moe_layer_ids: Capturer layer index of each MoE layer, in order.
        num_experts: Routed experts per layer.
        expert_cost_ms: ``[len(moe_layer_ids)]`` default cost of activating one
            expert of each layer (its weight bytes over memory bandwidth).
        device: Device of the table.
        buckets: Token-hash buckets (a power of two).

    """

    def __init__(
        self,
        routes: torch.Tensor,
        moe_layer_ids: list[int],
        num_experts: int,
        expert_cost_ms: torch.Tensor,
        device: torch.device,
        buckets: int = DEFAULT_BUCKETS,
    ):
        assert buckets & (buckets - 1) == 0, buckets
        assert num_experts < 1 << 15
        self.routes = routes
        self.top_k = routes.shape[2]
        self.num_layers = len(moe_layer_ids)
        self.num_experts = num_experts
        self.buckets = buckets
        self.layer_ids = torch.tensor(moe_layer_ids, dtype=torch.int32, device=device)
        self.table = torch.full(
            (self.num_layers, buckets, self.top_k),
            -1,
            dtype=torch.int16,
            device=device,
        )
        # Cost (ms) of activating each expert; a residency or streaming
        # manager may set it per expert (set_expert_costs).
        self.cost_ms = (
            expert_cost_ms.to(device=device, dtype=torch.float32)[:, None]
            .expand(self.num_layers, num_experts)
            .contiguous()
        )
        self._update_unknown_cost()
        # (predicted experts the token then used, predicted experts): the
        # predictor's hit rate, measured whenever a token's entry is replaced
        # and decayed every _HIT_DECAY_EVERY records, from a prior near what
        # Qwen3.6-35B-A3B measures (0.42).
        self.hit_stats = torch.tensor(
            [0.4 * _HIT_PRIOR, _HIT_PRIOR], dtype=torch.float32, device=device
        )
        self._records = 0

    def set_expert_costs(self, cost_ms: torch.Tensor) -> None:
        """Per-expert activation cost (ms), ``[num_moe_layers, num_experts]``."""
        assert cost_ms.shape == self.cost_ms.shape
        self.cost_ms.copy_(cost_ms)
        self._update_unknown_cost()

    def _update_unknown_cost(self) -> None:
        # A token never seen activates top_k experts per layer; assume half of
        # them are new (one host sync, only when the costs change).
        self.unknown_cost_ms = float(
            0.5 * self.top_k * self.cost_ms.mean(dim=1).sum().item()
        )

    def record(self, input_ids: torch.Tensor, num_tokens: int) -> None:
        """Fold the last target forward's routing (``num_tokens`` rows of the
        capturer buffer, whose tokens are ``input_ids``) into the table."""
        if num_tokens == 0:
            return
        self._records += 1
        if self._records % _HIT_DECAY_EVERY == 0:
            self.hit_stats.mul_(0.5)
        _record_kernel[(num_tokens,)](
            self.table,
            input_ids,
            self.routes,
            self.routes.stride(0),
            self.routes.stride(1),
            self.layer_ids,
            self.hit_stats,
            H=self.buckets,
            NL=self.num_layers,
            KE=self.top_k,
            NLP=triton.next_power_of_2(self.num_layers),
            KEP=triton.next_power_of_2(self.top_k),
        )


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


def build_moe_routing_memory(
    model: torch.nn.Module,
    vllm_config: Any,
    max_num_tokens: int,
    device: torch.device,
) -> MoERoutingMemory:
    """Capture the target's MoE routing every forward and price each layer's
    experts by their weight bytes over the device's read bandwidth.

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
    return MoERoutingMemory(
        capturer.device_buffer,
        [r.layer_id for r in runners],
        runners[0].moe_config.num_experts,
        torch.tensor(cost_ms),
        device,
    )
