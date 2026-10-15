# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 output projection through b12x's fused MXFP8 WO kernels.

One b12x chain replaces inverse RoPE + FP8 quant, the grouped ``wo_a`` einsum
and ``wo_b`` (``b12x.gemm.wo_projection``, SM12x). Weights are packed once from
the checkpoint's 128x128 block scales, before the linear methods rewrite them.
Plans are shared by every layer: an exact-count plan per CUDA-graph capture
size and one dynamic plan for every other count. All of them are prepared by
the b12x warmup, before any graph is captured: preparing a plan later can
reclaim b12x programs that captured graphs still launch.
"""

from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils.b12x import B12xWarmupUnit, has_b12x

logger = init_logger(__name__)

# (device, num_tokens or None for dynamic, dims, cos_sin dtype) -> (plan, scratch)
_PLANS: dict[tuple, tuple[Any, torch.Tensor]] = {}


def _wo_module():
    from b12x.gemm import wo_projection

    return wo_projection


def b12x_wo_unsupported_reason(layer: nn.Module) -> str | None:
    if not envs.VLLM_USE_B12X_WO_PROJECTION:
        return "VLLM_USE_B12X_WO_PROJECTION is off"
    if not has_b12x():
        return "b12x is not installed"
    if not _wo_module().is_supported():
        return "b12x WO projection is unsupported on this device"
    for name in ("wo_a", "wo_b"):
        linear = getattr(layer, name)
        if linear.weight.dtype != torch.float8_e4m3fn:
            return f"{name} is not FP8"
        if not hasattr(linear, "weight_scale_inv"):
            return f"{name} has no block scales"
    return None


class B12xWoProjection:
    """Packed WO weights of one attention layer, run through shared plans."""

    def __init__(self, layer: nn.Module) -> None:
        self.groups = layer.n_local_groups
        self.heads_per_group = layer.n_local_heads // layer.n_local_groups
        self.group_width = self.heads_per_group * layer.head_dim
        self.rank = layer.o_lora_rank
        self.hidden = layer.hidden_size
        self.nope_dim = layer.nope_head_dim
        self.rope_dim = layer.rope_head_dim
        self.reduce = layer.wo_b.reduce_results and layer.wo_b.tp_size > 1
        self.weights = _wo_module().pack_weights(
            layer.wo_a.weight.detach(),
            layer.wo_a.weight_scale_inv.detach(),
            layer.wo_b.weight.detach(),
            layer.wo_b.weight_scale_inv.detach(),
            groups=self.groups,
            group_width=self.group_width,
            rank=self.rank,
            hidden=self.hidden,
        )
        config = get_current_vllm_config()
        self.exact_sizes = frozenset(
            config.compilation_config.cudagraph_capture_sizes or ()
        )
        self.max_dynamic_tokens = config.scheduler_config.max_num_batched_tokens
        self.checked = False

    def _plan(
        self, num_tokens: int, cos_sin_cache: torch.Tensor
    ) -> tuple[Any, torch.Tensor]:
        exact = num_tokens in self.exact_sizes
        dims = (self.groups, self.group_width, self.rank, self.hidden)
        key = (
            cos_sin_cache.device,
            num_tokens if exact else None,
            dims,
            cos_sin_cache.dtype,
        )
        entry = _PLANS.get(key)
        if entry is not None:
            return entry
        wo = _wo_module()
        plan = wo.plan(
            wo.Caps(
                device=cos_sin_cache.device,
                max_tokens=num_tokens if exact else self.max_dynamic_tokens,
                groups=self.groups,
                group_width=self.group_width,
                rank=self.rank,
                hidden=self.hidden,
                dtype=torch.bfloat16,
            ),
            invocation={
                "operation": "inv_rope",
                "heads_per_group": self.heads_per_group,
                "nope_dim": self.nope_dim,
                "rope_dim": self.rope_dim,
                "positions_dtype": "int64",
                "cos_sin_dtype": str(cos_sin_cache.dtype).removeprefix("torch."),
                "dynamic_tokens": not exact,
            },
        )
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=cos_sin_cache.device)
        _PLANS[key] = entry = (plan, scratch)
        return entry

    def _run(
        self, o: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        """The local (unreduced) projection, written into shared scratch."""
        wo = _wo_module()
        plan, scratch = self._plan(o.shape[0], cos_sin_cache)
        binding = wo.bind_inv_rope(
            plan,
            scratch=scratch,
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=self.weights,
            heads_per_group=self.heads_per_group,
            nope_dim=self.nope_dim,
            rope_dim=self.rope_dim,
        )
        return wo.run_inv_rope(binding=binding, plan=plan)

    def __call__(
        self, o: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        if o.shape[0] == 0:
            return o.new_empty((0, self.hidden))
        out = self._run(o, positions, cos_sin_cache)
        # The output lives in scratch the next layer reuses.
        return tensor_model_parallel_all_reduce(out) if self.reduce else out.clone()

    def get_b12x_warmup_unit(
        self,
        layer: nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        """Prepare every plan this projection can use (all layers share them)."""
        cos_sin_cache = layer.rotary_emb.cos_sin_cache
        device = cos_sin_cache.device
        # Every exact plan, plus one count outside the capture sizes for the
        # dynamic plan.
        counts = sorted(self.exact_sizes)
        dynamic = max(counts, default=0) + 1
        if dynamic <= self.max_dynamic_tokens:
            counts.append(dynamic)

        def compile() -> None:
            for num_tokens in counts:
                o = torch.zeros(
                    (num_tokens, layer.n_local_heads, layer.head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
                self._run(o, positions, cos_sin_cache)

        return B12xWarmupUnit(
            name="WO projection",
            key=(
                type(self),
                cos_sin_cache.device,
                cos_sin_cache.dtype,
                self.groups,
                self.group_width,
                self.rank,
                self.hidden,
            ),
            compile=compile,
        )

    def check(self, out: torch.Tensor, reference: torch.Tensor, prefix: str) -> None:
        """Log one comparison against the DeepGEMM path (eager, nonzero input).

        ``out`` must be a copy: a reduced output may share the all-reduce
        buffer the reference path writes next.
        """
        ref_norm = reference.float().norm().item()
        if ref_norm == 0.0:
            return
        self.checked = True
        rel = (out.float() - reference.float()).norm().item() / ref_norm
        log = logger.warning if rel > 5e-2 else logger.info
        log(
            "b12x WO projection %s: relative error %.3e vs DeepGEMM over %d tokens",
            prefix,
            rel,
            out.shape[0],
        )
