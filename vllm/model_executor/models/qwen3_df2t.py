# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM

logger = init_logger(__name__)


class DF2TQwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    """DFlash2 draft backbone with the df2t acceptance-confidence head.

    Adds a DSpark-style single-output ``nn.Linear(hidden_size, 1)`` head
    (weights ``df2t_conf_head.*``) that scores each draft position for
    acceptance, gated on ``dflash_config.enable_confidence_head``. The head is
    what the per-request budget policy consumes; it is built lazily from the
    config so a plain DFlash2 checkpoint reuses the exact same loading path.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        self.df2t_conf_head: nn.Linear | None = None
        if draft_config.get("enable_confidence_head", False):
            self.df2t_conf_head = nn.Linear(self.config.hidden_size, 1)

    def compute_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-token acceptance confidence for the df2t draft positions.

        Args:
            hidden_states: Last-layer draft hidden states, shape
                ``[num_tokens, hidden_size]``.

        Returns:
            Per-token confidence, shape ``[num_tokens]``, in float32.

        """
        if self.df2t_conf_head is None:
            raise RuntimeError(
                "df2t_conf_head is None: set dflash_config.enable_confidence_head."
            )
        return self.df2t_conf_head(hidden_states.float()).squeeze(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        head_weights: dict[str, torch.Tensor] = {}
        rest_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            if name.startswith("df2t_conf_head."):
                head_weights[name] = loaded_weight
            else:
                rest_weights.append((name, loaded_weight))

        if head_weights and self.df2t_conf_head is None:
            logger.warning(
                "Checkpoint carries df2t_conf_head weights but "
                "dflash_config.enable_confidence_head is not set; ignoring them."
            )
        elif self.df2t_conf_head is not None:
            expected = frozenset(("df2t_conf_head.weight", "df2t_conf_head.bias"))
            if head_weights.keys() != expected:
                raise RuntimeError(
                    "dflash_config.enable_confidence_head is set but the "
                    "df2t_conf_head weights are incomplete; expected "
                    f"{sorted(expected)}, got {sorted(head_weights)}."
                )

        super().load_weights(rest_weights)
        if head_weights and self.df2t_conf_head is not None:
            self.df2t_conf_head.weight.data.copy_(head_weights["df2t_conf_head.weight"])
            self.df2t_conf_head.bias.data.copy_(head_weights["df2t_conf_head.bias"])


EntryClass = DF2TQwen3ForCausalLM
