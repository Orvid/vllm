# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator for this config."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_tree():
        # Tree verification composes onto the drafter: the drafter proposes
        # candidates, the configured selection method shapes the tree.
        from vllm.v1.worker.gpu.spec_decode.tree.speculator import (
            tree_speculator_class,
        )

        drafter_cls = _speculator_class(vllm_config)
        return tree_speculator_class(drafter_cls)(vllm_config, device)
    return _speculator_class(vllm_config)(vllm_config, device)


def _speculator_class(vllm_config: VllmConfig) -> type:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator
    elif speculative_config.method == "dflash":
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import (
                LiLiCorrSpeculator,
            )

            return LiLiCorrSpeculator
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator
    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator
    elif speculative_config.use_multi_module_mtp():
        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        return MultiModuleMTPSpeculator
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    elif speculative_config.uses_draft_model():
        from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        return StandaloneARSpeculator(vllm_config, device)
    elif speculative_config.use_ngram():
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
            NgramGPUSpeculator,
        )

        return NgramGPUSpeculator(vllm_config, device, req_states)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
