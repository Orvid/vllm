# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-mixer-family tree verification support.

Tree verification is generic over mixer families, not models. A family takes
part through two hooks:

- **verify**: during the verify forward a layer reads the batch's
  ``ForwardContext.tree_verify`` (:class:`TreeVerifyContext`) and computes
  every row as a continuation of its parent row, leaving the committed state
  of tree requests untouched.
- **commit**: after the walk, the accepted path (:class:`TreeCommit`) becomes
  part of each request's sequence.

Softmax attention needs no layer code: the runner applies the tree mask and
relocates the accepted KV (``TreeKVCompaction``). Recurrent layers (GDN,
Mamba2, short conv, ...) implement :class:`TreeRecurrentLayer`: they stash
their per-row inputs during verify and replay the committed rows through their
own chain kernels from the request's state slot, so the commit reuses each
family's existing state-update path.
"""

from typing import Any, Protocol, runtime_checkable

import torch

from vllm.v1.worker.gpu.spec_decode.tree.context import TreeCommit


@runtime_checkable
class TreeRecurrentLayer(Protocol):
    """A stateful mixer that supports tree verification."""

    def tree_stash_spec(self) -> tuple[int, torch.dtype]:
        """Columns and dtype of the per-row inputs the commit replays."""
        ...

    def bind_tree_stash(self, stash: torch.Tensor) -> None:
        """Receive the persistent ``[max_rows, columns]`` stash that the
        verify forward writes row ``i``'s inputs to."""
        ...

    def tree_commit(self, commit: TreeCommit, attn_metadata: Any) -> None:
        """Advance each tree request's state over its committed rows (read
        from the stash), using this step's ``attn_metadata`` for the layer's
        state slots."""
        ...


class TreeRecurrentLayers:
    """The model's tree-capable recurrent layers and their row stashes.

    Stashes cover the batch's leading ``max_rows`` rows, which hold every
    tree request (verify requests lead the batch).

    Args:
        layers: Layer name -> layer, for every stateful (non-attention)
            mixer of the target model.
        max_rows: Rows each stash holds.
        device: Device of the stashes.

    Raises:
        ValueError: if a stateful layer does not implement
            :class:`TreeRecurrentLayer`.

    """

    def __init__(
        self, layers: dict[str, Any], max_rows: int, device: torch.device
    ) -> None:
        unsupported = [
            name
            for name, layer in layers.items()
            if not isinstance(layer, TreeRecurrentLayer)
        ]
        if unsupported:
            raise ValueError(
                "tree verification needs every stateful mixer to support it, "
                f"but {type(layers[unsupported[0]]).__name__} layers "
                f"{unsupported[:4]} do not (TreeRecurrentLayer)."
            )
        self.layers: dict[str, TreeRecurrentLayer] = layers
        self.max_rows = max_rows
        for layer in self.layers.values():
            columns, dtype = layer.tree_stash_spec()
            layer.bind_tree_stash(
                torch.empty(max_rows, columns, dtype=dtype, device=device)
            )

    def commit(self, commit: TreeCommit, attn_metadata: Any) -> None:
        for layer in self.layers.values():
            layer.tree_commit(commit, attn_metadata)
