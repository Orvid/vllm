# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DF2T tree budget: ``s`` resolution and the per-request ``(L, B)`` shape.

Two pure operators feed the DF2T runtime:

- ``resolve_nodes_max`` maps the deployment-scale operator ``s`` and the
  node-count ``policy`` to a concrete tree budget ``nodes_max``, clamped to
  ``[floor, ceiling]``. ``s == 1.0`` reproduces the policy exactly.
- ``shape`` turns that budget (or a confidence-head output ``c_t``) into the
  tree shape ``(L, B)``. Without a head the shape is static: ``B =
  nodes_max`` at the full depth ``L = max_tree_depth``, degrading to a chain
  ``(L, B) = (nodes_max, nodes_max)`` when the budget falls below the depth.
  With a head, ``B`` is bucketed from ``c_t`` to the configured budget set
  while ``L`` stays at max depth.

``BudgetPolicy`` bundles both operators with the ``SpeculativeConfig`` knobs.
``s`` is uniform per deployment in this cut (the per-request tensor interface
is deferred to the bucketed-graph milestone); callers must assert uniformity
before passing a scalar ``s``.
"""

import math
from dataclasses import dataclass
from typing import Any

_DEFAULT_FLOOR = 1
"""Minimum node budget when no asset-driven bound is configured."""

_HEAD_SCALE_FACTOR = 4.0
"""Placeholder confidence-head width scale.

No head-bearing checkpoint exists yet to calibrate against. This is the
documented v5-consistent default (`B = clamp(round(max_depth * sigmoid(c_t) *
scale_factor), floor, ceiling)` snapped to the budget set) and MUST be
recalibrated once a real head ships.
"""

_HEAD_BUDGETS = (4, 8, 16, 32, 64)
"""Configured budget set for bucketing the head-derived width."""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bucket(raw: float, budgets: tuple[int, ...]) -> int:
    """Nearest budget-set value to ``raw``; ties resolve upward so the
    mapping stays monotone non-decreasing in ``raw``/``c_t``."""
    return min(budgets, key=lambda b: (abs(b - raw), -b))


def resolve_nodes_max(s: float, policy: int, ceiling: int, floor: int) -> int:
    """Tree node budget ``nodes_max`` from operator scale and policy.

    Args:
        s: Operator compute scale, uniform per deployment. ``1.0``
            reproduces ``policy`` exactly.
        policy: Node-count policy (``tree_budget``) at ``s == 1.0``.
        ceiling: Hard cap on ``nodes_max`` (``nodes_ceiling``).
        floor: Lower bound on ``nodes_max`` (``nodes_floor``).

    Returns:
        ``round(s * policy)`` clamped to ``[floor, ceiling]``.

    """
    return max(floor, min(ceiling, round(s * policy)))


def shape(
    nodes_max: int,
    c_t: float | None,
    max_depth: int,
    floor: int = _DEFAULT_FLOOR,
    ceiling: int = _HEAD_BUDGETS[-1],
    scale_factor: float = _HEAD_SCALE_FACTOR,
) -> tuple[int, int]:
    """Per-request tree shape ``(L, B)`` with the degradation order.

    Args:
        nodes_max: Static budget from ``resolve_nodes_max`` (no-head path).
        c_t: Per-request confidence-head logit, or ``None`` for the static
            path.
        max_depth: Tree depth cap ``L`` (``max_tree_depth``).
        floor: Lower bound for the head-derived width.
        ceiling: Upper bound for the head-derived width.
        scale_factor: Placeholder head width scale (see ``_HEAD_SCALE_FACTOR``).

    Returns:
        ``(L, B)``. ``L == B == nodes_max`` (a chain) when the budget falls
        below ``max_depth``; otherwise ``L == max_depth``.

    """
    if c_t is None:
        b = nodes_max
    else:
        raw = max_depth * _sigmoid(c_t) * scale_factor
        b = max(floor, min(ceiling, _bucket(raw, _HEAD_BUDGETS)))
    if b >= max_depth:
        return max_depth, b
    return b, b


@dataclass(frozen=True)
class BudgetPolicy:
    """Bundled DF2T budget knobs and operators.

    Args:
        s: Operator compute scale (uniform per deployment this cut).
        policy: Node-count policy at ``s == 1.0`` (``tree_budget``).
        ceiling: Cap on ``nodes_max``/``B`` (``nodes_ceiling``).
        max_depth: Tree depth cap ``L`` (``max_tree_depth``).
        floor: Lower bound on ``nodes_max``/``B`` (``nodes_floor``).
        scale_factor: Placeholder head width scale.

    """

    policy: int
    ceiling: int
    max_depth: int
    s: float = 1.0
    floor: int = _DEFAULT_FLOOR
    scale_factor: float = _HEAD_SCALE_FACTOR

    def __post_init__(self) -> None:
        if self.floor > self.ceiling:
            raise ValueError(
                "nodes_floor must not exceed nodes_ceiling: "
                f"floor={self.floor} > ceiling={self.ceiling}"
            )

    @property
    def nodes_max(self) -> int:
        """Deployment tree budget at the configured ``s``."""
        return resolve_nodes_max(self.s, self.policy, self.ceiling, self.floor)

    def resolve_nodes_max(self) -> int:
        return self.nodes_max

    def shape(self, c_t: float | None) -> tuple[int, int]:
        return shape(
            self.nodes_max,
            c_t,
            self.max_depth,
            floor=self.floor,
            ceiling=self.ceiling,
            scale_factor=self.scale_factor,
        )

    @classmethod
    def from_config(cls, d: Any) -> "BudgetPolicy":
        """Build from the ``SpeculativeConfig`` df2t tree knobs.

        Args:
            d: Object exposing ``tree_s``, ``tree_budget``, ``nodes_ceiling``
                and ``max_tree_depth`` (a resolved ``SpeculativeConfig``).

        """
        return cls(
            s=d.tree_s,
            policy=d.tree_budget,
            ceiling=d.nodes_ceiling,
            max_depth=d.max_tree_depth,
        )
