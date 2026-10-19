# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DF2T losslessness harness: traversal verification losslessness.

The verifier must emit, at every visited row, a token drawn from the target's
fully processed distribution at that row (``argmax`` at T=0, one categorical
draw otherwise), regardless of what the drafter proposed. Because a decoded
token's row is always anchored at the previously accepted token, this makes the
tree-decoded next-token distribution identical to the target's.

This harness takes that claim at the empirical level: a synthetic Markov target
(``P``) and an approximate drafter (``Q = P + noise``) run through the REAL
``TreeBuilder`` and the REAL ``TreeVerifier`` in a batched
autoregressive decode loop. It tallies (anchor, emitted) transitions and asserts

  * T=0: every emitted token is bit-identical to the target's argmax at its
    row (invariant 4) and ``KL(tree || target)`` is ~0;
  * production params (T>0, top-p/top-k): ``KL(tree || target) < 1e-3``.

Run as a test::

    .venv/bin/python -m pytest -s tests/evals/df2t_losslessness.py

or as a script with tunable knobs::

    .venv/bin/python tests/evals/df2t_losslessness.py --steps 1000 --replicas 64

Requires CUDA (the tree builder is a Triton kernel). The gate is hard: lower
the sample-count/threshold knobs or skip only if CUDA is truly unavailable;
never weaken ``KL < 1e-3`` to make it pass, because that passes a broken
verifier.
"""

import argparse
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tests.v1.worker.gpu.spec_decode.tree.reference_verifier import _processed
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.tree.selection.df2t import TreeBuilder
from vllm.v1.worker.gpu.spec_decode.tree.tree_verifier import TreeVerifier

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.states import SamplingStates

# Synthetic model knobs (small enough for a tight, fast gate).
VOCAB = 12  # contexts/last-token rows
L = 4  # tree depth cap
K = 8  # drafter candidates per row
B = 24  # emitted nodes per request
R = 96  # parallel sequences per step
STEPS = 2400  # decode steps
MIN_CONTEXT_SAMPLES = 2000
KL_GATE = 1e-3

_DEFAULTS = dict(
    VOCAB=VOCAB,
    L=L,
    K=K,
    B=B,
    R=R,
    STEPS=STEPS,
    MIN_CONTEXT_SAMPLES=MIN_CONTEXT_SAMPLES,
    KL_GATE=KL_GATE,
)


def _tables(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic Markov target ``P`` and approximate drafter ``Q``.

    ``P`` has mild per-row variance so its stationary distribution is near
    uniform (every context gets enough samples); ``Q`` is a noisy copy of ``P``
    so drafts are often wrong and both the accept and mismatch branches (and
    the leaf bonus) are exercised.
    """
    g = torch.Generator().manual_seed(seed)
    p_logits = torch.randn(VOCAB, VOCAB, generator=g) * 0.6
    q_logits = p_logits + torch.randn(VOCAB, VOCAB, generator=g) * 0.4
    return p_logits.to(device), q_logits.to(device)


def _build_step(
    anchors: torch.Tensor,
    q_logits: torch.Tensor,
    builder: TreeBuilder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draft a top-K successor tree per request from the last-token ``Q``.

    ``cand[d]`` is the top-K of ``Q(·|cand[d-1, 0])`` (``cand[0]`` from the
    anchor), and ``scores[d, p, :]`` is ``Q(·|cand[d-1, p])`` restricted to the
    depth-``d`` candidate set, so each ``(depth, predecessor)`` row is a real
    transition mass the lazy best-first builder can price.
    """
    R = anchors.shape[0]
    L, K, dev = builder.L, builder.K, anchors.device
    cand = torch.empty(R, L, K, dtype=torch.long, device=dev)
    cand[:, 0] = torch.topk(q_logits[anchors], K).indices
    for d in range(1, L):
        cand[:, d] = torch.topk(q_logits[cand[:, d - 1, 0]], K).indices

    scores = torch.empty(R, L, K, K, dtype=torch.float32, device=dev)
    scores[:, 0] = q_logits[anchors].gather(-1, cand[:, 0]).unsqueeze(1).expand(R, K, K)
    for d in range(1, L):
        picks = cand[:, d].unsqueeze(1).expand(R, K, K)
        scores[:, d] = q_logits[cand[:, d - 1]].gather(-1, picks)
    tree = builder.build(scores, cand)
    return tree.token_ids, tree.parents, tree.num_nodes, tree.depths


def _target_rows(
    anchors: torch.Tensor,
    token_ids: torch.Tensor,
    num_nodes: torch.Tensor,
    p_logits: torch.Tensor,
) -> torch.Tensor:
    """``[R, 1 + B, V]`` target logits: row 0 = ``P(·|anchor)``, row ``1 + i``
    = ``P(·| token of node i)``. Padding rows (i >= num_nodes) never drawn."""
    R, B = token_ids.shape
    target = torch.zeros(R, 1 + B, VOCAB, dtype=torch.float32, device=p_logits.device)
    target[:, 0] = p_logits[anchors]
    safe = token_ids.clamp(0, VOCAB - 1)
    rows = p_logits[safe]  # [R, B, V]
    valid = torch.arange(B, device=p_logits.device)[None, :] < num_nodes.unsqueeze(-1)
    target[:, 1:] = torch.where(valid[:, :, None], rows, 0.0)
    return target


def _decode(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    builder: TreeBuilder,
    verifier: TreeVerifier,
    greedy: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    vocab: int,
    steps: int,
    replicas: int,
    seed: int,
    prod_ss: "SamplingStates | None" = None,
) -> tuple[torch.Tensor, int]:
    """Batched tree decode loop; returns the (anchor, emitted) count matrix and
    the count of any emitted token that deviates from the target argmax.

    With ``prod_ss`` set, every target row is preprocessed through the real
    production sampling kernels (apply_temperature / apply_min_p /
    apply_top_k_top_p on a per-request SamplingStates), one token is drawn
    per row with the production Gumbel sampler keyed by (request seed, row
    position), and ``TreeVerifier.walk`` traverses the trees — the exact
    architecture ``_tree_verify`` uses in serving — instead of the
    verifier's reference-convention ``_processed`` and per-round draws. Row
    positions follow serving: anchor at ``C``, node at ``C + depth``, and
    ``C`` advances by the emitted token count each step.
    """
    dev = p_logits.device
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(vocab, vocab, dtype=torch.long, device=dev)
    anchors = torch.randint(vocab, (replicas,), generator=g).to(dev)
    bad = torch.zeros((), dtype=torch.long, device=dev)
    argmax_t = (
        torch.stack(
            [_processed(p_logits[c], temperature, top_k, top_p) for c in range(vocab)]
        )
        .argmax(dim=-1)
        .to(dev)
    )
    pos = torch.arange(builder.B + 1, device=dev)[None, :].expand(replicas, -1)
    rr = torch.arange(replicas, device=dev)[:, None].expand(replicas, builder.B + 1)
    ctx = torch.zeros(replicas, dtype=torch.long, device=dev)
    for _ in range(steps):
        token_ids, parents, num_nodes, depths = _build_step(anchors, q_logits, builder)
        target = _target_rows(anchors, token_ids, num_nodes, p_logits)
        if prod_ss is not None:
            rows = target.shape[1]
            flat = target.reshape(replicas * rows, vocab).clone()
            exp_map = (
                torch.arange(replicas * rows, device=dev)
                .div(rows, rounding_mode="floor")
                .to(torch.int32)
            )
            idx_np = np.arange(replicas)
            prod_ss.apply_temperature(flat, exp_map, idx_np)
            prod_ss.apply_min_p(flat, exp_map, idx_np)
            flat = prod_ss.apply_top_k_top_p(flat, exp_map, idx_np)
            row_pos = torch.cat([ctx[:, None], ctx[:, None] + depths], dim=1)
            row_tokens = gumbel_sample(
                flat,
                exp_map,
                prod_ss.temperature.gpu,
                prod_ss.seeds.gpu,
                row_pos.reshape(-1),
                apply_temperature=False,
                is_drafting=False,
            ).view(replicas, rows)
            sampled, num_sampled = verifier.walk(
                row_tokens, token_ids, parents, num_nodes
            )
            ctx += num_sampled.to(torch.long)
        else:
            sampled, num_sampled = verifier(
                target,
                token_ids,
                parents,
                num_nodes,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        active = pos < num_sampled.to(torch.long)[:, None]
        em = sampled[active]
        ra = rr[active]
        pa = pos[active]
        anchor_tok = torch.where(
            pa == 0, anchors[ra], sampled[ra, (pa - 1).clamp(min=0)]
        )
        counts += torch.bincount(anchor_tok * vocab + em, minlength=vocab * vocab).view(
            vocab, vocab
        )
        if greedy:
            bad += (em != argmax_t[anchor_tok]).sum()
        last = (num_sampled - 1).clamp(min=0)
        anchors = sampled[torch.arange(replicas, device=dev), last]
    return counts, int(bad.item())


def _target_tensor(
    p_logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    vocab: int,
    greedy: bool,
) -> torch.Tensor:
    """The per-context next-token distribution of a target-only decoder.

    At T=0 greedy it is one-hot at the processed argmax (bit-identical to the
    verifier's emitted token); otherwise it is the processed full distribution.
    """
    dev = p_logits.device
    processed = torch.stack(
        [_processed(p_logits[c], temperature, top_k, top_p) for c in range(vocab)]
    ).to(dev)
    if not greedy:
        return processed
    target = torch.zeros_like(processed)
    target.scatter_(-1, processed.argmax(-1, keepdim=True), 1.0)
    return target


def _kl_mean(
    counts: torch.Tensor,
    target: torch.Tensor,
    min_samples: int,
) -> tuple[float, int]:
    """Mean per-context ``KL(tree-emitted || target)`` over contexts with at
    least ``min_samples`` transitions."""
    q = counts.to(torch.float32)
    rowsum = q.sum(-1, keepdim=True)
    q = q / rowsum.clamp(min=1)
    eps = 1e-12
    kl = (q * (q / (target + eps)).clamp(min=eps).log()).sum(-1)
    enough = counts.sum(-1) >= min_samples
    if not bool(enough.any()):
        return float("inf"), 0
    return float(kl[enough].mean().item()), int(enough.sum().item())


def _run_gates(**knobs: int | float) -> dict[str, float]:
    """Run both losslessness gates and return the measured KLs and sizes."""
    vocab = int(knobs["VOCAB"])
    r = int(knobs["R"])
    steps = int(knobs["STEPS"])
    min_samples = int(knobs["MIN_CONTEXT_SAMPLES"])
    device = torch.device("cuda")
    p_logits, q_logits = _tables(7, device)
    builder = TreeBuilder(r, int(knobs["L"]), int(knobs["K"]), int(knobs["B"]), device)
    verifier = TreeVerifier(r, int(knobs["B"]), vocab, device)
    # The real production sampling pipeline the serving path routes the verify
    # block through (apply_temperature / apply_min_p / apply_top_k_top_p).
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    prod_ss = SamplingStates(r, vocab)
    prod_ss.temperature.np[:] = float(knobs.get("PROD_TEMPERATURE", 0.9))
    prod_ss.top_k.np[:] = int(knobs.get("PROD_TOP_K", 8))
    prod_ss.top_p.np[:] = float(knobs.get("PROD_TOP_P", 0.95))
    prod_ss.min_p.np[:] = 0.0
    prod_ss.seeds.np[:] = np.arange(r) * 7919 + 17
    prod_ss.seeds.copy_to_uva()
    prod_ss.seeds_set[:] = False
    prod_ss.temperature.copy_to_uva()
    prod_ss.top_k.copy_to_uva()
    prod_ss.top_p.copy_to_uva()
    prod_ss.min_p.copy_to_uva()

    def gate(
        greedy: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        production: bool,
    ) -> tuple[float, float, int, int]:
        counts, bad = _decode(
            p_logits,
            q_logits,
            builder,
            verifier,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            vocab=vocab,
            steps=steps,
            replicas=r,
            seed=vocab + (0 if greedy else 1),
            prod_ss=prod_ss if production else None,
        )
        # T=0 is deterministic (one-hot target), so even the ~R/12 samples a
        # greedy-unreachable context gets are conclusive; the sampling gate
        # needs a few thousand per context to beat finite-sample noise.
        target_min = 1 if greedy else min_samples
        target = _target_tensor(p_logits, temperature, top_k, top_p, vocab, greedy)
        mean_kl, covered = _kl_mean(counts, target, target_min)
        total = int(counts.sum().item())
        return mean_kl, float(bad), total, covered

    kl_t0, bad_t0, total_t0, covered_t0 = gate(True, 1.0, None, None, False)
    kl_raw, bad_raw, total_raw, covered_raw = gate(False, 0.9, 8, 0.95, False)
    kl_tgt, bad_tgt, total_tgt, covered_tgt = gate(False, 0.9, 8, 0.95, True)
    return {
        "VOCAB": vocab,
        "R": r,
        "STEPS": steps,
        "KL_GATE": float(knobs["KL_GATE"]),
        "total_t0": total_t0,
        "covered_t0": covered_t0,
        "kl_t0": kl_t0,
        "bad_t0": int(bad_t0),
        "total_raw": total_raw,
        "covered_raw": covered_raw,
        "kl_raw": kl_raw,
        "total_tgt": total_tgt,
        "covered_tgt": covered_tgt,
        "kl_tgt": kl_tgt,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (triton)")
def test_df2t_losslessness() -> None:
    """Hard gate: bit-identical accepted tokens at T=0 and KL < 1e-3 in both
    the greedy and the production (T>0, top-p/top-k) regimes.

    The T>0 gate runs through the REAL production sampling kernels (via
    ``SamplingStates.apply_temperature/apply_min_p/apply_top_k_top_p``), not
    just the standalone verifier's reference-convention processing, so a
    dropped or wrongly ordered stage in the serving path fails the gate.
    """
    results = _run_gates(**_DEFAULTS)
    print(f"\n  df2t losslessness: {results}")
    assert results["covered_t0"] == results["VOCAB"], "T=0: contexts starved"
    assert results["bad_t0"] == 0, (
        f"T=0: {results['bad_t0']} emitted tokens != target argmax"
    )
    assert results["kl_t0"] < results["KL_GATE"], f"T=0 KL {results['kl_t0']}"
    assert results["covered_raw"] == results["VOCAB"], "raw: contexts starved"
    assert results["kl_raw"] < results["KL_GATE"], f"raw KL {results['kl_raw']}"
    assert results["covered_tgt"] == results["VOCAB"], "prod: contexts starved"
    assert results["kl_tgt"] < results["KL_GATE"], (
        f"production-kernel KL {results['kl_tgt']}"
    )
    # The per-transition KL cannot see a traversal that matches the wrong
    # children (it changes how many tokens a step accepts, not which
    # distribution each emitted token comes from). Both T>0 gates target the
    # same processed distribution, so their emitted-token totals must agree.
    ratio = results["total_tgt"] / results["total_raw"]
    assert abs(ratio - 1.0) < 0.02, (
        f"production path accepts {ratio:.3f}x the reference traversal's tokens"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="DF2T losslessness harness")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--replicas", type=int, default=R)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    args = parser.parse_args()
    knobs = dict(_DEFAULTS)
    knobs.update(STEPS=args.steps, R=args.replicas, VOCAB=args.vocab)
    results = _run_gates(**knobs)
    print("DF2T losslessness:", results)
    ok = (
        results["bad_t0"] == 0
        and results["kl_t0"] < knobs["KL_GATE"]
        and results["kl_raw"] < knobs["KL_GATE"]
        and results["kl_tgt"] < knobs["KL_GATE"]
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _main()
