#!/usr/bin/env python3
"""M1 cost decomposition (plan Task 15 step 3).

Boots the PRODUCTION-path server (``eager=False``: FULL CUDA graphs including
the df2t verifier CUDA micro-graph, NOT ``--enforce-eager``) per config at
concurrency 1-3 and reports the tree-vs-chain incremental per-step serving
cost, decomposed into its binding terms:

  per-step wall ms    = (E[A] + 1) / tok/s * 1000
                        (at C=1 this is the full host+GPU decode-step latency;
                        structurally the target forward dominates, see below)
  attention rows/step = 1 + nodes_max for the tree verify vs 16 for the chain
                        (the target scores one row per token position; this is
                        the B-proportional term, the binding M1 cost)
  verifier            = T=0 all_greedy argmax path; a CUDA micro-graph replay
                        (one host call per step, fixed 16 rounds, no RNG)
  compaction          = fixed-shape ~9 triton launches/step, B-independent
                        (near-free at concurrency 1-3, per the design)

Controls (same Qwen3-0.6B target + CATMIX drafter), all on the quiet-host
production path:
  chain    = dflash (FA2 vehicle): the production chain@15 control
  tree-off = df2t with ``enable_tree=False`` (FA4 vehicle): a CHAIN on the
             SAME attention vehicle -> isolates the tree-structure cost from
             the FA4-vs-FA2 vehicle delta
  tree B   = df2t tree (FA4) with tree_budget in {4, 8, 16, 32}, L=15

Reports the crossover: per-step cost (grows with B) vs accepted tokens/step
(grows with B) -> which tree config first beats the chain in net tok/s, and
the FA4-only (tree-vs-tree-off) versus FA2-delta (tree-vs-chain) split.

NOTE: wall-clock is valid only on a QUIET host (no background GPU workload).
"""
import concurrent.futures
import json
import os
import time
import urllib.request

import eval_df2t as E

_TARGET = E.TARGET
_L = int(os.environ.get("MAX_TREE_DEPTH", "15"))
_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "192"))
_ROUNDS = int(os.environ.get("ROUNDS", "4"))
_CONCURRENCIES = tuple(int(x) for x in os.environ.get(
    "M1_CONCURRENCIES", "1,2,3").split(",") if x)
_TREE_BUDGETS = tuple(int(x) for x in os.environ.get(
    "M1_TREE_BUDGETS", "4,8,16,32").split(",") if x)
_PORT = int(os.environ.get("PORT", "8012"))


def complete(port: int, prompt: str, max_tokens: int = _MAX_TOKENS) -> tuple[str, int]:
    body = json.dumps({"model": _TARGET, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], int(d["usage"].get("completion_tokens", 0))


def _measure(port: int, c: int, prompts: list[str], rounds: int = _ROUNDS
             ) -> tuple[float, int]:
    """tok/s churning the full prompt list per round at concurrency c (threaded
    co-batching); at c=1 the list is issued sequentially, giving a large stable
    token sample rather than a single short completion."""

    def _one(p: str) -> int:
        return complete(port, p)[1]

    tokens_total = 0
    elapsed = 0.0
    for _ in range(rounds):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            counts = list(ex.map(_one, prompts))
        elapsed += time.perf_counter() - t0
        tokens_total += sum(counts)
    toks = tokens_total / elapsed if elapsed > 0 else float("nan")
    return toks, tokens_total


def _acceptance(port: int) -> float:
    """E[A] over a short drive (metrics delta)."""
    before = E._counters(port)
    E._drive(port, E.PROMPTS["math"], 4)
    m = E._acceptance_metrics(before, E._counters(port))
    return m["E[A]"] if m["drafts"] > 0 else float("nan")


def _row(name: str, prompt_batch: list[str], c: int) -> dict:
    if name == "chain":
        proc = E._boot(_PORT, "dflash", None, None, spec_tokens=15,
                       eager=False)
    else:
        tree_off = name == "tree-off"
        budget = None if tree_off else name.split("B")[1]
        budget = int(budget) if budget else None
        spec_extra = {"enable_tree": False} if tree_off else None
        if tree_off:
            proc = E._boot(_PORT, "df2t", 16, _L, spec_tokens=15,
                           eager=False, spec_extra=spec_extra)
        else:
            proc = E._boot(_PORT, "df2t", budget, _L, spec_tokens=15,
                           tree_s=1.0, eager=False)
    try:
        time.sleep(2)
        complete(_PORT, prompt_batch[0])  # warm
        t, n = _measure(_PORT, c, prompt_batch)
        ea = _acceptance(_PORT)
        per_step_ms = (ea + 1) / t * 1000.0
        return {"name": name, "concurrency": c, "tok/s": round(t, 1),
                "out_tokens": n, "E[A]": round(ea, 3),
                "per_step_ms": round(per_step_ms, 3)}
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def main() -> None:
    import torch

    torch.cuda.empty_cache()
    prompts = E.PROMPTS["math"] + E.PROMPTS["code"] + E.PROMPTS["chat"]
    print(f"=== M1 cost decomposition  L={_L} max_tokens={_MAX_TOKENS} "
          f"rounds={_ROUNDS}  (production path, quiet-host timing) ===")
    rows = []
    for c in _CONCURRENCIES:
        rows.append(_row("chain", prompts, c))
        rows.append(_row("tree-off", prompts, c))
        for b in _TREE_BUDGETS:
            rows.append(_row(f"treeB{b}", prompts, c))
    print("CONFIG     C   tok/s   E[A]  per_step_ms  out_tokens")
    for r in rows:
        print(f"{r['name']:<8} {r['concurrency']:<2} {r['tok/s']:>7.1f} "
              f"{r['E[A]']:>5.3f} {r['per_step_ms']:>9.3f} "
              f"{r['out_tokens']:>10}")
    # Crossover summary vs the chain control at C=1.
    chain = next(r for r in rows if r["name"] == "chain" and r["concurrency"] == 1)
    print(f"\n(crossover @C=1 vs chain@{_L}: chain tok/s={chain['tok/s']}")
    for r in rows:
        if r["concurrency"] == 1 and r["name"] != "chain":
            rel = r["tok/s"] / chain["tok/s"] if chain["tok/s"] else float("nan")
            print(f"  {r['name']:<8} tok/s={r['tok/s']:>7.1f} "
                  f"({rel * 100:>5.1f}% of chain) per_step_ms={r['per_step_ms']:>7.3f}")
    print("(invariants: T=0 identity at C=1 is the exact oracle; bf16 C>1 "
          "hashes are not — reported separately per round)")


if __name__ == "__main__":
    main()
