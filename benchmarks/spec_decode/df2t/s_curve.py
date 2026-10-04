#!/usr/bin/env python3
"""DF2T s-curve (server-based): tok/s + acceptance across tree_s, and T=0
output identity across s (v5 SS3.4 invariant 3: bit/stream-identical at T=0).

Boots a vLLM spec-decode server per config on the PRODUCTION path (FULL CUDA
graphs, NOT ``--enforce-eager`` — the df2t verifier micro-graph captures and
replays under the same graph region), threads ``tree_s`` into each server's
``--speculative-config`` so the knob genuinely varies the deployed node budget
(B = round(s * tree_budget) in ``resolve_nodes_max``), warms up, then measures
steady-state wall-clock output tok/s over a fixed corpus at a given concurrency
(threaded requests = co-batched streams), plus the /metrics acceptance counters,
and captures completion text hashes to confirm T=0 identity across s.

Note: wall-clock is only valid on a QUIET host; if a background workload is
running, the numbers are untrustworthy. Also T=0 stream identity is only an
exact oracle at single-request shapes — at concurrency > 1, bf16 batch-shape
reduction noise makes exact-token identity across *rounds* of the same config
nondeterministic for the plain chain too (documented; use fp32/flex as the
exact-parity vehicle).
"""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request

import eval_df2t as E
import torch

_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
_L = int(os.environ.get("MAX_TREE_DEPTH", "15"))
# tree_s values to sweep (plan Step 4: {0.25, 0.5, 1, 2, 4}). Note: tree_s that
# resolve to nodes_max < max_tree_depth (s <= 0.5 -> B in {4, 8}) cannot boot
# the production server on this stack — a pre-existing upstream VLLM_COMPILE
# shape assert in the DFlash2 candidate_selector at the df2t 1+B decode widths
# (reproduced at base HEAD) — so the benchmark driver overrides S_VALUES with
# the servable subset {1.0, 2.0, 4.0} (B in {16, 32, 64}).
_S_VALUES = tuple(
    float(v) for v in os.environ.get("S_VALUES", "0.25,0.5,1.0,2.0,4.0").split(",") if v
)
_PROMPTS = E.PROMPTS["chat"] + E.PROMPTS["math"] + E.PROMPTS["code"]


def complete(port: int, prompt: str, max_tokens: int = _MAX_TOKENS) -> tuple[str, int]:
    body = json.dumps({"model": E.TARGET, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], int(d["usage"].get("completion_tokens", 0))


def bench(port: int, c: int, rounds: int = 4) -> tuple[float, float, str]:
    """Return (tok/s, out_tokens, stream_hash). Concurrency c = threaded stream
    count co-batched by the engine; rounds repeats for stable timing. The batch
    replicates the prompt list to at least ``c`` streams, so C=1 still churns
    the full held-out list sequentially (a large stable token sample) while
    C>1 co-batches >= c concurrent streams."""
    batch = (_PROMPTS * ((c // len(_PROMPTS)) + 1))[: max(c, len(_PROMPTS))]
    tokens_total = 0
    elapsed = 0.0
    hashes: list[str] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            results = list(ex.map(lambda p: complete(port, p), batch))
        elapsed += time.perf_counter() - t0
        tokens_total += sum(nt for _, nt in results)
        hashes.append(hashlib.sha256(
            "".join(t for t, _ in results).encode()).hexdigest()[:12])
    toks = tokens_total / elapsed if elapsed > 0 else float("nan")
    return toks, tokens_total, ",".join(sorted(set(hashes)))


def main() -> None:
    port = int(os.environ.get("PORT", "8006"))
    c = int(os.environ.get("CONCURRENCY", "1"))
    print(f"=== s-curve (server) L={_L} max_tokens={_MAX_TOKENS} "
          f"concurrency={c} (quiet-host timing) ===")

    # chain baseline (production path: FULL CUDA graphs, not eager)
    proc = E._boot(port, "dflash", None, None, eager=False)
    try:
        time.sleep(2)
        complete(port, _PROMPTS[0])  # warm
        t, n, h = bench(port, c)
        print(f"  chain: tok/s={t:.1f} out_tokens={n} hash={h}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    ref = None
    for s in _S_VALUES:
        # tree_s is threaded into the server's speculative-config JSON so each
        # point genuinely varies the deployed node budget (B = round(s * policy)
        # in resolve_nodes_max), and eager=False measures the production path
        # (FULL CUDA graphs + the df2t verifier micro-graph) instead of eager.
        proc = E._boot(port, "df2t", 16, _L, spec_tokens=15, tree_s=s, eager=False)
        try:
            time.sleep(2)
            complete(port, _PROMPTS[0])  # warm
            t, n, h = bench(port, c)
            lines = json.dumps({"tree_s": s, "tok/s": round(t, 1),
                                "out_tokens": n, "hash": h})
            print(f"  s={s:<4} {lines}")
            # acceptance via metrics delta over a short drive
            before = E._counters(port)
            E._drive(port, E.PROMPTS["math"], c)
            m = E._acceptance_metrics(before, E._counters(port))
            print(f"         math accept: E[A]={m['E[A]']:.3f} "
                  f"sum_surv={m['sum_survival']:.3f}")
            if ref is None:
                ref = h
            print(f"         T=0 identity vs s=0.25: {'OK' if h == ref else 'DIFF'}")
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    print("  (invariant: df2t T=0 output stream-identical across all s)")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
