#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DF2T acceptance eval harness (Task 15, acceptance half).

Drives a vLLM spec-decode server over a held-out prompt corpus and reports the
v5 SS8 acceptance metrics from the /metrics spec-decode counters, stratified by
domain:

  mean acceptance per draft cycle   E[A] = (accepted + drafts) / drafts
  per-position survival             survival[d] = per_pos_accepted[d] / drafts
  invariant                         sum_d survival[d] == E[A] (asserted)
  P(reached last depth)             survival[L-1]  (censoring lower bound)

Usage:
  eval_df2t.py [--port 8006] [--max-tokens 64] [--domain {math,code,chat,all}]
               [--method {dflash,df2t}] [--tree-budget B] [--max-tree-depth L]

boots its own server per config (env TARGET/DRAFT), drives the corpus at T=0,
reads the /metrics delta, and prints a per-domain table plus the pooled check.

NOTE: this measures ACCEPTANCE only (call-count derived). Wall-clock / tok/s is
intentionally absent (untrusted on non-quiet hosts, e.g. a background game).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

import torch

TARGET = os.environ.get("TARGET", "/mnt/e/dev/Qwen3-0.6B")
DRAFT = os.environ.get("DRAFT", "/mnt/e/dev/Qwen3-0.6B-DF2T-CATMIX-B16")
HF_HOME = os.environ.get("HF_HOME", "/mnt/e/dev/df2t-run/.hf")
VENV_PY = os.environ.get("VENV_PY", "/home/orvid/vllm/.venv/bin/python")

# Fallback held-out prompts (used only if the datasets corpus cannot load).
PROMPTS: dict[str, list[str]] = {
    "math": [
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Solve for x: 3x + 7 = 22.",
        "What is the sum of the first ten prime numbers?",
        "A rectangle has length 8 and width 5. What is its area?",
        (
            "If 5 machines produce 40 items in 4 hours, how many items do "
            "3 machines produce in 4 hours?"
        ),
    ],
    "code": [
        "def is_palindrome(s):",
        "Write a function that returns the maximum of a list.",
        "def fibonacci(n):",
        "class Stack:",
        "def binary_search(arr, target):",
    ],
    "chat": [
        "The capital of Japan is",
        "The Great Wall of China was built",
        "Photosynthesis converts light energy into",
        "The mitochondria is the powerhouse of",
        "Water freezes at 0 degrees",
    ],
}


def load_corpus(n: int, seed: int) -> dict[str, list[str]]:
    """Load a per-domain corpus (openai/gsm8k, openai_humaneval, ShareGPT) from
    the writable HF cache, sampling n prompts per domain with a fixed seed.
    Falls back to PROMPTS if the datasets library or cache is unavailable.
    """
    import random

    try:
        import datasets
    except ImportError:
        return PROMPTS
    cache_dir = "/mnt/e/dev/df2t-run/.hf"
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    try:
        math = datasets.load_dataset(
            "openai/gsm8k", "main", split="test", cache_dir=cache_dir
        )
        out["math"] = [
            row["question"] for row in rng.sample(list(math), min(n, len(math)))
        ]
    except Exception as e:  # noqa: BLE001  (fall back to the hand set)
        print(f"  [corpus] math fallback: {e}")
        out["math"] = PROMPTS["math"]
    try:
        code = datasets.load_dataset(
            "openai/openai_humaneval", split="test", cache_dir=cache_dir
        )
        out["code"] = [
            row["prompt"] for row in rng.sample(list(code), min(n, len(code)))
        ]
    except Exception as e:  # noqa: BLE001
        print(f"  [corpus] code fallback: {e}")
        out["code"] = PROMPTS["code"]
    try:
        chat = datasets.load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered", split="train", cache_dir=cache_dir
        )

        def _first_user(row: dict) -> str:
            conv = row.get("conversations") or []
            for turn in conv:
                if turn.get("from") in ("human", "user") and turn.get("value"):
                    return str(turn["value"])[:200]
            return ""

        out["chat"] = [
            _first_user(row) for row in rng.sample(list(chat), min(n, len(chat)))
        ]
        out["chat"] = [p for p in out["chat"] if p]
    except Exception as e:  # noqa: BLE001
        print(f"  [corpus] chat fallback: {e}")
        out["chat"] = PROMPTS["chat"]
    return out


MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))


def _counters(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
        text = r.read().decode()
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total{"):
            out["drafts"] = float(line.split("} ")[-1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            out["draft_tokens"] = float(line.split("} ")[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            out["accepted"] = float(line.split("} ")[-1])
        elif (
            line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")
            and "position=" in line
        ):
            pos = line.split('position="')[1].split('"')[0]
            out[f"pos_{pos}"] = float(line.split("} ")[-1])
    return out


def _boot(
    port: int,
    method: str,
    tree_budget: int | None,
    max_tree_depth: int | None,
    spec_tokens: int = 15,
    tree_s: float | None = None,
    eager: bool = True,
    spec_extra: dict | None = None,
) -> subprocess.Popen:
    # ``method="df2t"`` here means the dflash drafter with df2t tree selection.
    spec = {
        "method": "dflash" if method == "df2t" else method,
        "model": DRAFT,
        "num_speculative_tokens": spec_tokens,
    }
    if method == "df2t":
        spec["tree_selection"] = "df2t"
        if tree_budget is not None:
            spec["tree_budget"] = tree_budget
        if max_tree_depth is not None:
            spec["max_tree_depth"] = max_tree_depth
        if tree_s is not None:
            spec["tree_s"] = tree_s
    if spec_extra:
        spec.update(spec_extra)
    cmd = [
        VENV_PY,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        TARGET,
        "--speculative-config",
        json.dumps(spec),
        "--gpu-memory-utilization",
        "0.3",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        "512",
    ]
    if eager:
        # Acceptance is call-count based and timing-free, so eager is fine and
        # fastest. The TIMING harness passes eager=False to measure the real
        # production path (FULL CUDA graphs).
        cmd.append("--enforce-eager")
    env = dict(os.environ, HF_HOME=HF_HOME, CUDA_VISIBLE_DEVICES="0")
    log_path = f"/tmp/df2t_eval_{method}_{tree_budget or 0}_{max_tree_depth or 0}.log"
    # The handle lives for the server's lifetime (appended by the child), so it
    # is deliberately not scope-closed — hence the noqa.
    log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return proc
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"server failed to start ({method=} {tree_budget=})")


def _drive(port: int, prompts: list[str], concurrency: int = 1) -> None:
    import concurrent.futures

    def _one(p: str) -> None:
        body = json.dumps(
            {"model": TARGET, "prompt": p, "max_tokens": MAX_TOKENS, "temperature": 0.0}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(_one, prompts))


def _acceptance_metrics(before: dict, after: dict) -> dict:
    d0 = lambda k: after.get(k, 0) - before.get(k, 0)
    drafts = d0("drafts")
    accepted = d0("accepted")
    ea = (accepted + drafts) / drafts if drafts > 0 else float("nan")
    pos = sorted(int(k[4:]) for k in after if k.startswith("pos_"))
    surv = {
        d: (after.get(f"pos_{d}", 0) - before.get(f"pos_{d}", 0)) / drafts for d in pos
    }
    sum_surv = sum(surv.values())
    return {
        "drafts": drafts,
        "draft_tokens": d0("draft_tokens"),
        "accepted": accepted,
        "E[A]": ea,
        "survival": surv,
        "sum_survival": sum_surv,
    }


def _report(port: int, name: str, prompts: list[str], concurrency: int = 1) -> dict:
    before = _counters(port)
    _drive(port, prompts, concurrency)
    after = _counters(port)
    m = _acceptance_metrics(before, after)
    # Invariant: sum_d survival_d == E[A] (= accepted/drafts).
    if m["drafts"] > 0:
        assert abs(m["sum_survival"] - m["accepted"] / m["drafts"]) < 1e-6, (
            f"{name}: sum_survival {m['sum_survival']} != "
            f"accepted/drafts {m['accepted'] / m['drafts']}"
        )
    return {"name": name, **m}


def main() -> None:
    torch.accelerator.empty_cache()
    port = int(os.environ.get("PORT", "8006"))
    method = (
        sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "df2t"
    )
    budget = (
        sys.argv[sys.argv.index("--tree-budget") + 1]
        if "--tree-budget" in sys.argv
        else None
    )
    depth = (
        sys.argv[sys.argv.index("--max-tree-depth") + 1]
        if "--max-tree-depth" in sys.argv
        else None
    )
    budget = int(budget) if budget else None
    depth = int(depth) if depth else None
    seed = int(os.environ.get("CORPUS_SEED", "0"))
    n = int(os.environ.get("CORPUS_N", "40"))
    concurrency = int(os.environ.get("CONCURRENCY", "1"))
    corpus = load_corpus(n, seed)
    print(
        f"corpus: { {d: len(p) for d, p in corpus.items()} } "
        f"(seed={seed}, n_per_domain={n})"
    )

    print(
        f"=== {method}  tree_budget={budget}  max_tree_depth={depth}  "
        f"max_tokens={MAX_TOKENS}  (T=0, concurrency={concurrency}) ==="
    )
    proc = _boot(port, method, budget, depth)
    try:
        rows = []
        for domain, plist in corpus.items():
            rows.append(
                _report(port, f"{method}/B={budget}/{domain}", plist, concurrency)
            )
        for row in rows:
            sv = row["survival"]
            tail = ""
            if sv:
                last = max(sv)
                tail = (
                    f"surv0={sv.get(0, 0):.3f}  last_depth={last}@{sv.get(last, 0):.3f}"
                )
            else:
                tail = "(no per-position counters)"
            print(
                f"  {row['name']:<28} drafts={row['drafts']:>6.0f} "
                f"E[A]={row['E[A]']:.3f}  sum_surv={row['sum_survival']:.3f}  {tail}"
            )
        # pooled invariant check across domains
        print("  (invariant: sum_d survival_d == E[A] per domain — asserted above)")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
