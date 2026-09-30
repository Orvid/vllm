# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the df2t FlexAttention tree mask.

Task 8 builds the tree-verify attention mask on the FlexAttention vehicle: a
``logical_mask_mod`` closure over per-request ancestor words, so the verify
forward attends only to context + anchor + ancestor-or-self nodes. Within a
request's verify block the logical KV index equals the block-slot: keys with
logical index < C are context (always allowed for the fully-causal target), the
key at logical index C is the anchor, and keys at ``C + 1 + j`` are nodes that
are allowed iff ``(words[r, q] >> j) & 1`` (ancestor-or-self of query node q).

These tests pin (1) the boolean mask matrix against a manual expectation from a
concrete small tree and (2) that the mask actually gates a packed FlexAttention
forward against a reference attention computed with the same mask.

These tests must stay FULLY OFFLINE: they construct a real `VllmConfig` via
`create_vllm_config`, which resolves the model config through vLLM's loader.
The `qwen3_config_path` fixture (see conftest.py) provides a locally-written
Qwen3 config directory so no HuggingFace access ever happens.
"""

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    MockAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionImpl,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheLayout
from vllm.v1.worker.gpu.spec_decode.df2t.attention import (
    attach_tree_flex_mask,
    build_flex_tree_mask,
)


def _manual_matrix(C: int, words: list[int], B: int) -> torch.Tensor:
    """Manual [C + 1 + B, C + 1 + B] boolean matrix over logical (q, k).

    The real verify block only holds query rows ``[C, C + B]`` (anchor at C,
    node with arena ``a`` at ``C + 1 + a``); rows below C are context positions
    that never query in the verify layout and are left False. Key ``k`` is
    allowed iff ``k < C`` (context), ``k == C`` (anchor), or ``k == C + 1 + j``
    with bit ``j`` set in the query node's ancestor-or-self word.
    """
    n = C + 1 + B
    matrix = torch.zeros(n, n, dtype=torch.bool)
    words_t = torch.tensor(words, dtype=torch.long)
    for arena in range(B):
        q = C + 1 + arena
        matrix[q, : C + 1] = True
        for j in range(B):
            k = C + 1 + j
            matrix[q, k] = bool((words_t[arena] >> j) & 1)
    matrix[C, : C + 1] = True
    return matrix


def test_flex_tree_mask_matches_manual():
    """The closure reproduces the hand-computed matrix on the verify-block
    rows: context always allowed, the anchor allowed, and node keys allowed
    exactly when their bit is set in the query node's ancestor-or-self word."""
    C, B = 4, 3
    words = [0b1, 0b11, 0b101]  # node1: self; node2: self + node0 ancestors
    request_lookup = torch.zeros(1, dtype=torch.int32)
    mask_mod, hint = build_flex_tree_mask(
        1,
        torch.tensor([C], dtype=torch.long),
        torch.tensor([words], dtype=torch.long),
        B,
        request_lookup,
    )
    assert hint is not None

    n = C + 1 + B
    b = torch.zeros(n, 1, dtype=torch.long)
    h = torch.zeros(n, 1, dtype=torch.long)
    q = torch.arange(n, dtype=torch.long)[:, None].expand(n, n)
    k = torch.arange(n, dtype=torch.long)[None, :].expand(n, n)
    got = mask_mod(b.expand(n, n), h, q, k)

    expected = _manual_matrix(C, words, B)
    # Rows below C are context positions, never queries in the verify layout.
    assert torch.equal(got[C:], expected[C:])
    assert torch.equal(got[C], expected[C])  # anchor row: context + anchor
    assert got[4].sum().item() == C + 1


def test_flex_tree_mask_matches_manual_multi_request():
    """Per-request data is found through the packed query index -> request
    lookup, so one closure serves different trees in a single packed batch."""
    specs = [(4, [0b1, 0b11, 0b101], 3), (2, [0b1, 0b11], 2), (3, [0b1], 1)]
    num_reqs = len(specs)
    max_b = 3
    query_lens = [b + 1 for _, _, b in specs]
    query_start_loc = [0]
    for qlen in query_lens:
        query_start_loc.append(query_start_loc[-1] + qlen)
    num_tokens = query_start_loc[-1]
    request_lookup = torch.zeros(num_tokens, dtype=torch.int32)
    for r, (_, _, b) in enumerate(specs):
        request_lookup[query_start_loc[r] : query_start_loc[r + 1]] = r

    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.long)
    words = torch.zeros(num_reqs, max_b, dtype=torch.long)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.long)

    mask_mod, _ = build_flex_tree_mask(
        num_reqs, c_per_req, words, max_b, request_lookup
    )

    h = torch.zeros(1, dtype=torch.long)
    for r, (c, w, b_nodes) in enumerate(specs):
        n_r = c + 1 + b_nodes
        # logical query position -> packed token index of that verify row.
        b_for_q = torch.full((n_r,), query_start_loc[r], dtype=torch.long)
        b_for_q[c + 1 : c + 1 + b_nodes] = (
            query_start_loc[r] + 1 + torch.arange(b_nodes, dtype=torch.long)
        )
        q = torch.arange(n_r, dtype=torch.long)[:, None].expand(n_r, n_r)
        k = torch.arange(n_r, dtype=torch.long)[None, :].expand(n_r, n_r)
        b = b_for_q[:, None].expand(n_r, n_r)
        got = mask_mod(b, h, q, k)
        expected = _manual_matrix(c, w, b_nodes)
        assert torch.equal(got[c:], expected[c:]), f"request {r}"


def test_flex_tree_mask_contextless_request_is_causal():
    """A request with C == 0 (fresh prefill co-scheduled with a tree decode)
    has no tree and no context: the mask falls back to per-request causal
    attention over its own rows, so it can verify its first token."""
    max_b = 3
    # Request 0: a tree request at C == 4. Request 1: C == 0, no tree.
    n0, n1 = 5, 4
    query_start_loc = [0, n0, n0 + n1]
    num_tokens = query_start_loc[-1]
    request_lookup = torch.zeros(num_tokens, dtype=torch.int32)
    request_lookup[query_start_loc[0] : query_start_loc[1]] = 0
    request_lookup[query_start_loc[1] : query_start_loc[2]] = 1

    c_per_req = torch.tensor([4, 0], dtype=torch.long)
    words = torch.zeros(2, max_b, dtype=torch.long)
    words[0] = torch.tensor([0b1, 0b11, 0b101], dtype=torch.long)

    mask_mod, _ = build_flex_tree_mask(2, c_per_req, words, max_b, request_lookup)
    h = torch.zeros(1, dtype=torch.long)
    q = torch.arange(n1, dtype=torch.long)[:, None].expand(n1, n1)
    k = torch.arange(n1, dtype=torch.long)[None, :].expand(n1, n1)
    b = torch.full((n1, n1), query_start_loc[1], dtype=torch.long)
    got = mask_mod(b, h, q, k)
    # Causal: row q attends exactly keys 0..q.
    assert torch.equal(got, k <= q), "contextless request must attend causally"


def test_flex_tree_mask_treeless_request_with_context_is_causal():
    """A request with context (C > 0) but no tree this step (a chunked-prefill
    continuation or prefix-cache hit co-scheduled with a tree decode) must
    attend causally over context and its own rows, not just context + row 0,
    and the block-sparsity hint must never prune those keys."""
    max_b = 3
    C, n1 = 6, 4
    n0 = 5
    query_start_loc = [0, n0, n0 + n1]
    request_lookup = torch.zeros(query_start_loc[-1], dtype=torch.int32)
    request_lookup[query_start_loc[1] :] = 1

    c_per_req = torch.tensor([4, C], dtype=torch.long)
    words = torch.zeros(2, max_b, dtype=torch.long)
    words[0] = torch.tensor([0b1, 0b11, 0b101], dtype=torch.long)

    mask_mod, hint = build_flex_tree_mask(2, c_per_req, words, max_b, request_lookup)
    n = C + n1
    q = torch.arange(C, n, dtype=torch.long)[:, None].expand(n1, n)
    k = torch.arange(n, dtype=torch.long)[None, :].expand(n1, n)
    b = (query_start_loc[1] + torch.arange(n1, dtype=torch.long))[:, None]
    got = mask_mod(b.expand(n1, n), torch.zeros(1, dtype=torch.long), q, k)
    assert torch.equal(got, k <= q), "treeless request must attend causally"

    block_size = 2
    kv_block = torch.arange(n // block_size, dtype=torch.long)[None, :]
    visible = hint.hint_fn(
        b[:, 0], torch.arange(C, n, dtype=torch.long), kv_block, block_size
    )
    needed = (kv_block * block_size) <= torch.arange(C, n)[:, None]
    assert (visible | ~needed).all(), "hint pruned a causally visible block"


def test_flex_tree_mask_hint_never_prunes_visible_blocks():
    """The BlockSparsityHint is a superset of the real mask at block
    granularity (a block holding any allowed key stays visible) and prunes
    node-region blocks an anchor query cannot reach."""
    C, B = 4, 3
    words = [0b1, 0b11, 0b101]
    request_lookup = torch.zeros(B + 1, dtype=torch.int32)  # one req's block
    _, hint = build_flex_tree_mask(
        1,
        torch.tensor([C], dtype=torch.long),
        torch.tensor([words], dtype=torch.long),
        B,
        request_lookup,
    )

    block_size = 2
    num_blocks = 4  # logical keys [0, 8) at block granularity
    # One packed query per verify row: anchor + the B nodes.
    q_idx = torch.arange(B + 1, dtype=torch.long)
    logical_q = torch.tensor([C, C + 1, C + 2, C + 3], dtype=torch.long)
    kv_block = torch.arange(num_blocks, dtype=torch.long)[None, :]
    pruned = ~hint.hint_fn(q_idx, logical_q, kv_block, block_size)

    # Anchor query (row 0) reaches only context + anchor blocks 0..2.
    assert pruned[0, 3].item()
    assert (~pruned[0, :3]).all().item()
    # Node 2 (row 3) attends node 0 and 2 (keys 5, 7): block 3 stays visible.
    assert (~pruned[3, :4]).all().item()

    # Consistency: no block that holds an allowed key is ever pruned.
    matrix = _manual_matrix(C, words, B)
    for row in range(B + 1):
        q_logical = logical_q[row].item()
        for bloc in range(num_blocks):
            keys = range(bloc * block_size, (bloc + 1) * block_size)
            any_visible = any(matrix[q_logical, k].item() for k in keys)
            if any_visible:
                assert not pruned[row, bloc].item(), (
                    f"pruned a block with a visible key (q={q_logical}, block={bloc})"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_tree_mask_gates_attention(qwen3_config_path):
    """The tree mask actually gates a packed FlexAttention forward: attaching it
    to the Flex metadata and running the impl matches a reference attention
    computed with the same mask, so sibling nodes do not leak and ancestors
    stay visible."""
    device = torch.device("cuda")
    vllm_config = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=1024,
        max_num_seqs=8,
        max_num_batched_tokens=256,
    )
    head_size = vllm_config.model_config.get_head_size()
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    scale = 1.0 / (head_size**0.5)

    specs = [(4, [0b1, 0b11, 0b101], 3), (2, [0b1, 0b11], 2)]
    num_reqs = len(specs)
    query_lens = [b + 1 for _, _, b in specs]
    seq_lens = [c + 1 + b for c, _, b in specs]
    max_b = max(b for _, _, b in specs)

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens, name="df2t_tree")
    common = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    torch.manual_seed(0)
    k_full, v_full, k_contexts, v_contexts = [], [], [], []
    for c, _, b_nodes in specs:
        seq_len = c + 1 + b_nodes
        k_full.append(torch.randn(seq_len, num_kv_heads, head_size, device=device))
        v_full.append(torch.randn(seq_len, num_kv_heads, head_size, device=device))
        k_contexts.append(k_full[-1][:c])
        v_contexts.append(v_full[-1][:c])

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float32,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1024,
        common_attn_metadata=common,
        layout=KVCacheLayout.LBNHC,
    )

    # Reference output: per request, SDPA over the tree mask matrix for the
    # verify-block query rows (logical slots C..C+B).
    q_rows, k_rows, v_rows, sdpa_rows = [], [], [], []
    for r, (c, w, b_nodes) in enumerate(specs):
        q_r = torch.randn(b_nodes + 1, num_heads, head_size, device=device)
        kfull = k_full[r].transpose(0, 1)[None]
        vfull = v_full[r].transpose(0, 1)[None]
        if num_heads != num_kv_heads:
            kfull = kfull.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vfull = vfull.repeat_interleave(num_heads // num_kv_heads, dim=1)
        mask = _manual_matrix(c, w, b_nodes)[c:].unsqueeze(0).unsqueeze(0).to(device)
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                q_r.unsqueeze(0).transpose(1, 2),
                kfull,
                vfull,
                attn_mask=mask,
                scale=scale,
            )
            .transpose(1, 2)
            .squeeze(0)
        )

        q_rows.append(q_r)
        k_rows.append(k_full[r][c:])
        v_rows.append(v_full[r][c:])
        sdpa_rows.append(out)

    query = torch.cat(q_rows, dim=0)
    key = torch.cat(k_rows, dim=0)
    value = torch.cat(v_rows, dim=0)
    sdpa_output = torch.cat(sdpa_rows, dim=0)

    # df2t per-request SoA inputs for the tree mask.
    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.long, device=device)
    words = torch.zeros(num_reqs, max_b, dtype=torch.long, device=device)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.long, device=device)

    kv_spec = create_standard_kv_cache_spec(vllm_config)
    with set_current_vllm_config(vllm_config):
        builder = FlexAttentionMetadataBuilder(
            kv_spec, ["placeholder"], vllm_config, device
        )
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        attach_tree_flex_mask(attn_metadata, num_reqs, c_per_req, words, max_b)

        impl = FlexAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            attn_type=AttentionType.DECODER,
            kv_cache_dtype="auto",
        )

    layer = MockAttentionLayer(device)
    output = torch.empty_like(query)
    impl.do_kv_cache_update(layer, key, value, kv_cache, attn_metadata.slot_mapping)
    backend_output = impl.forward(
        layer, query, key, value, kv_cache, attn_metadata, output=output
    )

    assert backend_output.shape == sdpa_output.shape
    torch.testing.assert_close(
        backend_output.float(), sdpa_output.float(), rtol=2e-2, atol=2e-2
    )


def _fake_flex_metadata(doc_ids: torch.Tensor) -> FlexAttentionMetadata:
    """Uninitialised FlexAttentionMetadata (real class, so the runner's
    isinstance gate passes) with only the fields the df2t attach helper and the
    assertions touch."""
    meta = object.__new__(FlexAttentionMetadata)
    meta.doc_ids = doc_ids
    meta.logical_mask_mod = None
    meta.block_sparsity_hint = None
    meta.mask_mod = None
    meta.direct_build = False
    meta.block_mask = None
    meta.build_calls = 0

    def get_mask_mod():
        return meta.logical_mask_mod

    def build_block_mask():
        meta.build_calls += 1
        meta.block_mask = object()
        return meta.block_mask

    meta.get_mask_mod = get_mask_mod
    meta.build_block_mask = build_block_mask
    return meta


class _FakeBackend:
    def __init__(self, name: str):
        self._name = name

    def get_name(self):
        return self._name


def _fake_tree(words, device) -> "object":
    return type(
        "Tree", (), {"words": torch.tensor(words, dtype=torch.long, device=device)}
    )()


def _fake_runner(spec, req_states, device, attn_groups):
    import types

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = types.SimpleNamespace(
        speculative_config=spec,
        req_states=req_states,
        device=device,
        _pending_tree_flex_mask=None,
        attn_groups=attn_groups,
    )
    for name in (
        "_stage_tree_flex_mask",
        "_tree_verify_layer_names",
        "_attach_tree_flex_mask",
    ):
        setattr(runner, name, types.MethodType(getattr(GPUModelRunner, name), runner))
    return runner


def test_runner_stage_attach_targets_only_the_flex_verify_group():
    """The runner's stage -> attach path wires the tree mask onto the target
    Flex verify group only, keyed by the right per-request C and words, and
    leaves a second Flex metadata (another group) untouched."""
    device = torch.device("cpu")
    spec = type(
        "Spec",
        (),
        {"tree_attention_backend": "flex", "max_tree_budget": 64},
    )()
    # Two requests: req0 C=4 with a 3-node tree, req1 C=2 with a 2-node tree.
    num_computed = torch.tensor([4, 2, 0, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11, 0b101], device), _fake_tree([0b1, 0b11], device)]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    attn_groups = [
        [
            type(
                "G",
                (),
                {
                    "backend": _FakeBackend("FLEX_ATTENTION"),
                    "layer_names": ["target.attn"],
                },
            )(),
            type(
                "G",
                (),
                {"backend": _FakeBackend("FLASH_ATTN"), "layer_names": ["other.attn"]},
            )(),
        ]
    ]
    runner = _fake_runner(spec, req_states, device, attn_groups)

    runner._stage_tree_flex_mask(True, trees, 2, idx_mapping)
    assert runner._pending_tree_flex_mask is not None

    target_meta = _fake_flex_metadata(
        torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    )
    other_meta = _fake_flex_metadata(
        torch.tensor([2, 2, 2], dtype=torch.int32)  # a different group's tokens
    )
    runner._attach_tree_flex_mask(
        {"target.attn": target_meta, "other.attn": other_meta}
    )

    assert runner._pending_tree_flex_mask is None
    # Only the target Flex group received the tree mask.
    assert target_meta.logical_mask_mod is not None
    assert target_meta.block_sparsity_hint is not None
    assert target_meta.build_calls == 1
    assert other_meta.logical_mask_mod is None
    assert other_meta.build_calls == 0

    # The mask lands on the right requests: per-request rows match the manual
    # matrix through the target metadata's doc_ids lookup.
    h = torch.zeros(1, dtype=torch.long)
    query_lens = [4, 3]  # 1 + B per request (anchor + nodes)
    start = 0
    for r, (c, w, b_nodes) in enumerate(
        [(4, [0b1, 0b11, 0b101], 3), (2, [0b1, 0b11], 2)]
    ):
        n_r = c + 1 + b_nodes
        b_for_q = torch.full((n_r,), start, dtype=torch.long)
        b_for_q[c] = start
        b_for_q[c + 1 : c + 1 + b_nodes] = (
            start + 1 + torch.arange(b_nodes, dtype=torch.long)
        )
        q = torch.arange(n_r, dtype=torch.long)[:, None].expand(n_r, n_r)
        k = torch.arange(n_r, dtype=torch.long)[None, :].expand(n_r, n_r)
        b_grid = b_for_q[:, None].expand(n_r, n_r)
        got = target_meta.logical_mask_mod(b_grid, h, q, k)
        assert torch.equal(got[c:], _manual_matrix(c, w, b_nodes)[c:])
        start += query_lens[r]


def test_runner_stage_scopes_zero_context():
    """A zero-context (fresh-prompt prefill, C == 0) request staged in a
    flex-tree batch is scoped per-request (causal over its own rows) instead
    of rejected, so it can verify its first token alongside a tree decode."""
    device = torch.device("cpu")
    spec = type("Spec", (), {"tree_attention_backend": "flex", "max_tree_budget": 64})()
    num_computed = torch.tensor([4, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11], device)]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    runner = _fake_runner(
        spec,
        req_states,
        device,
        [
            [
                type(
                    "G",
                    (),
                    {
                        "backend": _FakeBackend("FLEX_ATTENTION"),
                        "layer_names": ["target.attn"],
                    },
                )()
            ]
        ],
    )

    runner._stage_tree_flex_mask(True, trees, 2, idx_mapping)
    pending = runner._pending_tree_flex_mask
    assert pending is not None
    _, c_per_req, words, _ = pending
    # The contextless request is staged with a zero anchor length, not dropped.
    assert c_per_req.tolist() == [4, 0]
    assert bool((words[1] == 0).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_flex_selector_forces_flex_backend(qwen3_config_path):
    """tree_attention_backend == 'flex' routes the plain decoder full-attention
    layers to the FlexAttention vehicle on df2t, while windowed / MLA / encoder
    kinds and the default 'fa4' stay on their usual backends."""
    from types import SimpleNamespace

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.selector import get_attn_backend

    def spec_obj(tree_attention_backend):
        return SimpleNamespace(
            use_df2t=lambda: True,
            tree_attention_backend=tree_attention_backend,
            enable_adaptive_verification=False,
            num_speculative_tokens=0,
        )

    def resolve(tree_attention_backend, **kwargs):
        cfg = create_vllm_config(
            model_name=qwen3_config_path, max_model_len=64, block_size=16
        )
        cfg.speculative_config = spec_obj(tree_attention_backend)
        head_size = cfg.model_config.get_head_size()
        with set_current_vllm_config(cfg):
            return get_attn_backend(
                head_size=head_size,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                num_heads=cfg.model_config.get_num_attention_heads(cfg.parallel_config),
                **kwargs,
            ).get_name()

    assert resolve("flex") == "FLEX_ATTENTION"
    # Windowed / MLA / encoder kinds and the default 'fa4' are not forced.
    assert resolve("flex", has_sliding_window=True) != "FLEX_ATTENTION"
    assert resolve("flex", use_mla=True) != "FLEX_ATTENTION"
    assert resolve("flex", attn_type=AttentionType.ENCODER_ONLY) != "FLEX_ATTENTION"
    assert resolve("fa4") != "FLEX_ATTENTION"
