# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for ``SanaWmArStateAdapter`` (RFC #4480).

These drive the adapter directly with tiny CPU tensors -- no model, no GPU -- and
pin the per-session state contract the AR rollout relies on: correct GDN/softmax
block partitioning, fixed-size recurrent state per GDN block carried in place
across chunks, branch/layer isolation, session pinning under eviction, and a
clean reset. Equivalence against the upstream chunk-causal checkpoint is a later
(GPU) step; this is the allocator-free Phase-0 contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.experimental.world_models.adapters.state_sana_wm_ar_adapter import SanaWmArStateAdapter
from vllm_omni.experimental.world_models.memory import SessionMemoryManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Small stand-in for the SANA-WM backbone: 8 blocks, softmax every 4th.
BATCH, BLOCKS, HEADS, DIM, EVERY_N = 1, 8, 2, 4, 4
DTYPE, DEVICE = torch.float32, torch.device("cpu")
GDN_LAYERS = (0, 1, 2, 4, 5, 6)
SOFTMAX_LAYERS = (3, 7)


def _adapter(manager: SessionMemoryManager | None = None, session: str = "s0") -> SanaWmArStateAdapter:
    # NB: an empty SessionMemoryManager is falsy (it defines __len__), so use an
    # explicit None check rather than `manager or ...`, which would discard a
    # passed-in empty manager and break cross-instance sharing tests.
    return SanaWmArStateAdapter(
        session,
        SessionMemoryManager() if manager is None else manager,
        num_blocks=BLOCKS,
        num_heads=HEADS,
        head_dim=DIM,
        softmax_every_n=EVERY_N,
    )


def _created(manager: SessionMemoryManager | None = None, session: str = "s0") -> SanaWmArStateAdapter:
    adapter = _adapter(manager, session)
    adapter.create_state(BATCH, DTYPE, DEVICE)
    return adapter


def _gdn_payload() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(BATCH, HEADS, DIM, DIM), torch.randn(BATCH, HEADS, DIM, 1)


def test_block_partition_matches_transformer_rule() -> None:
    adapter = _adapter()
    assert adapter.gdn_layers == GDN_LAYERS
    assert adapter.softmax_layers == SOFTMAX_LAYERS
    # Union covers all blocks, disjoint.
    assert set(adapter.gdn_layers) | set(adapter.softmax_layers) == set(range(BLOCKS))
    assert not set(adapter.gdn_layers) & set(adapter.softmax_layers)


def test_real_backbone_partition_15_gdn_5_softmax() -> None:
    # The actual SANA-WM backbone: 20 blocks, softmax every 4th.
    adapter = SanaWmArStateAdapter("s", SessionMemoryManager(), num_blocks=20, num_heads=20, head_dim=112, softmax_every_n=4)
    assert adapter.softmax_layers == (3, 7, 11, 15, 19)
    assert len(adapter.gdn_layers) == 15


def test_from_config_derives_heads_and_dim() -> None:
    config = SimpleNamespace(hidden_size=2240, linear_head_dim=112, num_blocks=20, softmax_every_n=4)
    adapter = SanaWmArStateAdapter.from_config("s", SessionMemoryManager(), config)
    assert adapter.gdn_layers == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18)
    adapter.create_state(BATCH, DTYPE, DEVICE)
    view = adapter.get_gdn_state(0)
    assert view["state_kv"].shape == (BATCH, 20, 112, 112)
    assert view["state_z"].shape == (BATCH, 20, 112, 1)


def test_create_state_allocates_zeroed_gdn_states() -> None:
    adapter = _created()
    for layer in GDN_LAYERS:
        view = adapter.get_gdn_state(layer)
        assert view["state_kv"].shape == (BATCH, HEADS, DIM, DIM)
        assert view["state_z"].shape == (BATCH, HEADS, DIM, 1)
        torch.testing.assert_close(view["state_kv"], torch.zeros(BATCH, HEADS, DIM, DIM))
        torch.testing.assert_close(view["state_z"], torch.zeros(BATCH, HEADS, DIM, 1))


def test_get_gdn_state_on_softmax_layer_raises() -> None:
    adapter = _created()
    with pytest.raises(RuntimeError):
        adapter.get_gdn_state(SOFTMAX_LAYERS[0])


def test_gdn_state_commit_carries_in_place_across_chunks() -> None:
    adapter = _created()
    buf_kv = adapter.get_gdn_state(0)["state_kv"]  # buffer identity must be stable
    for _ in range(3):  # simulate AR chunks
        kv, z = _gdn_payload()
        adapter.commit_gdn_state(0, kv, z)
        view = adapter.get_gdn_state(0)
        assert view["state_kv"] is buf_kv  # in-place: no realloc
        torch.testing.assert_close(view["state_kv"], kv)
        torch.testing.assert_close(view["state_z"], z)


def test_gdn_commit_copies_values_not_reference() -> None:
    adapter = _created()
    kv, z = _gdn_payload()
    adapter.commit_gdn_state(0, kv, z)
    snap_kv = kv.clone()
    kv.add_(1.0)  # mutate source after commit
    torch.testing.assert_close(adapter.get_gdn_state(0)["state_kv"], snap_kv)


def test_gdn_layers_are_isolated() -> None:
    adapter = _created()
    payloads = {layer: _gdn_payload() for layer in GDN_LAYERS}
    for layer, (kv, z) in payloads.items():
        adapter.commit_gdn_state(layer, kv, z)
    for layer, (kv, z) in payloads.items():
        torch.testing.assert_close(adapter.get_gdn_state(layer)["state_kv"], kv)


def test_gdn_state_cfg_branches_isolated() -> None:
    # The cond/uncond GDN recurrences are independent; their state must not alias.
    adapter = _created()
    pos_kv, pos_z = _gdn_payload()
    neg_kv, neg_z = _gdn_payload()
    adapter.commit_gdn_state(0, pos_kv, pos_z, is_negative=False)
    adapter.commit_gdn_state(0, neg_kv, neg_z, is_negative=True)
    torch.testing.assert_close(adapter.get_gdn_state(0, is_negative=False)["state_kv"], pos_kv)
    torch.testing.assert_close(adapter.get_gdn_state(0, is_negative=True)["state_kv"], neg_kv)
    assert not torch.allclose(pos_kv, neg_kv)


def test_softmax_kv_branches_isolated() -> None:
    adapter = _created()
    for layer in SOFTMAX_LAYERS:
        pos = torch.randn(2, BATCH, 4, HEADS, DIM)
        neg = torch.randn(2, BATCH, 4, HEADS, DIM)
        adapter.commit_softmax_kv(layer, pos, is_negative=False)
        adapter.commit_softmax_kv(layer, neg, is_negative=True)
        torch.testing.assert_close(adapter.get_softmax_kv(layer, is_negative=False), pos)
        torch.testing.assert_close(adapter.get_softmax_kv(layer, is_negative=True), neg)
        assert not torch.allclose(pos, neg)


def test_text_cache_encode_once_mutation_persists() -> None:
    adapter = _created()
    for is_neg in (False, True):
        cache = adapter.get_text_cache(0, is_negative=is_neg)
        assert cache["is_init"] is False
        cache["is_init"] = True
        cache["k"] = torch.randn(2, 3)
        cache["v"] = torch.randn(2, 3)
        again = adapter.get_text_cache(0, is_negative=is_neg)
        assert again["is_init"] is True
        torch.testing.assert_close(again["k"], cache["k"])


def test_warmup_seed_frame_round_trip() -> None:
    adapter = _created()
    assert adapter.last_seed_frame() is None
    frame_a = torch.randn(3, 4, 4)
    frame_b = torch.randn(3, 4, 4)
    adapter.seed_frame(frame_a)
    adapter.seed_frame(frame_b)  # maxlen=1: only the latest is kept
    torch.testing.assert_close(adapter.last_seed_frame(), frame_b)


def test_chunk_index_defaults_zero_and_persists() -> None:
    adapter = _created()
    assert adapter.chunk_index == 0
    adapter.chunk_index = 2
    assert adapter.chunk_index == 2


def test_state_uninitialized_before_create_state_raises() -> None:
    adapter = _adapter()
    with pytest.raises(RuntimeError):
        adapter.get_gdn_state(0)


def test_reset_clears_all_state() -> None:
    adapter = _created()
    adapter.commit_gdn_state(0, *_gdn_payload())
    adapter.chunk_index = 3
    adapter.reset()
    assert adapter.chunk_index == 0
    with pytest.raises(RuntimeError):
        adapter.get_gdn_state(0)


def test_metadata_and_state_persist_across_adapter_instances() -> None:
    # A fresh adapter for the same session sees prior state (manager is the
    # single source of truth).
    manager = SessionMemoryManager()
    first = _created(manager, session="shared")
    kv, z = _gdn_payload()
    first.commit_gdn_state(0, kv, z)
    first.chunk_index = 4

    second = _adapter(manager, session="shared")
    assert second.chunk_index == 4
    torch.testing.assert_close(second.get_gdn_state(0)["state_kv"], kv)


def test_adapter_keeps_state_when_its_session_is_evicted() -> None:
    # An adapter mid-rollout must not lose state when the manager evicts its
    # session from the lookup table to bound memory.
    manager = SessionMemoryManager(max_sessions=2)
    adapter = _created(manager, session="active")
    kv, z = _gdn_payload()
    adapter.commit_gdn_state(0, kv, z)
    for i in range(3):
        _adapter(manager, session=f"other{i}")
    assert "active" not in manager
    torch.testing.assert_close(adapter.get_gdn_state(0)["state_kv"], kv)
