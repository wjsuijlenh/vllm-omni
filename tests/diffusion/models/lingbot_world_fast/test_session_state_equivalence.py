# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence of ``LingbotWorldFastStateAdapter`` with the bespoke state.

Drives the bespoke ``LingbotWorldFastState`` and the manager-backed adapter
through the same write sequences the pipeline performs and asserts that every
field the pipeline reads comes back identical. Also covers what only the
manager-backed path provides: state keyed by session id, release through
``drop_session``, retention capping, and per-device byte accounting.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world_fast.state_lingbot_world_fast import (
    LingbotWorldFastState,
)
from vllm_omni.experimental.world_models.adapters.state_lingbot_world_fast_adapter import (
    LingbotWorldFastStateAdapter,
)
from vllm_omni.experimental.world_models.session_state import SessionStateManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 3
DEVICE = torch.device("cpu")

FIELDS = (
    "current_start_frame",
    "local_end_index",
    "global_end_index",
    "is_initialized",
    "current_lat_f",
    "session_id",
    "num_layers",
    "h",
    "w",
    "lat_h",
    "lat_w",
    "frame_seqlen",
    "last_decoded_latent",
)


def _bespoke() -> LingbotWorldFastState:
    return LingbotWorldFastState()


def _adapter(session_id: str = "session-a") -> LingbotWorldFastStateAdapter:
    return LingbotWorldFastStateAdapter(session_id, SessionStateManager(max_sessions=1))


def _assert_fields_equal(a: object, b: object) -> None:
    for field in FIELDS:
        left, right = getattr(a, field), getattr(b, field)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        elif isinstance(left, list):
            assert isinstance(right, list) and len(left) == len(right)
            for lt, rt in zip(left, right):
                torch.testing.assert_close(lt, rt)
        else:
            assert left == right, f"{field}: {left!r} != {right!r}"


def _run_first_call_writes(state: LingbotWorldFastState | LingbotWorldFastStateAdapter) -> None:
    """The writes the pipeline performs on the first call of a session."""
    state.reset()
    state.init_state(DEVICE, NUM_LAYERS)
    state.session_id = "session-a"
    state.h, state.w = 480, 832
    state.lat_h, state.lat_w = 60, 104
    state.frame_seqlen = 1560
    state.last_decoded_latent = torch.randn(16, 2, 60, 104, generator=torch.Generator().manual_seed(0))
    state.advance(4)


def test_defaults_match_bespoke() -> None:
    _assert_fields_equal(_bespoke(), _adapter())


def test_first_call_writes_read_back_identically() -> None:
    bespoke, adapter = _bespoke(), _adapter()
    _run_first_call_writes(bespoke)
    _run_first_call_writes(adapter)
    _assert_fields_equal(bespoke, adapter)
    assert adapter.current_lat_f == 4
    assert adapter.is_initialized is True


def test_reset_restores_defaults_identically() -> None:
    bespoke, adapter = _bespoke(), _adapter()
    _run_first_call_writes(bespoke)
    _run_first_call_writes(adapter)
    bespoke.reset()
    adapter.reset()
    _assert_fields_equal(bespoke, adapter)
    _assert_fields_equal(adapter, _bespoke())


def test_init_state_allocates_expected_shapes() -> None:
    adapter = _adapter()
    adapter.init_state(DEVICE, NUM_LAYERS)
    assert adapter.local_end_index is not None and adapter.global_end_index is not None
    for idx_list in (adapter.local_end_index, adapter.global_end_index):
        assert len(idx_list) == NUM_LAYERS
        for idx in idx_list:
            assert idx.shape == (1,)
            assert idx.dtype == torch.long
            assert int(idx.item()) == 0


def test_in_place_end_index_mutation_is_seen_by_a_new_view() -> None:
    # The transformer mutates the end-index tensors in place during forward;
    # the next request builds a fresh adapter and must see those mutations.
    manager = SessionStateManager(max_sessions=1)
    first = LingbotWorldFastStateAdapter("session-a", manager)
    first.init_state(DEVICE, NUM_LAYERS)
    assert first.local_end_index is not None
    first.local_end_index[0].fill_(7)

    second = LingbotWorldFastStateAdapter("session-a", manager)
    assert second.local_end_index is not None
    assert int(second.local_end_index[0].item()) == 7


def test_returning_session_keeps_state_and_drop_releases_it() -> None:
    manager = SessionStateManager(max_sessions=1)
    first = LingbotWorldFastStateAdapter("session-a", manager)
    _run_first_call_writes(first)

    returning = LingbotWorldFastStateAdapter("session-a", manager)
    assert returning.session_id == "session-a"
    assert returning.current_lat_f == 4

    assert manager.drop_session("session-a") is True
    fresh = LingbotWorldFastStateAdapter("session-a", manager)
    _assert_fields_equal(fresh, _bespoke())


def test_retention_cap_of_one_mirrors_the_bespoke_single_slot() -> None:
    # The bespoke pipeline holds one session's state at a time; with the
    # default cap the manager retains one too, so a session displaced by
    # another reads as fresh (session_id None) and takes the reset branch,
    # exactly as the overwritten bespoke singleton would.
    manager = SessionStateManager(max_sessions=1)
    _run_first_call_writes(LingbotWorldFastStateAdapter("session-a", manager))
    _run_first_call_writes(LingbotWorldFastStateAdapter("session-b", manager))

    displaced = LingbotWorldFastStateAdapter("session-a", manager)
    assert displaced.session_id is None
    assert displaced.is_initialized is False


def test_pipeline_reset_trigger_is_equivalent() -> None:
    # Replicates the fresh-versus-extension conditional in
    # ``LingbotWorldFastPipeline.forward`` on both state implementations.
    def should_reset(state: LingbotWorldFastState | LingbotWorldFastStateAdapter, incoming: str) -> bool:
        return state.session_id is None or state.session_id != incoming

    manager = SessionStateManager(max_sessions=1)
    bespoke, adapter = _bespoke(), LingbotWorldFastStateAdapter("session-a", manager)
    assert should_reset(bespoke, "session-a") is should_reset(adapter, "session-a") is True

    _run_first_call_writes(bespoke)
    _run_first_call_writes(adapter)
    assert should_reset(bespoke, "session-a") is should_reset(adapter, "session-a") is False
    assert should_reset(bespoke, "session-b") is should_reset(adapter, "session-b") is True


def test_byte_accounting_covers_carried_tensors() -> None:
    manager = SessionStateManager(max_sessions=1)
    adapter = LingbotWorldFastStateAdapter("session-a", manager)
    _run_first_call_writes(adapter)

    stats = manager.stats()
    latent = adapter.last_decoded_latent
    assert latent is not None
    assert stats["nbytes:cpu"] >= latent.untyped_storage().nbytes()
    assert stats["sessions"] == 1

    manager.drop_session("session-a")
    assert manager.stats()["total_nbytes"] == 0
