# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for ``FixedState`` (RFC #4480).

``FixedState`` backs a constant-memory recurrent state -- the Gated-DeltaNet
hidden state of an AR world model such as SANA-WM. These tests drive it directly
with tiny CPU tensors (no model, no GPU) and pin the behaviour the AR adapter
relies on: fixed-size allocation, in-place commit that copies values, a stable
buffer identity across the rollout, snapshot/restore round-trips, and the
never-evictable guarantee.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.world_models.memory import FixedState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# SANA-WM GDN hidden state: state_kv (B, H, D, D), state_z (B, H, D, 1).
BATCH, HEADS, DIM = 1, 2, 4
DTYPE, DEVICE = torch.float32, torch.device("cpu")
SHAPES = {
    "state_kv": (BATCH, HEADS, DIM, DIM),
    "state_z": (BATCH, HEADS, DIM, 1),
}


def _allocated() -> FixedState:
    state = FixedState()
    state.allocate(shapes=SHAPES, dtype=DTYPE, device=DEVICE)
    return state


def _random_payload() -> dict[str, torch.Tensor]:
    return {name: torch.randn(shape) for name, shape in SHAPES.items()}


def test_allocate_zeros_with_requested_shapes() -> None:
    state = _allocated()
    assert state.resident
    view = state.view()
    assert set(view) == set(SHAPES)
    for name, shape in SHAPES.items():
        assert view[name].shape == shape
        assert view[name].dtype == DTYPE
        torch.testing.assert_close(view[name], torch.zeros(shape))


def test_allocate_requires_at_least_one_shape() -> None:
    with pytest.raises(ValueError):
        FixedState().allocate(shapes={}, dtype=DTYPE, device=DEVICE)


def test_commit_overwrites_in_place_keeping_buffer_identity() -> None:
    state = _allocated()
    before = state.view()
    kv_buf, z_buf = before["state_kv"], before["state_z"]
    payload = _random_payload()
    state.commit(payload)
    after = state.view()
    # Same tensor objects (fixed-size, no realloc): copy_ wrote into them.
    assert after["state_kv"] is kv_buf
    assert after["state_z"] is z_buf
    torch.testing.assert_close(after["state_kv"], payload["state_kv"])
    torch.testing.assert_close(after["state_z"], payload["state_z"])


def test_commit_copies_values_not_reference() -> None:
    # A later mutation of the source must not corrupt the stored state.
    state = _allocated()
    payload = _random_payload()
    state.commit(payload)
    snapshot = {name: tensor.clone() for name, tensor in payload.items()}
    for tensor in payload.values():
        tensor.add_(1.0)  # mutate the source in place after commit
    for name in SHAPES:
        torch.testing.assert_close(state.view()[name], snapshot[name])


def test_repeated_commit_carries_state_like_an_ar_rollout() -> None:
    state = _allocated()
    for _ in range(3):
        payload = _random_payload()
        state.commit(payload)
        for name in SHAPES:
            torch.testing.assert_close(state.view()[name], payload[name])


def test_commit_rejects_unknown_or_missing_keys() -> None:
    state = _allocated()
    with pytest.raises(ValueError):
        state.commit({"state_kv": torch.zeros(SHAPES["state_kv"])})  # missing state_z
    with pytest.raises(ValueError):
        state.commit({**_random_payload(), "extra": torch.zeros(1)})


def test_commit_rejects_shape_mismatch() -> None:
    state = _allocated()
    bad = _random_payload()
    bad["state_kv"] = torch.randn(BATCH, HEADS, DIM, DIM + 1)
    with pytest.raises(ValueError):
        state.commit(bad)


def test_commit_before_allocate_raises() -> None:
    with pytest.raises(RuntimeError):
        FixedState().commit(_random_payload())


def test_snapshot_restore_round_trip() -> None:
    state = _allocated()
    original = _random_payload()
    state.commit(original)
    snap = state.snapshot()
    # Snapshot is decoupled: overwriting the live state does not change it.
    state.commit(_random_payload())
    for name in SHAPES:
        assert not torch.allclose(state.view()[name], snap[name])
    state.restore(snap)
    for name in SHAPES:
        torch.testing.assert_close(state.view()[name], original[name])


def test_nbytes_is_constant_and_matches_buffers() -> None:
    state = _allocated()
    expected = sum(torch.zeros(shape, dtype=DTYPE).numel() * 4 for shape in SHAPES.values())
    assert state.nbytes == expected
    state.commit(_random_payload())
    assert state.nbytes == expected  # fixed size: commit never changes the footprint


def test_evict_raises_never_evictable() -> None:
    state = _allocated()
    with pytest.raises(RuntimeError):
        state.evict()
    # State is untouched by the refused eviction.
    assert state.resident


def test_not_recomputable() -> None:
    state = _allocated()
    assert state.recompute_source is None
    assert state.recomputable is False


def test_reset_clears_to_unallocated() -> None:
    state = _allocated()
    state.commit(_random_payload())
    state.reset()
    assert not state.resident
    assert state.nbytes == 0
    with pytest.raises(RuntimeError):
        state.view()
