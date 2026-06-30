# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the SANA-WM autoregressive chunk-rollout orchestration (Step 4).

These pin the model-free orchestration the pipeline's AR backend drives: chunk
planning, per-denoise-step state seeding/capture, and committing a chunk's
terminal state for the next chunk. The integrated test simulates a multi-chunk
rollout against a real ``SanaWmArStateAdapter`` with a stub denoiser, proving the
GDN state genuinely carries chunk-to-chunk (and stays isolated per CFG branch).
End-to-end numerical equivalence with the transformer is a later GPU step.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.ar_rollout import (
    FIRST_CHUNK_PLUS_ONE,
    UNIFORM,
    ChunkSpan,
    commit_chunk_state,
    plan_chunks,
    slice_camera_frames,
    step_state,
)
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import GdnArState
from vllm_omni.experimental.world_models.adapters.state_sana_wm_ar_adapter import SanaWmArStateAdapter
from vllm_omni.experimental.world_models.memory import SessionMemoryManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BATCH, BLOCKS, HEADS, DIM, EVERY_N = 1, 8, 2, 4, 4
GDN_LAYERS = (0, 1, 2, 4, 5, 6)


# -- chunk planning -------------------------------------------------------


def test_plan_chunks_first_chunk_plus_one() -> None:
    spans = plan_chunks(10, 3, FIRST_CHUNK_PLUS_ONE)
    # first chunk = 3+1 = 4 frames, then 3, 3 (last truncated to fit).
    assert [(s.start, s.end) for s in spans] == [(0, 4), (4, 7), (7, 10)]
    assert spans[0].is_first and not spans[1].is_first
    assert [s.index for s in spans] == [0, 1, 2]
    assert [s.num_frames for s in spans] == [4, 3, 3]


def test_plan_chunks_uniform() -> None:
    spans = plan_chunks(7, 3, UNIFORM)
    assert [(s.start, s.end) for s in spans] == [(0, 3), (3, 6), (6, 7)]


def test_plan_chunks_covers_exactly_and_contiguous() -> None:
    for frames in (1, 3, 4, 5, 9, 13):
        for strategy in (FIRST_CHUNK_PLUS_ONE, UNIFORM):
            spans = plan_chunks(frames, 3, strategy)
            assert spans[0].start == 0
            assert spans[-1].end == frames
            for prev, nxt in zip(spans, spans[1:]):
                assert prev.end == nxt.start  # contiguous, no gaps/overlap


def test_plan_chunks_validates_inputs() -> None:
    with pytest.raises(ValueError):
        plan_chunks(0, 3)
    with pytest.raises(ValueError):
        plan_chunks(10, 0)
    with pytest.raises(ValueError):
        plan_chunks(10, 3, "bogus")


# -- per-step state seeding / capture ------------------------------------


class _StubStore:
    """Minimal GDN-state store satisfying the rollout's Protocol."""

    def __init__(self) -> None:
        self.gdn_layers = (0, 1, 2)
        self._state = {
            (layer, neg): {"state_kv": torch.full((1, 2, 2, 2), float(layer + (10 if neg else 0))),
                           "state_z": torch.zeros(1, 2, 2, 1)}
            for layer in self.gdn_layers
            for neg in (False, True)
        }

    def get_gdn_state(self, layer_index: int, is_negative: bool = False) -> dict[str, torch.Tensor]:
        return self._state[(layer_index, is_negative)]

    def commit_gdn_state(self, layer_index, state_kv, state_z, is_negative=False) -> None:  # type: ignore[no-untyped-def]
        self._state[(layer_index, is_negative)] = {"state_kv": state_kv, "state_z": state_z}


def test_step_state_first_chunk_seeds_from_zeros() -> None:
    state = step_state(_StubStore(), is_first_chunk=True, capture=False)
    assert state.init == {}  # no seed -> forward scan starts from zeros
    assert state.capture is False


def test_step_state_later_chunk_seeds_from_store() -> None:
    store = _StubStore()
    state = step_state(store, is_first_chunk=False, capture=True, is_negative=False)
    assert set(state.init) == set(store.gdn_layers)
    torch.testing.assert_close(state.init[1][0], store.get_gdn_state(1, False)["state_kv"])
    assert state.capture is True


def test_step_state_branch_selects_correct_state() -> None:
    store = _StubStore()
    pos = step_state(store, is_first_chunk=False, capture=False, is_negative=False)
    neg = step_state(store, is_first_chunk=False, capture=False, is_negative=True)
    assert not torch.allclose(pos.init[0][0], neg.init[0][0])


def test_commit_chunk_state_writes_back() -> None:
    store = _StubStore()
    state = GdnArState()
    new_kv, new_z = torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 1)
    state.final[1] = (new_kv, new_z)
    commit_chunk_state(store, state, is_negative=False)
    torch.testing.assert_close(store.get_gdn_state(1, False)["state_kv"], new_kv)


# -- camera slicing -------------------------------------------------------


def test_slice_camera_frames() -> None:
    cam = torch.arange(6 * 20).reshape(1, 6, 20).float()  # (B, T_latent, 20) raymap
    span = ChunkSpan(index=1, start=2, end=5)
    sliced = slice_camera_frames(cam, span, frame_dim=1)
    assert sliced.shape == (1, 3, 20)
    torch.testing.assert_close(sliced, cam[:, 2:5])
    assert slice_camera_frames(None, span, frame_dim=1) is None


def test_slice_camera_frames_too_few() -> None:
    cam = torch.zeros(1, 3, 20)
    with pytest.raises(ValueError):
        slice_camera_frames(cam, ChunkSpan(index=0, start=0, end=5), frame_dim=1)


# -- integrated multi-chunk rollout with a real adapter -------------------


def _adapter() -> SanaWmArStateAdapter:
    adapter = SanaWmArStateAdapter(
        "roll", SessionMemoryManager(), num_blocks=BLOCKS, num_heads=HEADS, head_dim=DIM, softmax_every_n=EVERY_N
    )
    adapter.create_state(BATCH, torch.float32, torch.device("cpu"))
    return adapter


def _stub_denoise(state: GdnArState, chunk_index: int) -> None:
    """Stand in for the transformer forward: on the capture step, write a terminal
    state that depends on the seed, so carry can be verified downstream."""
    if not state.capture:
        return
    for layer in GDN_LAYERS:
        seed_kv, seed_z = state.init.get(layer, (None, None))
        base_kv = seed_kv if seed_kv is not None else torch.zeros(BATCH, HEADS, DIM, DIM)
        base_z = seed_z if seed_z is not None else torch.zeros(BATCH, HEADS, DIM, 1)
        state.final[layer] = (base_kv + (chunk_index + 1), base_z + (chunk_index + 1))


def test_integrated_rollout_carries_state_across_chunks() -> None:
    adapter = _adapter()
    spans = plan_chunks(num_latent_frames=10, chunk_size=3, strategy=FIRST_CHUNK_PLUS_ONE)
    num_steps = 4
    seen_init_first_layer: list[bool] = []

    for span in spans:
        last_state: GdnArState | None = None
        for step in range(num_steps):
            state = step_state(adapter, is_first_chunk=span.is_first, capture=(step == num_steps - 1))
            if step == 0:
                seen_init_first_layer.append(bool(state.init))
            _stub_denoise(state, span.index)
            last_state = state
        assert last_state is not None
        commit_chunk_state(adapter, last_state)
        adapter.chunk_index = span.index + 1

    # Chunk 0 seeds from zeros (empty init); every later chunk seeds from a
    # non-empty carried state.
    assert seen_init_first_layer == [False, True, True]

    # The recurrence accumulates +1, +2, +3 across the three chunks for each GDN
    # layer (seed of chunk k is the committed state of chunk k-1).
    expected = 1 + 2 + 3
    final = adapter.get_gdn_state(0)["state_kv"]
    torch.testing.assert_close(final, torch.full((BATCH, HEADS, DIM, DIM), float(expected)))
    assert adapter.chunk_index == 3


def test_integrated_rollout_branches_do_not_alias() -> None:
    adapter = _adapter()
    spans = plan_chunks(7, 3, UNIFORM)
    for span in spans:
        for is_neg in (False, True):
            state = step_state(adapter, is_first_chunk=span.is_first, capture=True, is_negative=is_neg)
            # Different per-branch terminal state.
            bump = 100 if is_neg else 1
            for layer in GDN_LAYERS:
                seed_kv, _ = state.init.get(layer, (None, None))
                base = seed_kv if seed_kv is not None else torch.zeros(BATCH, HEADS, DIM, DIM)
                state.final[layer] = (base + bump, torch.zeros(BATCH, HEADS, DIM, 1))
            commit_chunk_state(adapter, state, is_negative=is_neg)
        adapter.chunk_index = span.index + 1
    pos = adapter.get_gdn_state(0, is_negative=False)["state_kv"]
    neg = adapter.get_gdn_state(0, is_negative=True)["state_kv"]
    assert not torch.allclose(pos, neg)
