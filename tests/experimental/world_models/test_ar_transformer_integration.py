# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 (CPU) integration: AR state carry through a real SANA-WM transformer.

Steps 1-4 are unit-tested in isolation (memory objects, adapter, GDN scan carry,
carrier threading, rollout orchestration). This test composes them through a real
(tiny, randomly-initialised) ``SanaWmTransformer3DModel`` on CPU -- no weights, no
GPU -- to prove the whole stack wires together: the ``GdnArState`` carrier reaches
every GDN block of the full model, is non-invasive when not seeding, and a two-
chunk rollout driven by the ``ar_rollout`` helpers + ``SanaWmArStateAdapter``
genuinely carries state that changes the model's output.

Numerical equivalence against the upstream chunk-causal checkpoint is the GPU
step (real weights, run through gpuq); this pins the integration on CPU first.
"""

from __future__ import annotations

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.sana_wm.ar_rollout import (
    FIRST_CHUNK_PLUS_ONE,
    commit_chunk_state,
    plan_chunks,
    step_state,
)
from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SANA_WM_STAGE1_PROMPT_CHANNELS,
    GdnArState,
    SanaWmTransformer3DModel,
)
from vllm_omni.experimental.world_models.adapters.state_sana_wm_ar_adapter import SanaWmArStateAdapter
from vllm_omni.experimental.world_models.memory import SessionMemoryManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LATENT_CH = 128  # vae_latent_dim, the transformer's input channel count
GDN_LAYERS = (0, 1, 2)
SOFTMAX_LAYERS = (3,)


def _tiny_config() -> SanaWmConfig:
    return SanaWmConfig(
        num_blocks=4,
        hidden_size=8,
        linear_head_dim=4,  # -> 2 heads, head_dim 4
        softmax_every_n=4,  # softmax at block 3; GDN at 0,1,2
        conv_kernel_size=0,
        qk_norm=False,
        cam_attn_compress=1,
        pos_embed_type="wan_rope",
        patch_size=(1, 1, 1),
        mlp_ratio=2.0,
    )


@pytest.fixture(scope="module")
def transformer() -> SanaWmTransformer3DModel:
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        model = SanaWmTransformer3DModel(config=_tiny_config(), quant_config=None, prefix="transformer")
    model.eval()
    return model


def _latents(frames: int) -> torch.Tensor:
    return torch.randn(1, LATENT_CH, frames, 2, 2)


def _prompt() -> torch.Tensor:
    return torch.randn(1, 3, SANA_WM_STAGE1_PROMPT_CHANNELS)


def test_full_model_threads_state_to_every_gdn_block(transformer: SanaWmTransformer3DModel) -> None:
    lat, ehs = _latents(2), _prompt()
    with torch.no_grad():
        base = transformer(lat, 500.0, encoder_hidden_states=ehs)
        state = GdnArState(capture=True)
        out = transformer(lat, 500.0, encoder_hidden_states=ehs, gdn_state=state)
    assert torch.isfinite(base).all()
    # Capturing state is non-invasive: identical output to the plain forward.
    torch.testing.assert_close(out, base)
    # Exactly the GDN blocks capture state; the softmax block does not.
    assert sorted(state.final) == list(GDN_LAYERS)
    assert all(layer not in state.final for layer in SOFTMAX_LAYERS)


def test_frame_aware_path_also_threads_state(transformer: SanaWmTransformer3DModel) -> None:
    # Per-frame timestep (B, 1, F) dispatches the block's frame-aware path.
    lat, ehs = _latents(2), _prompt()
    model_timestep = torch.full((1, 1, 2), 500.0)
    with torch.no_grad():
        state = GdnArState(capture=True)
        out = transformer(lat, model_timestep, encoder_hidden_states=ehs, gdn_state=state)
    assert torch.isfinite(out).all()
    assert sorted(state.final) == list(GDN_LAYERS)


def test_seeding_changes_full_model_output(transformer: SanaWmTransformer3DModel) -> None:
    lat, ehs = _latents(2), _prompt()
    seed = {
        layer: (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 1))
        for layer in GDN_LAYERS
    }
    with torch.no_grad():
        zero_seed = transformer(lat, 500.0, encoder_hidden_states=ehs, gdn_state=GdnArState(capture=False))
        seeded = transformer(lat, 500.0, encoder_hidden_states=ehs, gdn_state=GdnArState(init=seed, capture=False))
    assert not torch.allclose(zero_seed, seeded)  # carried state feeds the scan


def test_two_chunk_rollout_through_real_model(transformer: SanaWmTransformer3DModel) -> None:
    config = _tiny_config()
    adapter = SanaWmArStateAdapter.from_config("integ", SessionMemoryManager(), config)
    adapter.create_state(1, torch.float32, torch.device("cpu"))
    assert adapter.gdn_layers == GDN_LAYERS

    ehs = _prompt()
    spans = plan_chunks(num_latent_frames=7, chunk_size=3, strategy=FIRST_CHUNK_PLUS_ONE)
    assert [s.num_frames for s in spans] == [4, 3]  # first_chunk_plus_one

    outputs = []
    chunk1_seeded_init: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for span in spans:
        lat = _latents(span.num_frames)
        state = step_state(adapter, is_first_chunk=span.is_first, capture=True, is_negative=False)
        if not span.is_first:
            chunk1_seeded_init = dict(state.init)
        with torch.no_grad():
            out = transformer(lat, 500.0, encoder_hidden_states=ehs, gdn_state=state)
        assert torch.isfinite(out).all()
        outputs.append(out)
        commit_chunk_state(adapter, state, is_negative=False)
        adapter.chunk_index = span.index + 1

    # Chunk 1 was seeded from chunk 0's committed state (non-empty, all GDN layers).
    assert set(chunk1_seeded_init) == set(GDN_LAYERS)
    # The adapter holds the terminal state after the final chunk.
    assert adapter.chunk_index == 2
    for layer in GDN_LAYERS:
        assert torch.isfinite(adapter.get_gdn_state(layer)["state_kv"]).all()

    # Chunk 1, re-run WITHOUT the carried seed, must differ from the seeded run --
    # i.e. the cross-chunk carry actually influenced the second chunk.
    torch.manual_seed(123)
    lat1 = _latents(spans[1].num_frames)
    with torch.no_grad():
        seeded = transformer(
            lat1, 500.0, encoder_hidden_states=ehs,
            gdn_state=GdnArState(init=chunk1_seeded_init, capture=False),
        )
        unseeded = transformer(
            lat1, 500.0, encoder_hidden_states=ehs, gdn_state=GdnArState(capture=False)
        )
    assert not torch.allclose(seeded, unseeded)
