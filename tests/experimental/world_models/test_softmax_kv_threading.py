# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the autoregressive softmax sliding-window KV cache (Fix 3).

The ``SoftmaxKvState`` carrier is threaded through the SANA-WM forward levels
(transformer -> block -> attention -> softmax SDPA) for the every-N-th softmax
blocks. These tests build a tiny softmax self-attention module on CPU and pin the
threading contract, mirroring ``test_gdn_state_threading``:

* passing the carrier with no cached window is **non-invasive** -- the output is
  identical to the plain forward;
* the current chunk's ``(key, value)`` is captured under ``(block_idx, False)``;
* prepending a cached window changes the output but keeps the query token count;
* ``frame_offset`` shifts the RoPE temporal positions (so a chunk's Q/K line up
  with a cached window whose keys carry absolute positions).
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SanaWmSelfAttention,
    SanaWmWanRotaryPosEmbed,
    SoftmaxKvState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN, HEAD_DIM, HEADS = 8, 4, 2
FRAMES, GRID = 2, 2  # spatial 2x2 -> S=4; N = FRAMES*4 = 8
SPATIAL_SHAPE = (FRAMES, GRID, GRID)
N = FRAMES * GRID * GRID


def _softmax_attn(block_idx: int = 3) -> SanaWmSelfAttention:
    config = SanaWmConfig(
        num_blocks=4,
        hidden_size=HIDDEN,
        linear_head_dim=HEAD_DIM,
        softmax_every_n=4,
        conv_kernel_size=0,
        qk_norm=False,
        cam_attn_compress=1,
        pos_embed_type="none",
    )
    attn = SanaWmSelfAttention(config, use_gdn=False, use_vllm_parallel_layers=False)
    attn.block_idx = block_idx
    attn.eval()
    return attn


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, N, HIDDEN)


def test_carrier_is_non_invasive_and_captures_kv() -> None:
    attn = _softmax_attn(block_idx=3)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = SoftmaxKvState(capture=True)
        out = attn(x, SPATIAL_SHAPE, None, None, softmax_state=state)
    torch.testing.assert_close(out, base)  # no cached window -> output unchanged
    assert set(state.final) == {(3, False)}
    key, value = state.final[(3, False)]
    assert key.shape[2] == N and value.shape[2] == N  # (B, H, N, D)
    assert key.shape[0] == 1 and key.shape[-1] == HEAD_DIM


def test_capture_false_records_nothing() -> None:
    attn = _softmax_attn(block_idx=3)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = SoftmaxKvState(capture=False)
        out = attn(x, SPATIAL_SHAPE, None, None, softmax_state=state)
    torch.testing.assert_close(out, base)
    assert state.final == {}


def test_cached_window_changes_output_but_keeps_token_count() -> None:
    attn = _softmax_attn(block_idx=3)
    x = _x()
    with torch.no_grad():
        # First capture the current chunk's K/V to learn the (B, H, D) shape.
        probe = SoftmaxKvState(capture=True)
        base = attn(x, SPATIAL_SHAPE, None, None, softmax_state=probe)
        key, _ = probe.final[(3, False)]
        b, h, _, d = key.shape
        window_k = torch.randn(b, h, 2 * GRID * GRID, d)  # a 2-frame cached window
        window_v = torch.randn(b, h, 2 * GRID * GRID, d)
        seeded = attn(
            x,
            SPATIAL_SHAPE,
            None,
            None,
            softmax_state=SoftmaxKvState(init={(3, False): (window_k, window_v)}, capture=False),
        )
    assert seeded.shape == base.shape  # queries stay current-chunk only (N tokens)
    assert not torch.allclose(seeded, base)  # attending over the window changes output


def test_rope_frame_offset_shifts_temporal_positions() -> None:
    rope = SanaWmWanRotaryPosEmbed(HEAD_DIM)
    dev = torch.device("cpu")
    # With H=W=1 the spatial components are position-0 (constant), so only the
    # temporal axis varies; an offset must reproduce the tail of a longer grid.
    full = rope((4, 1, 1), dev, frame_offset=0)  # positions 0..3
    shifted = rope((2, 1, 1), dev, frame_offset=2)  # positions 2..3
    torch.testing.assert_close(shifted, full[:, :, 2 * 1 * 1 : 4 * 1 * 1, :])
    # Offset 0 is the legacy behaviour.
    torch.testing.assert_close(rope((2, 1, 1), dev, frame_offset=0), full[:, :, :2, :])
