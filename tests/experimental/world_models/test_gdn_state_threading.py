# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the autoregressive GDN-state carrier threading (Step 3b).

The ``GdnArState`` carrier is threaded through the SANA-WM forward levels
(transformer -> block -> attention -> GDN scan). These tests build a tiny GDN
self-attention module on CPU (the non-vLLM-parallel reference path) and pin the
threading contract:

* passing the carrier is **non-invasive** -- the block output is identical to the
  plain forward (the AR plumbing only observes state, never perturbs the math);
* the per-block terminal ``(state_kv, state_z)`` is captured with the right
  shapes under the block's own index;
* seeding the carrier actually feeds the scan (it changes the output).
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import GdnArState, SanaWmSelfAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN, HEAD_DIM, HEADS = 8, 4, 2
FRAMES, GRID = 2, 2  # spatial 2x2 -> S=4; N = FRAMES*4 = 8
SPATIAL_SHAPE = (FRAMES, GRID, GRID)
N = FRAMES * GRID * GRID


def _gdn_attn(block_idx: int = 0) -> SanaWmSelfAttention:
    config = SanaWmConfig(
        num_blocks=4,
        hidden_size=HIDDEN,
        linear_head_dim=HEAD_DIM,
        softmax_every_n=4,
        conv_kernel_size=0,  # skip the temporal K-conv for a small CPU module
        qk_norm=False,
        cam_attn_compress=1,
        pos_embed_type="none",
    )
    attn = SanaWmSelfAttention(config, use_gdn=True, use_vllm_parallel_layers=False)
    attn.block_idx = block_idx
    attn.eval()
    return attn


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, N, HIDDEN)


def test_carrier_is_non_invasive_and_captures_state() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = GdnArState(capture=True)
        out = attn(x, SPATIAL_SHAPE, None, None, state)
    torch.testing.assert_close(out, base)  # threading does not change the output
    assert set(state.final) == {0}
    state_kv, state_z = state.final[0]
    assert state_kv.shape == (1, HEADS, HEAD_DIM, HEAD_DIM)
    assert state_z.shape == (1, HEADS, HEAD_DIM, 1)


def test_capture_false_records_nothing() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = GdnArState(capture=False)
        out = attn(x, SPATIAL_SHAPE, None, None, state)
    torch.testing.assert_close(out, base)
    assert state.final == {}


def test_state_captured_under_block_index() -> None:
    attn = _gdn_attn(block_idx=2)
    with torch.no_grad():
        state = GdnArState(capture=True)
        attn(_x(), SPATIAL_SHAPE, None, None, state)
    assert set(state.final) == {2}


def test_seeding_changes_output() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        zero_seed = attn(x, SPATIAL_SHAPE, None, None, GdnArState(capture=True))
        seed_kv = torch.randn(1, HEADS, HEAD_DIM, HEAD_DIM)
        seed_z = torch.randn(1, HEADS, HEAD_DIM, 1)
        seeded = attn(x, SPATIAL_SHAPE, None, None, GdnArState(init={0: (seed_kv, seed_z)}, capture=True))
    assert not torch.allclose(seeded, zero_seed)


def test_seed_for_defaults() -> None:
    empty = GdnArState()
    assert empty.seed_for(5) == (None, None)
    kv, z = torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM), torch.zeros(1, HEADS, HEAD_DIM, 1)
    populated = GdnArState(init={3: (kv, z)})
    assert populated.seed_for(3) == (kv, z)
    assert populated.seed_for(0) == (None, None)
