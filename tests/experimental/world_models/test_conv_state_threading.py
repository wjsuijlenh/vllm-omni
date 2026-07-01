# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the autoregressive temporal-conv carrier threading (Fix 5).

The ``ConvArState`` carrier threads each block's temporal-conv boundary state
through the SANA-WM forward levels: the attention key short-conv (``conv_k``,
slot ``"k"``) and the FFN temporal conv (``t_conv``, slot ``"ffn_t"``). These
tests build the tiny CPU reference modules and pin the same threading contract
the GDN/softmax carriers use:

* passing the carrier with an empty ``init`` is **non-invasive** -- the output is
  identical to the plain forward (an absent seed means zero-pad, exactly the
  production path);
* the trailing conv-input frames are captured under the block's own index with
  the right shape (``kernel-1`` frames for the causal ``conv_k``, ``kernel//2``
  for the symmetric ``t_conv``);
* seeding the carrier actually feeds the conv (it changes the output);
* ``capture=False`` records nothing.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    ConvArState,
    SanaWmMbConvFfn,
    SanaWmSelfAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN, HEAD_DIM, HEADS = 8, 4, 2
FRAMES, GRID = 4, 2  # F=4 > conv_kernel_size-1=3, so conv_k captures a full window
SPATIAL_SHAPE = (FRAMES, GRID, GRID)
N = FRAMES * GRID * GRID
SPATIAL_TOKENS = GRID * GRID
CONV_KERNEL = 4
T_KERNEL = 3


def _config() -> SanaWmConfig:
    return SanaWmConfig(
        num_blocks=4,
        hidden_size=HIDDEN,
        linear_head_dim=HEAD_DIM,
        softmax_every_n=4,
        conv_kernel_size=CONV_KERNEL,
        k_conv_only=True,
        t_kernel_size=T_KERNEL,
        qk_norm=False,
        cam_attn_compress=1,
        pos_embed_type="none",
    )


def _gdn_attn(block_idx: int = 0) -> SanaWmSelfAttention:
    attn = SanaWmSelfAttention(_config(), use_gdn=True, use_vllm_parallel_layers=False)
    attn.block_idx = block_idx
    # The short-conv ships identity-initialised (only the last tap is 1), so the
    # left context would not matter; randomise it to a genuine depthwise conv.
    with torch.no_grad():
        attn.conv_k.weight.normal_()
    attn.eval()
    return attn


def _ffn(block_idx: int = 0) -> SanaWmMbConvFfn:
    ffn = SanaWmMbConvFfn(_config())
    ffn.block_idx = block_idx
    with torch.no_grad():
        ffn.t_conv.weight.normal_()
    ffn.eval()
    return ffn


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, N, HIDDEN)


# -- attention conv_k (slot "k") ---------------------------------------------


def test_conv_k_carrier_is_non_invasive_and_captures() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = ConvArState(capture=True)
        out = attn(x, SPATIAL_SHAPE, None, None, None, None, state)
    torch.testing.assert_close(out, base)  # empty init == zero-pad == plain forward
    assert set(state.final) == {(0, "k")}
    # Trailing kernel-1 frames of the temporal conv input: (B*S, K-1, C).
    assert state.final[(0, "k")].shape == (SPATIAL_TOKENS, CONV_KERNEL - 1, HIDDEN)


def test_conv_k_capture_false_records_nothing() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        base = attn(x, SPATIAL_SHAPE, None, None)
        state = ConvArState(capture=False)
        out = attn(x, SPATIAL_SHAPE, None, None, None, None, state)
    torch.testing.assert_close(out, base)
    assert state.final == {}


def test_conv_k_captured_under_block_index() -> None:
    attn = _gdn_attn(block_idx=2)
    with torch.no_grad():
        state = ConvArState(capture=True)
        attn(_x(), SPATIAL_SHAPE, None, None, None, None, state)
    assert set(state.final) == {(2, "k")}


def test_conv_k_seeding_changes_output() -> None:
    attn = _gdn_attn(block_idx=0)
    x = _x()
    with torch.no_grad():
        zero_seed = attn(x, SPATIAL_SHAPE, None, None, None, None, ConvArState(capture=True))
        seed = torch.randn(SPATIAL_TOKENS, CONV_KERNEL - 1, HIDDEN)
        seeded = attn(x, SPATIAL_SHAPE, None, None, None, None, ConvArState(init={(0, "k"): seed}, capture=True))
    assert not torch.allclose(seeded, zero_seed)


# -- FFN t_conv (slot "ffn_t") -----------------------------------------------


def test_ffn_t_conv_is_non_invasive_and_captures() -> None:
    ffn = _ffn(block_idx=1)
    x = _x()
    with torch.no_grad():
        base = ffn(x, SPATIAL_SHAPE)
        state = ConvArState(capture=True)
        out = ffn(x, SPATIAL_SHAPE, state)
    torch.testing.assert_close(out, base)
    assert set(state.final) == {(1, "ffn_t")}
    # Trailing kernel//2 frames of the t_conv input: (B, C, t_padding, H*W).
    assert state.final[(1, "ffn_t")].shape == (1, HIDDEN, T_KERNEL // 2, SPATIAL_TOKENS)


def test_ffn_t_conv_capture_false_records_nothing() -> None:
    ffn = _ffn(block_idx=1)
    x = _x()
    with torch.no_grad():
        base = ffn(x, SPATIAL_SHAPE)
        state = ConvArState(capture=False)
        out = ffn(x, SPATIAL_SHAPE, state)
    torch.testing.assert_close(out, base)
    assert state.final == {}


def test_ffn_t_conv_seeding_changes_output() -> None:
    ffn = _ffn(block_idx=1)
    x = _x()
    with torch.no_grad():
        zero_seed = ffn(x, SPATIAL_SHAPE, ConvArState(capture=True))
        seed = torch.randn(1, HIDDEN, T_KERNEL // 2, SPATIAL_TOKENS)
        seeded = ffn(x, SPATIAL_SHAPE, ConvArState(init={(1, "ffn_t"): seed}, capture=True))
    assert not torch.allclose(seeded, zero_seed)


def test_seed_for_defaults() -> None:
    empty = ConvArState()
    assert empty.seed_for(5, "k") is None
    frames = torch.zeros(SPATIAL_TOKENS, CONV_KERNEL - 1, HIDDEN)
    populated = ConvArState(init={(3, "k"): frames})
    assert populated.seed_for(3, "k") is frames
    assert populated.seed_for(3, "ffn_t") is None
    assert populated.seed_for(0, "k") is None
