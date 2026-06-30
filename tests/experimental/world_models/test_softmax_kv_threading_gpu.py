# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke test for the SANA-WM softmax sliding-window KV cache (Fix 3).

The CPU tests run the softmax window through SDPA's math backend in fp32. This
exercises the realistic path on an actual GPU: a camera-enabled transformer in
bfloat16, where the softmax main and UCPE camera branches hit FlashAttention with
the cached window prepended (and the cam branch's head-dim padding). It pins the
same contract as the CPU integration test -- non-invasive capture, both streams
recorded, a seeded window changing the output -- but on the kernels that only run
on CUDA.

Skipped without CUDA. Run through ``gpuq`` (the shared-A100 queue).
"""

from __future__ import annotations

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SANA_WM_STAGE1_PROMPT_CHANNELS,
    SanaWmTransformer3DModel,
    SoftmaxKvState,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="softmax-window GPU path requires CUDA"),
]

LATENT_CH = 128
CAM_SOFTMAX_LAYER = 3


def _cam_config() -> SanaWmConfig:
    return SanaWmConfig(
        num_blocks=4,
        hidden_size=32,
        # head_dim 16: UCPE-valid (divisible by 8) AND >= 16 so the fused GDN
        # Triton kernel's tl.dot (K >= 16) compiles -- on GPU the GDN blocks use
        # the fused kernel, unlike the CPU reference scan.
        linear_head_dim=16,
        softmax_every_n=4,
        conv_kernel_size=0,
        qk_norm=True,
        cam_attn_compress=1,
        pos_embed_type="wan_rope",
        patch_size=(1, 1, 1),
        mlp_ratio=2.0,
    )


def _model() -> SanaWmTransformer3DModel:
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        model = SanaWmTransformer3DModel(config=_cam_config(), quant_config=None, prefix="transformer")
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def test_softmax_window_bf16_gpu_path() -> None:
    model = _model()
    lat = torch.randn(1, LATENT_CH, 2, 2, 2, device="cuda", dtype=torch.bfloat16)
    ehs = torch.randn(1, 3, SANA_WM_STAGE1_PROMPT_CHANNELS, device="cuda", dtype=torch.bfloat16)
    raymap = torch.randn(1, 2, 20, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        base = model(lat, 500.0, encoder_hidden_states=ehs, raymap=raymap)
        state = SoftmaxKvState(capture=True)
        out = model(lat, 500.0, encoder_hidden_states=ehs, raymap=raymap, softmax_state=state)

    assert torch.isfinite(base).all()
    torch.testing.assert_close(out, base)  # capture is non-invasive
    # Both the main and UCPE camera streams are recorded on the GPU path.
    assert (CAM_SOFTMAX_LAYER, False) in state.final
    assert (CAM_SOFTMAX_LAYER, True) in state.final

    # Seeding a cam-stream window through FlashAttention changes the output while
    # preserving the current-chunk token count.
    key, _ = state.final[(CAM_SOFTMAX_LAYER, True)]
    b, h, _, d = key.shape
    win = SoftmaxKvState(
        init={
            (CAM_SOFTMAX_LAYER, True): (
                torch.randn(b, h, 2 * 4, d, device="cuda", dtype=key.dtype),
                torch.randn(b, h, 2 * 4, d, device="cuda", dtype=key.dtype),
            )
        },
        capture=False,
    )
    with torch.no_grad():
        seeded = model(lat, 500.0, encoder_hidden_states=ehs, raymap=raymap, softmax_state=win)
    assert seeded.shape == base.shape
    assert not torch.allclose(seeded, base)
