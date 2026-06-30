# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU regression test for the fused SANA-WM GDN autoregressive state carry.

The fused Triton ``fused_bigdn_bidi_chunkwise`` returns its terminal forward
state in canonical ``(B, H, D, D)`` form, but its phase-B seed path expects the
padded ``(BH, BLOCK_D, BLOCK_D)`` layout. ``_seed_init_state`` bridges the two so
a saved state round-trips. This test pins the invariant the bridge enables: the
fused forward-state carry is exactly associative across a chunk split (the GPU
counterpart of ``test_gdn_ar_scan.test_delta_scan_carry_is_exactly_associative``).

Skipped without CUDA (the Triton kernel requires a GPU).
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.gdn import triton_bidirectional_gated_delta_net_from_qkv

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="fused GDN kernel requires CUDA"),
]

B, H, D = 1, 2, 112
FRAMES, S = 4, 16
SPLIT = 2


def _triton(qkv, beta, decay, norm, *, f0, f1, init=None, final):
    init_kv, init_z = (None, None) if init is None else init
    return triton_bidirectional_gated_delta_net_from_qkv(
        qkv[:, f0 * S : f1 * S].contiguous(),
        beta=beta[:, :, f0:f1].contiguous(),
        decay=decay[:, :, f0:f1].contiguous(),
        q_norm=norm,
        k_norm=norm,
        spatial_tokens=S,
        rotary_emb=None,
        k_scale=(D**-0.5) * (S**-0.5),
        init_state_kv=init_kv,
        init_state_z=init_z,
        return_final_state=final,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_carry_is_associative(dtype: torch.dtype) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    n = FRAMES * S
    qkv = torch.randn(B, n, 3, H, D, generator=g, dtype=dtype, device="cuda")
    beta = torch.rand(B, H, FRAMES, S, generator=g, dtype=dtype, device="cuda")
    decay = torch.rand(B, H, FRAMES, generator=g, dtype=dtype, device="cuda")
    norm = torch.nn.RMSNorm(H * D, eps=1e-5).cuda()

    _, kv_full, z_full = _triton(qkv, beta, decay, norm, f0=0, f1=FRAMES, final=True)
    _, kv_a, z_a = _triton(qkv, beta, decay, norm, f0=0, f1=SPLIT, final=True)
    _, kv_b, z_b = _triton(qkv, beta, decay, norm, f0=SPLIT, f1=FRAMES, init=(kv_a, z_a), final=True)

    # Carrying the saved state across the split reproduces the single-pass state.
    torch.testing.assert_close(kv_b, kv_full, rtol=0, atol=1e-4)
    torch.testing.assert_close(z_b, z_full, rtol=0, atol=1e-4)
