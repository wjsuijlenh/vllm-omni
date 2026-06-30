# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU equivalence tests for the SANA-WM GDN autoregressive state carry.

Step 3 of the AR SANA-WM port exposes the state-cached scan: the GDN forward
recurrence can be seeded from a prior chunk's terminal state and return its own
terminal state. The load-bearing invariant is that the forward recurrence is
*exactly associative* across a chunk boundary -- scanning ``[a; b]`` in one pass
equals scanning ``a``, then seeding ``b`` with ``a``'s final state. These tests
pin that on CPU with the PyTorch reference path (no Triton, no GPU), so the
autoregressive carry is validated before the GPU equivalence step.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.gdn import (
    SANA_WM_DISABLE_TRITON_GDN_ENV,
    _delta_scan,
    reference_bidirectional_gated_delta_net,
    triton_bidirectional_gated_delta_net_from_qkv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

B, H, D, S = 1, 2, 4, 3  # batch, heads, head_dim, spatial tokens per frame
FRAMES = 4
DTYPE = torch.float64  # tight equivalence; the recurrence is dtype-agnostic


def _inputs(frames: int = FRAMES) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    n = frames * S

    def qkv_like() -> torch.Tensor:
        return torch.randn(B, H, D, n, generator=g, dtype=DTYPE)

    return {
        "query": qkv_like(),
        "key": qkv_like(),
        "value": qkv_like(),
        "query_rot": qkv_like(),
        "key_rot": qkv_like(),
        # decay in (0, 1) for a stable recurrence; beta is an arbitrary gate.
        "beta": torch.rand(B, H, frames, S, generator=g, dtype=DTYPE),
        "decay": torch.rand(B, H, frames, generator=g, dtype=DTYPE),
    }


def _scan(t: dict[str, torch.Tensor], **kw: object) -> tuple[torch.Tensor, ...]:
    return _delta_scan(
        t["query"], t["key"], t["value"], t["query_rot"], t["key_rot"], t["beta"], t["decay"], spatial_tokens=S, **kw
    )


def _slice_frames(t: dict[str, torch.Tensor], f0: int, f1: int) -> dict[str, torch.Tensor]:
    out = dict(t)
    for name in ("query", "key", "value", "query_rot", "key_rot"):
        out[name] = t[name][..., f0 * S : f1 * S].contiguous()
    out["beta"] = t["beta"][:, :, f0:f1, :].contiguous()
    out["decay"] = t["decay"][:, :, f0:f1].contiguous()
    return out


def test_delta_scan_carry_is_exactly_associative() -> None:
    full = _inputs()
    num_full, den_full, kv_full, z_full = _scan(full, return_final_state=True)

    split = 2  # process frames [0,2) then [2,4) with carry
    a = _slice_frames(full, 0, split)
    b = _slice_frames(full, split, FRAMES)
    num_a, den_a, kv_a, z_a = _scan(a, return_final_state=True)
    num_b, den_b, kv_b, z_b = _scan(b, init_state_kv=kv_a, init_state_z=z_a, return_final_state=True)

    torch.testing.assert_close(torch.cat([num_a, num_b], dim=-1), num_full)
    torch.testing.assert_close(torch.cat([den_a, den_b], dim=-1), den_full)
    torch.testing.assert_close(kv_b, kv_full)
    torch.testing.assert_close(z_b, z_full)


def test_delta_scan_three_way_split_carry() -> None:
    full = _inputs()
    num_full, den_full, kv_full, z_full = _scan(full, return_final_state=True)

    kv = z = None
    nums, dens = [], []
    for f0, f1 in ((0, 1), (1, 3), (3, 4)):  # uneven chunks
        seg = _slice_frames(full, f0, f1)
        num, den, kv, z = _scan(seg, init_state_kv=kv, init_state_z=z, return_final_state=True)
        nums.append(num)
        dens.append(den)
    torch.testing.assert_close(torch.cat(nums, dim=-1), num_full)
    torch.testing.assert_close(torch.cat(dens, dim=-1), den_full)
    torch.testing.assert_close(kv, kv_full)
    torch.testing.assert_close(z, z_full)


def test_delta_scan_init_shape_validation() -> None:
    seg = _slice_frames(_inputs(), 0, 2)
    with pytest.raises(ValueError):
        _scan(seg, init_state_kv=torch.zeros(B, H, D, D + 1), init_state_z=torch.zeros(B, H, D, 1))
    with pytest.raises(ValueError):
        _scan(seg, init_state_kv=torch.zeros(B, H, D, D), init_state_z=torch.zeros(B, H, D, 2))


def test_reference_no_init_is_backward_compatible() -> None:
    t = _inputs()
    base = reference_bidirectional_gated_delta_net(
        t["query"], t["key"], t["value"], beta=t["beta"], decay=t["decay"], spatial_tokens=S,
        query_rot=t["query_rot"], key_rot=t["key_rot"],
    )
    out, kv, z = reference_bidirectional_gated_delta_net(
        t["query"], t["key"], t["value"], beta=t["beta"], decay=t["decay"], spatial_tokens=S,
        query_rot=t["query_rot"], key_rot=t["key_rot"], return_final_state=True,
    )
    # Returning the final state must not change the primary output.
    torch.testing.assert_close(out, base)
    assert kv.shape == (B, H, D, D)
    assert z.shape == (B, H, D, 1)


def test_reference_forward_state_matches_delta_scan() -> None:
    # The reference's returned (carried-forward) state is exactly the forward
    # delta-scan's terminal state -- the reverse scan never touches the carry.
    t = _inputs()
    _, kv_ref, z_ref = reference_bidirectional_gated_delta_net(
        t["query"], t["key"], t["value"], beta=t["beta"], decay=t["decay"], spatial_tokens=S,
        query_rot=t["query_rot"], key_rot=t["key_rot"], return_final_state=True,
    )
    # The reference upcasts to float32 internally; mirror that so the direct
    # forward scan is compared at the same precision.
    t32 = {name: tensor.float() for name, tensor in t.items()}
    _, _, kv_fwd, z_fwd = _scan(t32, return_final_state=True)
    torch.testing.assert_close(kv_ref, kv_fwd)
    torch.testing.assert_close(z_ref, z_fwd)


def test_triton_entry_point_accepts_state_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    # On CPU the fused path can't run, but the widened signature must accept the
    # state kwargs and fail on the CUDA guard (RuntimeError), not on a bad
    # signature (TypeError).
    monkeypatch.delenv(SANA_WM_DISABLE_TRITON_GDN_ENV, raising=False)
    qkv = torch.randn(B, FRAMES * S, 3, H, D)
    norm = torch.nn.RMSNorm(H * D)
    with pytest.raises(RuntimeError):
        triton_bidirectional_gated_delta_net_from_qkv(
            qkv, beta=torch.rand(B, H, FRAMES, S), decay=torch.rand(B, H, FRAMES),
            q_norm=norm, k_norm=norm, spatial_tokens=S, k_scale=1.0,
            init_state_kv=torch.zeros(B, H, D, D), init_state_z=torch.zeros(B, H, D, 1),
            return_final_state=True,
        )
