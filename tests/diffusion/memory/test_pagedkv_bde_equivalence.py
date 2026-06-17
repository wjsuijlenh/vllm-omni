# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence: PagedKV plain buffer (#4480 Phase 0) vs BDE-backed PagedKV (#4366).

Drives both objects with the identical sequence of ``commit(full_cumulative_kv)``
calls the DreamZero pipeline makes, and asserts the BDE-backed ``view()`` matches
the plain buffer's view. This is the prototype proof that the #4480 ``MemoryObject``
contract wraps BDE's real paged allocator (``BDEKVCache``).

Two regimes:
  * Window large enough to hold the sequence (DreamZero bounds growth via reset):
    BDE view is bit-identical to the plain full buffer.
  * Window exceeded (VGGT-style sliding eviction): BDE view equals the trailing
    window of the plain buffer.

Parametrized over device: CPU always; CUDA when present (the same test is the GPU
evaluation -- it exercises the real on-device pool allocation and gather).
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.memory.bde_backend import PagedKVBDE
from vllm_omni.diffusion.memory.objects import PagedKV

pytestmark = [pytest.mark.core_model]

HEADS, HEAD_DIM, CHUNK = 2, 4, 3  # tokens per chunk; block_size defaults to CHUNK


def _devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def _grow(prev: torch.Tensor | None, n_new: int, device: torch.device) -> torch.Tensor:
    """Append ``n_new`` random tokens to a cumulative ``(2, 1, seq, H, D)`` KV."""
    block = torch.randn(2, 1, n_new, HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    return block if prev is None else torch.cat([prev, block], dim=2)


@pytest.mark.cpu
@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_view_matches_plain_no_eviction(device: torch.device) -> None:
    # Window far larger than the sequence -> no eviction -> bit-identical views.
    plain, bde = PagedKV(), PagedKVBDE()
    common = dict(batch_size=1, dtype=torch.float32, device=device, num_heads=HEADS, head_dim=HEAD_DIM)
    plain.allocate(**common)
    bde.allocate(**common, chunk_size=CHUNK, window_chunks=100)

    cumulative = None
    for _ in range(5):  # 5 forwards, one chunk each -> 15 tokens, window holds 300
        cumulative = _grow(cumulative, CHUNK, device)
        plain.commit(cumulative)
        bde.commit(cumulative)
        torch.testing.assert_close(bde.view(), plain.view())

    assert bde.view().shape == (2, 1, 5 * CHUNK, HEADS, HEAD_DIM)
    assert bde.resident and plain.resident


@pytest.mark.cpu
@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_view_matches_plain_tail_under_eviction(device: torch.device) -> None:
    # Small sliding window -> BDE evicts out-of-window chunks; its view must equal
    # the trailing window of the plain full buffer.
    window_chunks = 3
    window_tokens = window_chunks * CHUNK
    plain, bde = PagedKV(), PagedKVBDE()
    common = dict(batch_size=1, dtype=torch.float32, device=device, num_heads=HEADS, head_dim=HEAD_DIM)
    plain.allocate(**common)
    bde.allocate(**common, chunk_size=CHUNK, window_chunks=window_chunks, reset_at_boundary=False)

    cumulative = None
    for step in range(6):  # grow to 18 tokens, window holds 9
        cumulative = _grow(cumulative, CHUNK, device)
        plain.commit(cumulative)
        bde.commit(cumulative)
        w = bde.view()
        n = w.shape[2]
        assert n <= window_tokens
        # The resident window is the trailing n tokens of the full sequence.
        torch.testing.assert_close(w, plain.view()[:, :, -n:])

    # After enough growth the window is full and capped.
    assert bde.view().shape[2] == window_tokens


@pytest.mark.cpu
@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_reset_returns_to_empty(device: torch.device) -> None:
    bde = PagedKVBDE()
    common = dict(batch_size=1, dtype=torch.float32, device=device, num_heads=HEADS, head_dim=HEAD_DIM)
    bde.allocate(**common, chunk_size=CHUNK, window_chunks=8)
    cumulative = _grow(None, 2 * CHUNK, device)
    bde.commit(cumulative)
    assert bde.nbytes > 0 and bde.view().shape[2] == 2 * CHUNK

    bde.reset()
    assert not bde.resident
    assert bde.nbytes == 0

    # Re-allocate and confirm it starts empty again.
    bde.allocate(**common, chunk_size=CHUNK, window_chunks=8)
    assert bde.view().shape[2] == 0


@pytest.mark.cpu
@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_cfg_branches_are_independent_objects(device: torch.device) -> None:
    # Positive and negative branches are separate PagedKVBDE objects with separate
    # pools, so they never share storage -- mirroring the plain-buffer contract.
    common = dict(batch_size=1, dtype=torch.float32, device=device, num_heads=HEADS, head_dim=HEAD_DIM)
    pos, neg = PagedKVBDE(), PagedKVBDE()
    pos.allocate(**common, chunk_size=CHUNK, window_chunks=8)
    neg.allocate(**common, chunk_size=CHUNK, window_chunks=8)
    kv_pos = _grow(None, CHUNK, device)
    kv_neg = _grow(None, CHUNK, device)
    pos.commit(kv_pos)
    neg.commit(kv_neg)
    torch.testing.assert_close(pos.view(), kv_pos)
    torch.testing.assert_close(neg.view(), kv_neg)
    assert not torch.allclose(pos.view(), neg.view())


@pytest.mark.cpu
@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_multilayer_each_layer_matches_plain(device: torch.device) -> None:
    # The contract is per-(layer, branch). Drive N independent layers and confirm
    # each BDE-backed layer tracks its own plain counterpart.
    num_layers = 3
    common = dict(batch_size=1, dtype=torch.float32, device=device, num_heads=HEADS, head_dim=HEAD_DIM)
    plains = [PagedKV() for _ in range(num_layers)]
    bdes = [PagedKVBDE() for _ in range(num_layers)]
    for p in plains:
        p.allocate(**common)
    for b in bdes:
        b.allocate(**common, chunk_size=CHUNK, window_chunks=50)

    cumulative = [None] * num_layers
    for _ in range(4):
        for i in range(num_layers):
            cumulative[i] = _grow(cumulative[i], CHUNK, device)
            plains[i].commit(cumulative[i])
            bdes[i].commit(cumulative[i])
    for i in range(num_layers):
        torch.testing.assert_close(bdes[i].view(), plains[i].view())
