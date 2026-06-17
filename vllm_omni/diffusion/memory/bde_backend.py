# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A ``PagedKV`` whose storage is the BDE engine's paged KV pool (RFC #4480 over #4366).

This is the prototype that shows the ``MemoryObject`` contract from #4480 wraps the
Block Diffusion Engine's real allocator (``BDEKVCache``) from the ``bde-kv-phase1``
branch of #4366, rather than the Phase-0 plain buffer in :mod:`objects`.

It is deliberately a separate, optional module: the core contract (:mod:`base`,
:mod:`objects`) must not depend on the ``vllm_omni.bde`` engine. Only this bridge
imports it, so the dependency points the right way (the engine-specific backing
depends on the contract, never the reverse).

Mapping (one ``PagedKV`` = one self-attention layer of one CFG branch):

    ``allocate`` -> ``BDEKVCache.begin_request``
    ``commit``   -> ``allocate_chunk`` + ``write_chunk_kv`` + ``commit_chunk``
    ``view``     -> ``gather_window``
    ``reset``    -> ``end_request`` (return blocks to the pool)

Calling convention matches the Phase-0 ``PagedKV``: the model hands in the *full*
cumulative KV ``(2, 1, seq, H, D)`` each forward; this object stores only the new
tokens since the last commit (one or more chunks) into the pool. ``view()`` gathers
the resident window. While the sequence fits the window -- which DreamZero
guarantees by resetting at the attention-window boundary -- ``view()`` is identical
to the plain ``PagedKV`` view. When the window is exceeded (VGGT-style sliding),
``view()`` returns the same trailing window the plain buffer would, sliced.

Prototype simplification: this owns a single-layer ``BDEKVCache`` per (layer, branch)
object. The production form shares one multi-layer pool across all layers and both
CFG branches (that is what BDE's own ``BDEKVState`` does); the per-chunk calls are
the same, only the chunk allocation is hoisted to once per forward instead of once
per object. Keeping one pool per object here makes the equivalence test isolated.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.bde.kv_cache import BDEKVCache, BDEKVConfig, ChunkWindowSpec
from vllm_omni.diffusion.memory.base import MemoryObject

_REQUEST_ID = "pagedkv"


class PagedKVBDE(MemoryObject):
    """Self-attention KV for one layer and one CFG branch, backed by BDE's pool."""

    def __init__(self) -> None:
        self._cache: BDEKVCache | None = None
        self._adapter: Any = None
        self._committed: int = 0
        # (dtype, device, num_heads, head_dim) for building the empty view.
        self._spec: tuple[torch.dtype, torch.device | str, int, int] | None = None
        self._staged: Any = None

    def allocate(  # type: ignore[override]  # explicit spec; base is **spec
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        num_heads: int,
        head_dim: int,
        chunk_size: int,
        window_chunks: int,
        sink_chunks: int = 0,
        reset_at_boundary: bool = False,
        block_size: int | None = None,
        pool_blocks: int = 1024,
        max_model_len: int | None = None,
        **_: Any,
    ) -> None:
        if batch_size != 1:
            raise ValueError("PagedKVBDE backs one CFG branch (batch=1); branches are separate objects.")
        block_size = block_size or chunk_size
        if chunk_size % block_size != 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of block_size ({block_size}).")

        config = BDEKVConfig(
            enable=True,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            sink_chunks=sink_chunks,
            reset_at_boundary=reset_at_boundary,
            gpu_memory_fraction=1.0,
        )
        # Size the pool to ``pool_blocks`` blocks. Build a throwaway spec to read
        # the page size, then back out the byte budget (num_layers=1, fraction=1).
        probe = ChunkWindowSpec(
            block_size=block_size,
            num_kv_heads=num_heads,
            head_size=head_dim,
            dtype=dtype,
            sliding_window=window_chunks * chunk_size,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            sink_chunks=sink_chunks,
            reset_at_boundary=reset_at_boundary,
        )
        available_bytes = pool_blocks * probe.page_size_bytes
        self._cache = BDEKVCache(
            config,
            num_layers=1,
            num_kv_heads=num_heads,
            head_size=head_dim,
            dtype=dtype,
            block_size=block_size,
            max_model_len=max_model_len or pool_blocks * block_size,
            available_bytes=available_bytes,
            device=torch.device(device) if isinstance(device, str) else device,
        )
        self._adapter = self._cache.begin_request(_REQUEST_ID)
        self._committed = 0
        self._spec = (dtype, device, num_heads, head_dim)
        self._staged = None

    def commit(self, payload: torch.Tensor | None = None) -> None:
        if payload is None:
            raise ValueError("PagedKVBDE.commit requires a tensor payload.")
        if self._cache is None or self._adapter is None:
            raise RuntimeError("PagedKVBDE is not allocated; call allocate() first.")
        chunk_size = self._cache.spec.chunk_size
        seq = int(payload.shape[2])
        new_tokens = seq - self._committed
        if new_tokens < 0:
            raise ValueError("Cumulative KV shrank; call reset() to start a new session.")
        if new_tokens == 0:
            return
        if new_tokens % chunk_size != 0:
            raise ValueError(
                f"New tokens ({new_tokens}) since last commit must be a multiple of "
                f"chunk_size ({chunk_size}); got cumulative seq {seq}."
            )
        k_full, v_full = payload[0], payload[1]  # each (1, seq, H, D)
        for c in range(new_tokens // chunk_size):
            lo = self._committed + c * chunk_size
            hi = lo + chunk_size
            self._cache.allocate_chunk(self._adapter)
            self._cache.write_chunk_kv(0, k_full[:, lo:hi], v_full[:, lo:hi], self._adapter)
            self._cache.commit_chunk(self._adapter)
        self._committed = seq

    def view(self, *, include_staged: bool = True) -> torch.Tensor:
        if self._cache is None or self._adapter is None:
            raise RuntimeError("PagedKVBDE is not allocated; call allocate() first.")
        if self._committed == 0:
            dtype, device, num_heads, head_dim = self._spec  # type: ignore[misc]
            return torch.zeros(2, 1, 0, num_heads, head_dim, dtype=dtype, device=device)
        return self._cache.gather_window(0, self._adapter)

    def reset(self) -> None:
        if self._cache is not None and self._adapter is not None:
            self._cache.end_request(self._adapter)  # return blocks to the pool
        self._cache = None
        self._adapter = None
        self._committed = 0
        self._spec = None
        self._staged = None

    @property
    def nbytes(self) -> int:
        if self._cache is None or self._adapter is None or self._committed == 0:
            return 0
        resident_blocks = len(self._cache.window_block_ids(self._adapter))
        return resident_blocks * self._cache.spec.page_size_bytes

    @property
    def resident(self) -> bool:
        return self._cache is not None
