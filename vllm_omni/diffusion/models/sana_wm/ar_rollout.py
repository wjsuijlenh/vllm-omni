# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autoregressive chunk-rollout orchestration for SANA-WM.

The chunk-causal SANA-WM checkpoint generates a clip as a sequence of small
chunks (``chunk_size`` latent frames each), carrying the Gated-DeltaNet recurrent
state forward across chunks instead of denoising the whole clip bidirectionally
at once. This module holds the *orchestration* that the pipeline's AR backend
drives -- pure, model-free helpers so the state seed/capture/commit sequencing
can be unit-tested on CPU without the transformer, VAE, or weights:

* :func:`plan_chunks` -- split ``num_latent_frames`` into ``ChunkSpan``s.
* :func:`step_state` -- build the per-denoise-step ``GdnArState`` (seed from the
  adapter's carried state; request capture only on the final step).
* :func:`commit_chunk_state` -- write a chunk's captured terminal state back to
  the adapter for the next chunk.
* :func:`slice_camera_frames` -- slice a per-latent-frame camera tensor to a span.

The heavy denoise loop (timesteps, scheduler, CFG, transformer call) stays in the
pipeline; these helpers decide *what state to thread*, and they are what the CPU
tests pin. End-to-end numerical equivalence is validated separately on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import GdnArState, SoftmaxKvState

if TYPE_CHECKING:
    from collections.abc import Sequence

FIRST_CHUNK_PLUS_ONE = "first_chunk_plus_one"
UNIFORM = "uniform"

# Default frames retained at the front of the softmax window (the clean anchor).
SOFTMAX_SINK_FRAMES = 1


class _GdnStateStore(Protocol):
    """The subset of ``SanaWmArStateAdapter`` the rollout helpers depend on."""

    @property
    def gdn_layers(self) -> tuple[int, ...]: ...

    def get_gdn_state(self, layer_index: int, is_negative: bool = ...) -> dict[str, torch.Tensor]: ...

    def commit_gdn_state(
        self, layer_index: int, state_kv: torch.Tensor, state_z: torch.Tensor, is_negative: bool = ...
    ) -> None: ...


class _SoftmaxKvStore(Protocol):
    """The subset of ``SanaWmArStateAdapter`` the softmax-window helpers depend on.

    ``get_softmax_kv`` / ``commit_softmax_kv`` round-trip a single stacked tensor
    ``(2, B, seq, H, D)`` (slot 0 = key, slot 1 = value) per softmax block and CFG
    branch -- the ``PagedKV`` layout already used by the adapter. An empty window
    is ``seq == 0``.
    """

    @property
    def softmax_layers(self) -> tuple[int, ...]: ...

    def get_softmax_kv(self, layer_index: int, is_negative: bool = ...) -> torch.Tensor: ...

    def commit_softmax_kv(self, layer_index: int, kv: torch.Tensor, is_negative: bool = ...) -> None: ...


@dataclass(frozen=True)
class ChunkSpan:
    """One autoregressive chunk: latent frames ``[start, end)``."""

    index: int
    start: int
    end: int

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def num_frames(self) -> int:
        return self.end - self.start


def plan_chunks(
    num_latent_frames: int,
    chunk_size: int,
    strategy: str = FIRST_CHUNK_PLUS_ONE,
) -> list[ChunkSpan]:
    """Partition ``[0, num_latent_frames)`` into contiguous chunk spans.

    ``first_chunk_plus_one`` (the chunk-causal release default) makes the first
    chunk ``chunk_size + 1`` frames -- the image-conditioning frame 0 plus a full
    ``chunk_size`` of generated frames -- and every later chunk ``chunk_size``
    frames. ``uniform`` makes every chunk ``chunk_size`` (the last may be
    shorter). Either way the spans are contiguous, in order, and cover exactly
    ``[0, num_latent_frames)``.

    NOTE: the exact ``first_chunk_plus_one`` semantics are taken from the release
    config name; confirm against the upstream sampler during GPU equivalence.
    """
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    if strategy == FIRST_CHUNK_PLUS_ONE:
        first_len = chunk_size + 1
    elif strategy == UNIFORM:
        first_len = chunk_size
    else:
        raise ValueError(f"Unknown chunk split strategy {strategy!r}.")

    spans: list[ChunkSpan] = []
    start = 0
    index = 0
    while start < num_latent_frames:
        length = first_len if index == 0 else chunk_size
        end = min(start + length, num_latent_frames)
        spans.append(ChunkSpan(index=index, start=start, end=end))
        start = end
        index += 1
    return spans


def autoregressive_segments(num_latent_frames: int, chunk_size: int) -> list[ChunkSpan]:
    """Chunk spans matching the upstream SANA-WM inference sampler.

    Mirrors ``SelfForcingFlowEuler.create_autoregressive_segments`` in
    ``NVlabs/Sana`` (``diffusion/scheduler/self_forcing_flow_euler_sampler.py``):
    the **first chunk absorbs the remainder** (``num_latent_frames % chunk_size``)
    and every later chunk is exactly ``chunk_size``. Spans are contiguous and
    non-overlapping -- each chunk denoises only its own frames; cross-chunk
    context is carried through cached state, never by re-feeding a frame. This is
    the segmentation the AR pipeline backend uses (preferred over the
    ``first_chunk_plus_one`` name strategy, which only matches when the remainder
    is 0 or 1).
    """
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if num_latent_frames <= chunk_size:
        # Upstream degenerates here (no chunks); cover everything in one chunk.
        return [ChunkSpan(index=0, start=0, end=num_latent_frames)]

    remainder = num_latent_frames % chunk_size
    num_chunks = num_latent_frames // chunk_size
    spans: list[ChunkSpan] = []
    start = 0
    for i in range(num_chunks):
        length = chunk_size + (remainder if i == 0 else 0)
        end = start + length
        spans.append(ChunkSpan(index=i, start=start, end=end))
        start = end
    return spans


def step_state(
    store: _GdnStateStore,
    *,
    is_first_chunk: bool,
    capture: bool,
    is_negative: bool = False,
) -> GdnArState:
    """Build the ``GdnArState`` for one denoise step of a chunk.

    For the first chunk the forward scan seeds from zeros (no ``init`` entries);
    otherwise it seeds from the terminal state the previous chunk committed to
    ``store``. ``capture`` should be ``True`` only on the chunk's final denoise
    step, so the recorded terminal state corresponds to the settled latent.
    """
    init: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if not is_first_chunk:
        for layer in store.gdn_layers:
            view = store.get_gdn_state(layer, is_negative)
            init[layer] = (view["state_kv"], view["state_z"])
    return GdnArState(init=init, capture=capture)


def commit_chunk_state(store: _GdnStateStore, state: GdnArState, *, is_negative: bool = False) -> None:
    """Write a chunk's captured terminal GDN state back to the store."""
    for layer, (state_kv, state_z) in state.final.items():
        store.commit_gdn_state(layer, state_kv, state_z, is_negative)


def softmax_step_state(
    store: _SoftmaxKvStore,
    *,
    is_first_chunk: bool,
    capture: bool,
    is_negative: bool = False,
) -> SoftmaxKvState:
    """Build the ``SoftmaxKvState`` carrier for one denoise step of a chunk.

    The first chunk has no cached context (empty ``init``). Later chunks seed each
    softmax block from the windowed ``(key, value)`` the store accumulated from
    earlier chunks, converted from the ``PagedKV`` layout ``(2, B, seq, H, D)`` to
    the per-branch ``(B, H, seq, D)`` the transformer prepends. ``capture`` should
    be ``True`` only on the chunk's clean-sigma pass, so the recorded K/V belong
    to the settled latent (mirroring the GDN carry).
    """
    init: dict[tuple[int, bool], tuple[torch.Tensor, torch.Tensor]] = {}
    if not is_first_chunk:
        for layer in store.softmax_layers:
            kv = store.get_softmax_kv(layer, is_negative)  # (2, B, seq, H, D)
            if kv.shape[2] > 0:
                key = kv[0].permute(0, 2, 1, 3).contiguous()  # (B, H, seq, D)
                value = kv[1].permute(0, 2, 1, 3).contiguous()
                init[(layer, False)] = (key, value)
    return SoftmaxKvState(init=init, capture=capture)


def _evict_window(kv: torch.Tensor, spatial_tokens: int, window_frames: int, sink_frames: int) -> torch.Tensor:
    """Trim a stacked ``(2, B, seq, H, D)`` window to ``sink + sliding window``.

    Eviction is at frame granularity (``seq`` is a multiple of ``spatial_tokens``):
    the first ``sink_frames`` frames are always kept (the anchor), plus the most
    recent ``window_frames`` frames. This realises the upstream "sink + sliding
    window" policy; it evicts by frame rather than by whole chunk, which differs
    only at the (larger) first chunk's boundary.
    """
    if window_frames <= 0:
        return kv  # unbounded: keep the full history (correct but not memory-bounded)
    seq = kv.shape[2]
    total_frames = seq // spatial_tokens
    if total_frames <= sink_frames + window_frames:
        return kv
    sink = kv[:, :, : sink_frames * spatial_tokens]
    window = kv[:, :, seq - window_frames * spatial_tokens :]
    return torch.cat([sink, window], dim=2)


def commit_softmax_window(
    store: _SoftmaxKvStore,
    state: SoftmaxKvState,
    *,
    spatial_tokens: int,
    window_frames: int,
    sink_frames: int = SOFTMAX_SINK_FRAMES,
    is_negative: bool = False,
) -> None:
    """Append a chunk's captured softmax K/V to the window and slide it forward.

    For each cached softmax block the current chunk's ``(key, value)``
    (``(B, H, N, D)``) is converted to the ``PagedKV`` layout, concatenated onto
    the existing window along the token axis, trimmed by :func:`_evict_window`,
    and written back. The ``is_cam`` camera branch is not windowed yet and is
    skipped here.
    """
    for (layer, is_cam), (key, value) in state.final.items():
        if is_cam:
            continue
        key_store = key.permute(0, 2, 1, 3)  # (B, N, H, D)
        value_store = value.permute(0, 2, 1, 3)
        new = torch.stack([key_store, value_store], dim=0)  # (2, B, N, H, D)
        prev = store.get_softmax_kv(layer, is_negative)  # (2, B, seq, H, D); seq == 0 when empty
        combined = torch.cat([prev, new], dim=2) if prev.shape[2] > 0 else new
        kept = _evict_window(combined, spatial_tokens, window_frames, sink_frames)
        store.commit_softmax_kv(layer, kept, is_negative)


def slice_camera_frames(tensor: torch.Tensor | None, span: ChunkSpan, *, frame_dim: int) -> torch.Tensor | None:
    """Slice a per-latent-frame camera tensor to ``span`` along ``frame_dim``."""
    if tensor is None:
        return None
    if tensor.shape[frame_dim] < span.end:
        raise ValueError(
            f"Camera tensor has {tensor.shape[frame_dim]} frames along dim {frame_dim}, "
            f"too few for chunk span [{span.start}, {span.end})."
        )
    index = torch.arange(span.start, span.end, device=tensor.device)
    return tensor.index_select(frame_dim, index)


def gdn_layer_set(layers: Sequence[int]) -> frozenset[int]:
    """Convenience: the GDN layer indices as a set (softmax blocks excluded)."""
    return frozenset(layers)
