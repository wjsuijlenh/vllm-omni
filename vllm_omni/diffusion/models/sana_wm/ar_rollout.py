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

from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import GdnArState

if TYPE_CHECKING:
    from collections.abc import Sequence

FIRST_CHUNK_PLUS_ONE = "first_chunk_plus_one"
UNIFORM = "uniform"


class _GdnStateStore(Protocol):
    """The subset of ``SanaWmArStateAdapter`` the rollout helpers depend on."""

    @property
    def gdn_layers(self) -> tuple[int, ...]: ...

    def get_gdn_state(self, layer_index: int, is_negative: bool = ...) -> dict[str, torch.Tensor]: ...

    def commit_gdn_state(
        self, layer_index: int, state_kv: torch.Tensor, state_z: torch.Tensor, is_negative: bool = ...
    ) -> None: ...


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
