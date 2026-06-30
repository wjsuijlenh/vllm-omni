# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-memory adapter for autoregressive (chunk-causal) SANA-WM.

``SanaWmArStateAdapter`` stores the per-session state of the AR SANA-WM world
model on the shared ``SessionMemoryManager`` (RFC #4480), the same way
``DreamZeroStateAdapter`` does for DreamZero. Unlike DreamZero, SANA-WM's
per-stream state is **constant-memory**: the 15 Gated-DeltaNet (GDN) blocks each
carry a fixed-size recurrent state, and the 5 softmax blocks carry a bounded
sink + local-window KV. Nothing grows without bound, so this maps onto the
dense Phase-0 path with no paged-KV allocator (#4366/#4534).

Per-session objects:

    * GDN recurrent state, per GDN block          -> ``FixedState``
      (``state_kv`` of shape ``(B, H, D, D)`` and ``state_z`` of ``(B, H, D, 1)``)
    * softmax sink+window self-attn KV, per
      softmax block and CFG branch               -> ``PagedKV`` (bounded, dense)
    * text cross-attention KV, per block & branch -> ``EncodeOnceKV``
    * warm-up / seed latent frames                -> ``LatentBuffer``

The camera branch is recomputed per chunk (``cam_scan_bidi_chunkwise`` carries no
state), so it is an input, not session state.

Block typing mirrors ``SanaWmTransformerBlock``: block ``i`` is softmax when
``softmax_every_n > 0 and (i + 1) % softmax_every_n == 0``, GDN otherwise.

The adapter holds no heavy state itself; scalar metadata lives in the session's
``attrs`` so a freshly constructed adapter for an existing session sees the same
data (the manager is the single source of truth and LRU authority). It takes
plain dimensions rather than a model config object so it stays importable and
CPU-testable without pulling in the model/vLLM stack; ``from_config`` is the
integration entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm_omni.experimental.world_models.memory.manager import SessionMemoryManager
from vllm_omni.experimental.world_models.memory.objects import (
    EncodeOnceKV,
    FixedState,
    LatentBuffer,
    PagedKV,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig

_GDN_KEY = "gdn"
_SOFTMAX_KV_KEY = "softmax_kv"
_TEXT_KEY = "text"
_WARMUP_KEY = "warmup_latents"
_NUM_GDN_LAYERS = "_num_gdn_layers"
_STATE_KV = "state_kv"
_STATE_Z = "state_z"


def _gdn_key(layer_index: int) -> str:
    return f"{_GDN_KEY}/{layer_index}"


def _softmax_kv_key(layer_index: int, is_negative: bool) -> str:
    branch = "neg" if is_negative else "pos"
    return f"{_SOFTMAX_KV_KEY}/{branch}/{layer_index}"


def _text_key(layer_index: int, is_negative: bool) -> str:
    branch = "neg" if is_negative else "pos"
    return f"{_TEXT_KEY}/{branch}/{layer_index}"


class SanaWmArStateAdapter:
    """Per-session state for AR SANA-WM, backed by the session memory manager."""

    def __init__(
        self,
        session_id: str | None,
        manager: SessionMemoryManager,
        *,
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        softmax_every_n: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self._session_id = session_id
        # Pin the session for this adapter's lifetime; see DreamZeroStateAdapter.
        self._session = manager.get_or_create_session(session_id)
        self._num_blocks = num_blocks
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._softmax_every_n = softmax_every_n
        self._gdn_layers = tuple(i for i in range(num_blocks) if self._is_gdn(i))
        self._softmax_layers = tuple(i for i in range(num_blocks) if not self._is_gdn(i))

    @classmethod
    def from_config(
        cls,
        session_id: str | None,
        manager: SessionMemoryManager,
        config: SanaWmConfig,
    ) -> SanaWmArStateAdapter:
        """Build from a parsed ``SanaWmConfig`` (the integration entry point)."""
        num_heads = max(config.hidden_size // max(config.linear_head_dim, 1), 1)
        head_dim = config.hidden_size // num_heads
        return cls(
            session_id,
            manager,
            num_blocks=config.num_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            softmax_every_n=config.softmax_every_n,
        )

    def _is_gdn(self, block_idx: int) -> bool:
        """Mirror ``SanaWmTransformerBlock``: every ``softmax_every_n``-th is softmax."""
        return self._softmax_every_n <= 0 or (block_idx + 1) % self._softmax_every_n != 0

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return self._gdn_layers

    @property
    def softmax_layers(self) -> tuple[int, ...]:
        return self._softmax_layers

    # -- metadata -------------------------------------------------------

    @property
    def chunk_index(self) -> int:
        """0 for the reset chunk (GDN scan seeds from zeros), >0 thereafter."""
        return int(self._session.attrs.get("chunk_index", 0))

    @chunk_index.setter
    def chunk_index(self, value: int) -> None:
        self._session.attrs["chunk_index"] = int(value)

    @property
    def language(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("language"))

    @language.setter
    def language(self, value: torch.Tensor | None) -> None:
        self._session.attrs["language"] = value

    # -- allocation -----------------------------------------------------

    def create_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Allocate all per-session objects in their empty/zero state.

        Called on the session's reset chunk. GDN states are zeroed (the chunk-0
        scan seeds from zeros); softmax/text KV start empty and grow once,
        bounded; the warm-up buffer starts empty.
        """
        session = self._session
        h, d = self._num_heads, self._head_dim
        for layer in self._gdn_layers:
            state = FixedState()
            state.allocate(
                shapes={_STATE_KV: (batch_size, h, d, d), _STATE_Z: (batch_size, h, d, 1)},
                dtype=dtype,
                device=device,
            )
            session.put(_gdn_key(layer), state)

        for layer in self._softmax_layers:
            for is_neg in (False, True):
                kv = PagedKV()
                kv.allocate(batch_size=batch_size, dtype=dtype, device=device, num_heads=h, head_dim=d)
                session.put(_softmax_kv_key(layer, is_neg), kv)

        for layer in range(self._num_blocks):
            for is_neg in (False, True):
                text = EncodeOnceKV()
                text.allocate()
                session.put(_text_key(layer, is_neg), text)

        warmup = LatentBuffer()
        warmup.allocate(maxlen=1)  # previous chunk's last frame, used to seed frame 0
        session.put(_WARMUP_KEY, warmup)

        session.attrs[_NUM_GDN_LAYERS] = len(self._gdn_layers)
        self.chunk_index = 0

    # -- GDN recurrent state --------------------------------------------

    def get_gdn_state(self, layer_index: int) -> dict[str, torch.Tensor]:
        """Return the live ``{state_kv, state_z}`` for a GDN block.

        On the reset chunk (``chunk_index == 0``) these are zeros and the caller
        should pass ``init_state=None`` to the scan; from chunk 1 on they hold the
        terminal forward state saved by the previous chunk.
        """
        return self._gdn_object(layer_index).view()

    def commit_gdn_state(self, layer_index: int, state_kv: torch.Tensor, state_z: torch.Tensor) -> None:
        """Overwrite a GDN block's recurrent state in place (end of a chunk)."""
        self._gdn_object(layer_index).commit({_STATE_KV: state_kv, _STATE_Z: state_z})

    def _gdn_object(self, layer_index: int) -> FixedState:
        obj = self._session.get(_gdn_key(layer_index))
        if not isinstance(obj, FixedState) or not obj.resident:
            raise RuntimeError(
                f"GDN state for layer {layer_index} not initialized; call create_state first "
                f"(GDN layers are {self._gdn_layers})."
            )
        return obj

    # -- softmax sink/window KV -----------------------------------------

    def get_softmax_kv(self, layer_index: int, is_negative: bool = False) -> torch.Tensor:
        return self._softmax_object(layer_index, is_negative).view()

    def commit_softmax_kv(self, layer_index: int, kv: torch.Tensor, is_negative: bool = False) -> None:
        self._softmax_object(layer_index, is_negative).commit(kv)

    def _softmax_object(self, layer_index: int, is_negative: bool) -> PagedKV:
        obj = self._session.get(_softmax_kv_key(layer_index, is_negative))
        if not isinstance(obj, PagedKV) or not obj.resident:
            raise RuntimeError(
                f"Softmax KV for layer {layer_index} not initialized; call create_state first "
                f"(softmax layers are {self._softmax_layers})."
            )
        return obj

    # -- text cross-attention KV ----------------------------------------

    def get_text_cache(self, layer_index: int, is_negative: bool = False) -> dict[str, bool | torch.Tensor | None]:
        obj = self._session.get(_text_key(layer_index, is_negative))
        if not isinstance(obj, EncodeOnceKV) or not obj.resident:
            raise RuntimeError(f"Text cache for layer {layer_index} not initialized; call create_state first.")
        return obj.view()

    # -- warm-up latent buffer ------------------------------------------

    @property
    def warmup_buffer(self) -> LatentBuffer:
        obj = self._session.get(_WARMUP_KEY)
        if not isinstance(obj, LatentBuffer) or not obj.resident:
            raise RuntimeError("Warm-up buffer not initialized; call create_state first.")
        return obj

    def seed_frame(self, frame: Any) -> None:
        """Record the previous chunk's last frame to seed the next chunk's frame 0."""
        self.warmup_buffer.append(frame)

    def last_seed_frame(self) -> Any | None:
        frames = self.warmup_buffer.view()
        return frames[-1] if frames else None

    # -- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        """Drop all state and metadata; the next ``create_state`` re-allocates."""
        self._session.reset()
