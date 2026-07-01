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
_CONV_KEY = "conv"
_NUM_GDN_LAYERS = "_num_gdn_layers"
_CONV_SLOTS_ATTR = "_conv_slots"
_STATE_KV = "state_kv"
_STATE_Z = "state_z"
_CONV_FRAMES = "frames"


def _gdn_key(layer_index: int, is_negative: bool = False) -> str:
    # GDN recurrent state is per CFG branch: the cond and uncond passes are
    # independent sequences, each with its own forward recurrence across chunks.
    branch = "neg" if is_negative else "pos"
    return f"{_GDN_KEY}/{branch}/{layer_index}"


def _softmax_kv_key(layer_index: int, is_negative: bool, is_cam: bool = False) -> str:
    # Two independent windows per softmax block: the main self-attention K/V and
    # the UCPE camera branch's K/V (separate head dims, separate sliding windows).
    branch = "neg" if is_negative else "pos"
    stream = "cam" if is_cam else "main"
    return f"{_SOFTMAX_KV_KEY}/{stream}/{branch}/{layer_index}"


def _text_key(layer_index: int, is_negative: bool) -> str:
    branch = "neg" if is_negative else "pos"
    return f"{_TEXT_KEY}/{branch}/{layer_index}"


def _conv_key(layer_index: int, slot: str, is_negative: bool = False) -> str:
    # Temporal-conv boundary state is per CFG branch and per conv slot ("k",
    # "k_cam", "ffn_t"): the previous chunk's trailing conv-input frames that seed
    # the next chunk's leading pad.
    branch = "neg" if is_negative else "pos"
    return f"{_CONV_KEY}/{branch}/{slot}/{layer_index}"


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
            for is_neg in (False, True):
                state = FixedState()
                state.allocate(
                    shapes={_STATE_KV: (batch_size, h, d, d), _STATE_Z: (batch_size, h, d, 1)},
                    dtype=dtype,
                    device=device,
                )
                session.put(_gdn_key(layer, is_neg), state)

        for layer in self._softmax_layers:
            for is_neg in (False, True):
                for is_cam in (False, True):
                    kv = PagedKV()
                    # The empty (seq==0) buffer is only an "is-empty" sentinel; the
                    # real head count/dim come from the first committed chunk, so
                    # the cam branch's differing dims need no special allocation.
                    kv.allocate(batch_size=batch_size, dtype=dtype, device=device, num_heads=h, head_dim=d)
                    session.put(_softmax_kv_key(layer, is_neg, is_cam), kv)

        for layer in range(self._num_blocks):
            for is_neg in (False, True):
                text = EncodeOnceKV()
                text.allocate()
                session.put(_text_key(layer, is_neg), text)

        warmup = LatentBuffer()
        warmup.allocate(maxlen=1)  # previous chunk's last frame, used to seed frame 0
        session.put(_WARMUP_KEY, warmup)

        session.attrs[_NUM_GDN_LAYERS] = len(self._gdn_layers)
        # Temporal-conv boundary state is created lazily on the first commit (its
        # shape depends on the spatial grid, unknown until the first forward), so
        # only the slot registry is (re)initialised here.
        session.attrs[_CONV_SLOTS_ATTR] = set()
        self.chunk_index = 0

    # -- GDN recurrent state --------------------------------------------

    def get_gdn_state(self, layer_index: int, is_negative: bool = False) -> dict[str, torch.Tensor]:
        """Return the live ``{state_kv, state_z}`` for a GDN block / CFG branch.

        On the reset chunk (``chunk_index == 0``) these are zeros and the caller
        should pass ``init_state=None`` to the scan; from chunk 1 on they hold the
        terminal forward state saved by the previous chunk.
        """
        return self._gdn_object(layer_index, is_negative).view()

    def commit_gdn_state(
        self, layer_index: int, state_kv: torch.Tensor, state_z: torch.Tensor, is_negative: bool = False
    ) -> None:
        """Overwrite a GDN block's recurrent state in place (end of a chunk)."""
        self._gdn_object(layer_index, is_negative).commit({_STATE_KV: state_kv, _STATE_Z: state_z})

    def _gdn_object(self, layer_index: int, is_negative: bool = False) -> FixedState:
        obj = self._session.get(_gdn_key(layer_index, is_negative))
        if not isinstance(obj, FixedState) or not obj.resident:
            raise RuntimeError(
                f"GDN state for layer {layer_index} not initialized; call create_state first "
                f"(GDN layers are {self._gdn_layers})."
            )
        return obj

    # -- softmax sink/window KV -----------------------------------------

    def get_softmax_kv(self, layer_index: int, is_negative: bool = False, is_cam: bool = False) -> torch.Tensor:
        return self._softmax_object(layer_index, is_negative, is_cam).view()

    def commit_softmax_kv(
        self, layer_index: int, kv: torch.Tensor, is_negative: bool = False, is_cam: bool = False
    ) -> None:
        self._softmax_object(layer_index, is_negative, is_cam).commit(kv)

    def _softmax_object(self, layer_index: int, is_negative: bool, is_cam: bool = False) -> PagedKV:
        obj = self._session.get(_softmax_kv_key(layer_index, is_negative, is_cam))
        if not isinstance(obj, PagedKV) or not obj.resident:
            raise RuntimeError(
                f"Softmax KV for layer {layer_index} not initialized; call create_state first "
                f"(softmax layers are {self._softmax_layers})."
            )
        return obj

    # -- temporal-conv boundary state -----------------------------------

    def conv_slot_keys(self) -> tuple[tuple[int, str], ...]:
        """The ``(block_index, slot)`` pairs that have committed conv state.

        Populated on commit as convs fire, so only the slots the model actually
        runs (e.g. no ``k_cam`` without a camera branch) are seeded next chunk.
        """
        return tuple(sorted(self._conv_slot_set()))

    def get_conv_state(self, layer_index: int, slot: str, is_negative: bool = False) -> torch.Tensor | None:
        """Return the previous chunk's trailing conv-input frames, or ``None``.

        ``None`` (the reset chunk, or a slot that has never fired) tells the caller
        to zero-pad -- identical to the non-autoregressive path.
        """
        obj = self._session.get(_conv_key(layer_index, slot, is_negative))
        if isinstance(obj, FixedState) and obj.resident:
            return obj.view()[_CONV_FRAMES]
        return None

    def commit_conv_state(
        self, layer_index: int, slot: str, frames: torch.Tensor, is_negative: bool = False
    ) -> None:
        """Store a chunk's trailing conv-input frames for the next chunk.

        The backing ``FixedState`` is allocated on first commit (its shape depends
        on the spatial grid) and overwritten in place thereafter.
        """
        key = _conv_key(layer_index, slot, is_negative)
        obj = self._session.get(key)
        if not isinstance(obj, FixedState) or not obj.resident:
            obj = FixedState()
            obj.allocate(shapes={_CONV_FRAMES: tuple(frames.shape)}, dtype=frames.dtype, device=frames.device)
            self._session.put(key, obj)
            self._conv_slot_set().add((layer_index, slot))
        obj.commit({_CONV_FRAMES: frames})

    def _conv_slot_set(self) -> set[tuple[int, str]]:
        slots = self._session.attrs.get(_CONV_SLOTS_ATTR)
        if not isinstance(slots, set):
            slots = set()
            self._session.attrs[_CONV_SLOTS_ATTR] = slots
        return slots

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
