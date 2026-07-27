# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter presenting ``LingbotWorldFastState``'s surface over the session manager.

``LingbotWorldFastStateAdapter`` exposes the exact public attributes and methods
that ``pipeline_lingbot_world_fast.py`` touches on its state object, so the
pipeline can use it interchangeably with the bespoke ``LingbotWorldFastState``
(behind the opt-in flag).

Lingbot's growing memory all lives in the AR-Diffusion engine's paged pool:
the self-attention KV that grows with every generated chunk, and the text
cross-attention K/V written once per session and reread. Neither is managed
here. What the pipeline itself holds per session is carried metadata only --
counters, shape constants, the per-layer attention end-index tensors, and the
last two decoded latents that warm the VAE decoder on the next extension call.
Every one of these is assigned wholesale by the model, read back, and cleared;
nothing accumulates and nothing can be released mid-session without changing
the output. Under the rule in ``docs/features/session_state_manager.md`` they
all belong in the session's ``attrs``, so this adapter declares no
``StateObject`` at all. That is a recorded decision, not an omission: it
measures how much of this model's session memory PR #3701 already moved into
the engine pool.

The manager still earns its keep on three counts: sessions are keyed by id
rather than held in a single mutable slot, ending a session releases its
buffers through one authority (``drop_session``), and every byte the session
carries -- including the decoder warm-up latents on the GPU -- shows up in
``SessionStateManager.stats()``, split by device.

The public methods mirror ``LingbotWorldFastState`` statement for statement, so
the two can be diffed to check equivalence. The one structural deviation is
construction: the bespoke class is built once per pipeline and calls
``reset()`` in ``__init__``, whereas an adapter is built per forward as a view
onto the pinned session and must not clear it -- a returning session keeps its
state. Resetting stays an explicit lifecycle event (``reset()``), never a side
effect of taking a view.
"""

from __future__ import annotations

import torch

from vllm_omni.experimental.world_models.session_state.attrs import SessionAttr
from vllm_omni.experimental.world_models.session_state.manager import SessionStateManager


class LingbotWorldFastStateAdapter:
    """Drop-in replacement for ``LingbotWorldFastState`` backed by the manager.

    Session-scoped metadata is declared once per attribute via ``SessionAttr``
    (reads fall back to the default; writes go through to the pinned session).
    The defaults are exactly the values the bespoke ``reset()`` assigns, so a
    fresh or freshly reset session reads identically on both paths.
    """

    current_start_frame = SessionAttr[int](default=0, coerce=int)
    local_end_index = SessionAttr[list[torch.Tensor] | None](default=None)
    global_end_index = SessionAttr[list[torch.Tensor] | None](default=None)
    is_initialized = SessionAttr[bool](default=False, coerce=bool)
    current_lat_f = SessionAttr[int](default=0, coerce=int)
    # The bespoke class stores the owning session's id and the pipeline compares
    # it against the incoming request to decide fresh-versus-extension. Keeping
    # it a session attribute preserves that logic verbatim: a session that has
    # not run yet (or was dropped) reads ``None`` and triggers the reset branch.
    session_id = SessionAttr[str | None](default=None)
    num_layers = SessionAttr[int | None](default=None)

    # Shape constants captured on the first call of a session and reused on
    # extension calls, where multi_modal_data["image"] is absent.
    h = SessionAttr[int | None](default=None)
    w = SessionAttr[int | None](default=None)
    lat_h = SessionAttr[int | None](default=None)
    lat_w = SessionAttr[int | None](default=None)
    frame_seqlen = SessionAttr[int | None](default=None)

    # Last few latents emitted by the diffusion loop on the previous call,
    # kept to warm the Wan VAE decoder's temporal feat_maps on extension.
    last_decoded_latent = SessionAttr[torch.Tensor | None](default=None)

    def __init__(self, session_id: str | None, manager: SessionStateManager) -> None:
        # Pin the session for this adapter's lifetime. The manager may evict the
        # session from its lookup table to bound retention, but an adapter that
        # is mid-forward keeps its own reference, so in-progress state is not
        # lost. A fresh adapter is built per forward, so the session is still
        # marked recently-used on each request.
        self._session = manager.get_or_create_session(session_id)

    # -- lifecycle, mirroring LingbotWorldFastState ----------------------

    def reset(self) -> None:
        """Clear all state.

        The bespoke ``reset()`` assigns every field its default; clearing the
        session's ``attrs`` makes every ``SessionAttr`` read fall back to that
        same default, and releases the tensors the attributes carried.
        """
        self._session.reset()

    def init_state(
        self,
        device: torch.device,
        num_layers: int,
    ) -> None:
        self.num_layers = num_layers

        self.local_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]
        self.global_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]

        self.is_initialized = True

    def advance(self, delta: int) -> None:
        self.current_lat_f += delta
