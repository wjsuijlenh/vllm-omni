# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 transformer.

Native vLLM-Omni port of the NVlabs SANA-WM DiT. Modules are built eagerly at
construction (under the loader's target-device context) and checkpoint tensors
are streamed in via ``load_weights``; the fused Bidirectional Gated DeltaNet
Triton kernels live in-tree in ``gdn.py``.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Iterable
from functools import reduce
from operator import mul
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.gdn import (
    SANA_WM_REQUIRE_TRITON_GDN_ENV,
    _flip_and_shift,
    cam_scan_bidi_chunkwise,
    reference_bidirectional_gated_delta_net,
    triton_bidirectional_gated_delta_net_from_qkv,
)
from vllm_omni.diffusion.models.sana_wm.ucpe import (
    cam_prep_func,
    prepare_cam_prep_context,
    prepare_prope_fns,
)

try:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.distributed.sp_plan import (
        SequenceParallelInput,
        SequenceParallelOutput,
    )
except ImportError:  # pragma: no cover - local macOS/dev environments may not install vLLM.
    Attention = None
    ColumnParallelLinear = None
    QKVParallelLinear = None
    QuantizationConfig = Any  # type: ignore[misc, assignment]
    RowParallelLinear = None
    SequenceParallelInput = None
    SequenceParallelOutput = None
    VllmRMSNorm = None
    get_tensor_model_parallel_rank = None
    get_tensor_model_parallel_world_size = None

SANA_WM_STAGE1_LATENT_CHANNELS = 128
SANA_WM_STAGE1_PROMPT_CHANNELS = 2304
SANA_WM_STAGE1_TIMESTEP_CHANNELS = 256
SANA_WM_DISABLE_VLLM_OPS_ENV = "VLLM_OMNI_SANA_WM_DISABLE_VLLM_OPS"
SANA_WM_STAGE1_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("y_embedder.", "transformer.y_embedder."),
    ("t_embedder.", "transformer.t_embedder."),
    ("t_block.", "transformer.t_block."),
    ("x_embedder.", "transformer.x_embedder."),
    ("final_layer.", "transformer.final_layer."),
    ("plucker_embedder.", "transformer.plucker_embedder."),
    ("raymap_embedder.", "transformer.raymap_embedder."),
    ("attention_y_norm.", "transformer.attention_y_norm."),
)

logger = logging.getLogger(__name__)

_SANA_WM_TRITON_GDN_FALLBACK_WARNED = False


def normalize_sana_wm_stage1_weight_name(name: str) -> str | None:
    """Map released Stage-1 checkpoint keys to planned vLLM-Omni names."""

    if name == "pos_embed":
        return "transformer.pos_embed"
    if name.startswith("blocks."):
        parts = name.split(".", 3)
        if len(parts) == 3 and parts[2] == "scale_shift_table":
            return f"transformer.blocks.{parts[1]}.scale_shift_table"
        if len(parts) < 4:
            return None
        _, block_idx, block_module, suffix = parts
        block_prefix = f"transformer.blocks.{block_idx}."
        if block_module == "attn":
            return f"{block_prefix}attn.{suffix}"
        if block_module == "cross_attn":
            return f"{block_prefix}cross_attn.{suffix}"
        if block_module == "mlp":
            return f"{block_prefix}mlp.{suffix}"
        if block_module == "plucker_post":
            return f"{block_prefix}plucker_proj.{suffix}"
        return f"{block_prefix}{block_module}.{suffix}"

    for source_prefix, target_prefix in SANA_WM_STAGE1_PREFIX_MAP:
        if name.startswith(source_prefix):
            return f"{target_prefix}{name.removeprefix(source_prefix)}"
    return None


def _vllm_tp_group_ready() -> bool:
    """Return True iff the vLLM tensor-parallel group is initialized."""
    if get_tensor_model_parallel_world_size is None:
        return False
    import vllm.distributed.parallel_state as ps

    return ps._TP is not None


def _get_tp_rank() -> int:
    if not _vllm_tp_group_ready():
        return 0
    return int(get_tensor_model_parallel_rank())


def _get_tp_world_size() -> int:
    if not _vllm_tp_group_ready():
        return 1
    return int(get_tensor_model_parallel_world_size())


def _sana_wm_disable_vllm_ops() -> bool:
    return os.environ.get(SANA_WM_DISABLE_VLLM_OPS_ENV, "").lower() in {"1", "true", "yes", "on"}


def _vllm_parallel_layers_available() -> bool:
    if _sana_wm_disable_vllm_ops():
        return False
    return (
        ColumnParallelLinear is not None
        and QKVParallelLinear is not None
        and RowParallelLinear is not None
        and _vllm_tp_group_ready()
    )


def _vllm_attention_available() -> bool:
    if _sana_wm_disable_vllm_ops():
        return False
    return Attention is not None


def _sequence_parallel_plan_available() -> bool:
    return SequenceParallelInput is not None and SequenceParallelOutput is not None


def _disable_tp_for_vllm_layer() -> bool:
    return not _vllm_tp_group_ready()


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, Any]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _linear_weight_dtype(module: nn.Module) -> torch.dtype:
    return module.weight.dtype


def _maybe_make_vllm_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    if VllmRMSNorm is None or _sana_wm_disable_vllm_ops():
        return SanaWmRMSNorm(hidden_size, eps=eps)
    return VllmRMSNorm(hidden_size, eps=eps)


def _is_sana_wm_transformer_block(name: str, module: Any) -> bool:
    del module
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()


def _prod(values: tuple[int, ...]) -> int:
    return reduce(mul, values, 1)


def _build_sana_wm_sp_plan() -> dict[str, Any] | None:
    """Declare Sana-WM Stage-1 sequence sharding boundaries.

    The first transformer block is the earliest point where video latents have
    already been patchified to ``[B, seq, hidden]``. The final layer is the last
    sequence-shaped output before unpatchify, so it is the natural gather point.
    RoPE is kept unsharded for now because the native GDN path consumes temporal
    layout information and needs a separate multi-card correctness pass.
    """

    if not _sequence_parallel_plan_available():
        return None
    assert SequenceParallelInput is not None
    assert SequenceParallelOutput is not None
    return {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
            "camera_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "final_layer": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }


def _to_3tuple(value: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 2:
        return (1, int(value[0]), int(value[1]))
    return (int(value[0]), int(value[1]), int(value[2]))


class SanaWmRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight


class SanaWmTensorParallelRMSNorm(nn.Module):
    """RMSNorm over a tensor-parallel sharded last dimension.

    SANA-WM q/k norm is defined across all heads. Under TP, q/k projection
    outputs are local head shards, so the RMS denominator must all-reduce the
    squared sum across TP ranks while keeping the affine weight local.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, tp_size: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.tp_size = max(int(tp_size), 1)
        self.global_hidden_size = hidden_size * self.tp_size

    @staticmethod
    def _all_reduce(tensor: torch.Tensor) -> None:
        # Only invoked when tp_size > 1 (gated by forward), so the vLLM TP group
        # must be initialized. Don't fall back to the default world group — that
        # is the data-parallel group and would silently corrupt the RMS.
        import vllm.distributed.parallel_state as ps

        if ps._TP is None:
            raise RuntimeError(
                "SanaWmTensorParallelRMSNorm requires an initialized vLLM tensor-parallel group when tp_size > 1."
            )
        ps._TP.all_reduce(tensor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        local_sum = hidden_float.pow(2).sum(dim=-1, keepdim=True)
        if self.tp_size > 1:
            self._all_reduce(local_sum)
        inv_rms = torch.rsqrt(local_sum / self.global_hidden_size + self.eps)
        out = hidden_float * inv_rms * self.weight.float()
        return out.to(input_dtype)


def _make_sharded_qk_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    tp_size = _get_tp_world_size()
    if tp_size > 1 and _vllm_tp_group_ready():
        return SanaWmTensorParallelRMSNorm(hidden_size, eps=eps, tp_size=tp_size)
    return _maybe_make_vllm_rms_norm(hidden_size, eps=eps)


class SanaWmTextProjection(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.fc1 = ColumnParallelLinear(
                prompt_channels,
                hidden_size,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.fc2 = RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.fc1 = nn.Linear(prompt_channels, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.fc2(self.act(_linear_output(self.fc1(prompt_embeds)))))


class SanaWmTextEmbedder(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        max_length: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.y_embedding = nn.Parameter(torch.zeros(max_length, prompt_channels))
        self.y_proj = SanaWmTextProjection(
            prompt_channels,
            hidden_size,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.y_proj" if prefix else "y_proj",
        )

    def forward(
        self,
        prompt_embeds: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prompt_embeds is None:
            prompt_embeds = self.y_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return self.y_proj(prompt_embeds.to(dtype=dtype))


class SanaWmTimestepEmbedder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            linear_1 = ColumnParallelLinear(
                in_features,
                hidden_size,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0" if prefix else "mlp.0",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            linear_2 = RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2" if prefix else "mlp.2",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            linear_1 = nn.Linear(in_features, hidden_size)
            linear_2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.mlp = nn.ModuleList([linear_1, self.act, linear_2])

    @staticmethod
    def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half, 1)
        )
        args = timestep.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(timestep, self.in_features)
        hidden_states = _linear_output(self.mlp[0](emb.to(dtype=_linear_weight_dtype(self.mlp[0]))))
        hidden_states = self.act(hidden_states)
        return _linear_output(self.mlp[2](hidden_states))


class SanaWmPatchEmbedMS3D(nn.Module):
    """Official-style 3D patch embedder used by SANA-WM Stage-1."""

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        hidden_size: int,
        *,
        kernel_size: tuple[int, int, int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.kernel_size = _to_3tuple(kernel_size or patch_size)
        self.flatten = True
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.kernel_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = nn.Identity()

    def project_with_shape(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        hidden_states = self.norm(self.proj(latents))
        _, _, frames, height, width = hidden_states.shape
        return hidden_states.flatten(2).transpose(1, 2), (frames, height, width)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.project_with_shape(latents)[0]


class SanaWmWanRotaryPosEmbed(nn.Module):
    """Wan-style 3D RoPE table used by the official SANA-WM GDN blocks."""

    def __init__(
        self,
        attention_head_dim: int,
        *,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self._split_sizes = (
            t_dim // 2,
            h_dim // 2,
            w_dim // 2,
        )
        self.freqs = self._build_freqs(max_seq_len)

    def _build_1d_freq(self, dim: int, positions: torch.Tensor) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(positions.shape[0], 0, dtype=torch.complex128)
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device)[: dim // 2] / dim)
        )
        phase = torch.outer(positions.to(torch.float64), freqs)
        return torch.polar(torch.ones_like(phase), phase)

    def _build_freqs(self, max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        dims = (
            self._split_sizes[0] * 2,
            self._split_sizes[1] * 2,
            self._split_sizes[2] * 2,
        )
        return torch.cat([self._build_1d_freq(dim, positions) for dim in dims], dim=1)

    def forward(self, spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        frames, height, width = spatial_shape
        if max(spatial_shape) > self.freqs.shape[0]:
            self.freqs = self._build_freqs(max(spatial_shape)).to(self.freqs.device)
        freqs = self.freqs.to(device=device)
        freqs_f, freqs_h, freqs_w = freqs.split(self._split_sizes, dim=1)
        f_dim, h_dim, w_dim = self._split_sizes
        parts = [
            freqs_f[:frames].view(frames, 1, 1, f_dim).expand(frames, height, width, f_dim),
            freqs_h[:height].view(1, height, 1, h_dim).expand(frames, height, width, h_dim),
            freqs_w[:width].view(1, 1, width, w_dim).expand(frames, height, width, w_dim),
        ]
        return torch.cat(parts, dim=-1).reshape(1, 1, frames * height * width, -1)


class SanaWmCameraEmbedder(nn.Module):
    """Small camera branch used by the native path."""

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        use_vllm_parallel_layers: bool = True,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = config or SanaWmConfig()
        self.hidden_size = config.hidden_size
        self.plucker = nn.Module()
        # Conv3d has no vLLM parallel equivalent — kept as nn.Conv3d. Under TP>1
        # its weight is replicated and every rank runs the same conv (redundant
        # but cheap for a 1x1 conv); intentional, and yields a full-width output.
        self.plucker.proj = nn.Conv3d(config.chunk_plucker_channels, config.hidden_size, kernel_size=1)
        self.raymap = nn.Module()
        # raymap.proj is a plain Linear over 20 Plücker features; use
        # ColumnParallelLinear when vLLM TP layers are available.
        if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
            self.raymap.proj: nn.Module = ColumnParallelLinear(
                20,
                config.hidden_size,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.raymap.proj" if prefix else "raymap.proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.raymap.proj = nn.Linear(20, config.hidden_size)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def forward(
        self,
        *,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        frames, height, width = spatial_shape
        expected_tokens = frames * height * width
        hidden_states = None

        if plucker is not None:
            if plucker.ndim == 4:
                plucker = plucker.unsqueeze(0)
            plucker = plucker.to(device=device, dtype=dtype)
            if plucker.shape[0] == 1 and batch_size > 1:
                plucker = plucker.expand(batch_size, -1, -1, -1, -1)
            plucker_hidden = self.plucker.proj(plucker).flatten(2).transpose(1, 2)
            hidden_states = self._match_tokens(plucker_hidden, expected_tokens)

        if raymap is not None:
            if raymap.ndim == 2:
                raymap = raymap.unsqueeze(0)
            raymap = raymap.to(device=device, dtype=dtype)
            if raymap.shape[0] == 1 and batch_size > 1:
                raymap = raymap.expand(batch_size, -1, -1)
            ray_hidden = self.raymap.proj(raymap)
            ray_hidden = ray_hidden.repeat_interleave(height * width, dim=1)
            ray_hidden = self._match_tokens(ray_hidden, expected_tokens)
            # Both branches are full-width and rank-consistent (plucker via the
            # replicated Conv3d, ray via ColumnParallelLinear gather_output=True),
            # so the sum is well-defined and matches the full hidden_size input
            # expected by the downstream QKVParallelLinear (which splits by head).
            hidden_states = ray_hidden if hidden_states is None else hidden_states + ray_hidden

        return hidden_states


class SanaWmSelfAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        use_gdn: bool = True,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_heads
        self.heads = self.total_num_heads
        self.dim = self.head_dim
        self.in_dim = hidden_size
        self.out_dim = hidden_size
        self.eps = 1e-8
        self.attn_type = config.attn_type
        self.use_gdn = use_gdn and "GDN" in config.attn_type
        self.use_vllm_attention = (not self.use_gdn) and _vllm_attention_available()
        self.k_conv_only = config.k_conv_only
        self.conv_kernel_size = config.conv_kernel_size
        self.patch_size = _to_3tuple(config.patch_size)
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        # Total camera branch width from the checkpoint. TP ColumnParallel
        # layers expose a per-rank local slice at runtime; keep both sizes so
        # layer construction uses the global contract while local conv/norm
        # tensors match q/k/v outputs.
        self.total_cam_dim = hidden_size // max(config.cam_attn_compress, 1)
        self.cam_dim = self.total_cam_dim
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert QKVParallelLinear is not None
            assert RowParallelLinear is not None
            self.qkv = QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv" if prefix else "qkv",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.num_heads = self.qkv.num_heads
            self.num_kv_heads = self.qkv.num_kv_heads
            self.proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj" if prefix else "proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.beta_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.beta_proj" if prefix else "beta_proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj" if prefix else "gate_proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.output_gate = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate" if prefix else "output_gate",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.q_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.total_cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj_cam" if prefix else "q_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.k_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.total_cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.k_proj_cam" if prefix else "k_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.v_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.total_cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.v_proj_cam" if prefix else "v_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.out_proj_cam = RowParallelLinear(
                self.total_cam_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj_cam" if prefix else "out_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.qkv = nn.Linear(hidden_size, self.num_heads * self.head_dim * 3, bias=False)
            self.proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)
            self.beta_proj = nn.Linear(hidden_size, self.num_heads)
            self.gate_proj = nn.Linear(hidden_size, self.num_heads)
            self.output_gate = nn.Linear(hidden_size, hidden_size)
            self.q_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.k_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.v_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.out_proj_cam = nn.Linear(self.cam_dim, hidden_size, bias=True)
        local_inner_dim = self.num_heads * self.head_dim
        norm_cls = _make_sharded_qk_rms_norm if config.qk_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner_dim)
        self.k_norm = norm_cls(local_inner_dim)
        self.softmax_attn = (
            Attention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=False,
                role="self",
                qkv_layout="BSND",
                prefix=prefix,
            )
            if self.use_vllm_attention
            else None
        )

        # These names mirror the official GDN checkpoint. The current forward
        # executes a pure PyTorch recurrence; the fused Triton scan is ported
        # separately.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.conv_k = (
            nn.Conv1d(
                local_inner_dim,
                local_inner_dim,
                kernel_size=config.conv_kernel_size,
                groups=local_inner_dim,
                bias=False,
            )
            if self.use_gdn and config.conv_kernel_size > 0
            else None
        )
        self.conv_q = None
        self.conv_v = None

        if self.uses_vllm_parallel_layers:
            self.cam_dim = int(getattr(self.q_proj_cam, "output_size_per_partition", self.total_cam_dim))

        cam_compress = max(config.cam_attn_compress, 1)
        self.cam_heads = max(self.num_heads // cam_compress, 1)
        if self.cam_dim % self.cam_heads != 0:
            raise ValueError(f"Sana-WM local cam_dim={self.cam_dim} must be divisible by cam_heads={self.cam_heads}.")
        self.cam_head_dim = self.cam_dim // self.cam_heads
        # Under TP, cam_head_dim must equal main head_dim so the WAN RoPE
        # (built for main head_dim) can be correctly sliced for the cam branch.
        # A mismatch indicates the cam ColumnParallelLinear layers were not
        # partitioned consistently with the main QKVParallelLinear.
        if self.cam_head_dim != self.head_dim:
            raise ValueError(
                f"Sana-WM cam_head_dim={self.cam_head_dim} != head_dim={self.head_dim}. "
                f"Under TP the cam layers must be partitioned by the same TP degree as the "
                f"main QKV (cam_dim={self.cam_dim}, cam_heads={self.cam_heads}, "
                f"total_cam_dim={self.total_cam_dim})."
            )
        # q_proj_cam / k_proj_cam / v_proj_cam / out_proj_cam are initialised
        # in the TP/fallback conditional above so they follow the same
        # ColumnParallelLinear / RowParallelLinear pattern as qkv and proj.
        self.q_norm_cam = norm_cls(self.cam_dim)
        self.k_norm_cam = norm_cls(self.cam_dim)
        self.conv_k_cam = (
            nn.Conv1d(
                self.cam_dim,
                self.cam_dim,
                kernel_size=config.conv_kernel_size,
                groups=self.cam_dim,
                bias=False,
            )
            if self.use_gdn and config.conv_kernel_size > 0
            else None
        )
        self.conv_q_cam = None
        self.conv_v_cam = None
        self._init_short_convs()

    @staticmethod
    def _init_short_conv(conv: nn.Conv1d | None) -> None:
        if conv is None:
            return
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()

    def _init_short_convs(self) -> None:
        for conv in (self.conv_q, self.conv_k, self.conv_v, self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            self._init_short_conv(conv)

    @staticmethod
    def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
        flipped = torch.flip(tensor, dims=[dim])
        shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = 1
        padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([padding, shifted], dim=dim)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
        output = torch.view_as_real(rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return output.type_as(hidden_states)

    @classmethod
    def _apply_rotary_emb_to_sdpa(cls, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return cls._apply_rotary_emb(hidden_states.transpose(2, 3), freqs).transpose(2, 3)

    @staticmethod
    def _reshape_to_temporal(
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM temporal conv expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        hidden_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size * spatial_tokens, frames, hidden_size)
        return hidden_states, batch_size, spatial_tokens, frames

    @staticmethod
    def _reshape_from_temporal(
        hidden_states: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        frames: int,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        return (
            hidden_states.reshape(batch_size, spatial_tokens, frames, hidden_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, frames * spatial_tokens, hidden_size)
        )

    @staticmethod
    def _causal_conv_1d(hidden_states: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        dtype = hidden_states.dtype
        conv_input = hidden_states.transpose(1, 2).to(conv.weight.dtype)
        conv_input = F.pad(conv_input, (conv.kernel_size[0] - 1, 0))
        output = conv(conv_input).transpose(1, 2)
        return output.to(dtype)

    def _bidirectional_temporal_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv: nn.Conv1d,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden_states, batch_size, spatial_tokens, frames = self._reshape_to_temporal(hidden_states, spatial_shape)
        forward_states = self._causal_conv_1d(hidden_states, conv)
        backward_states = self._causal_conv_1d(hidden_states.flip(1), conv).flip(1)
        center_weight = conv.weight[:, 0, -1].to(hidden_states.dtype)
        center_states = hidden_states * center_weight.view(1, 1, -1)
        output = forward_states + backward_states - center_states
        return self._reshape_from_temporal(output, batch_size, spatial_tokens, frames)

    def _compute_frame_gates(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN token layout mismatch: N={token_count}, expected {frames * spatial_tokens}.")
        beta = torch.sigmoid(_linear_output(self.beta_proj(hidden_states)))
        beta = beta.reshape(batch_size, frames, spatial_tokens, self.num_heads).permute(0, 3, 1, 2)
        frame_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size).mean(dim=2)
        gate = _linear_output(self.gate_proj(frame_states)).float()
        decay = torch.exp(
            -self.A_log.float().exp().view(1, 1, -1) * F.softplus(gate + self.dt_bias.float().view(1, 1, -1))
        )
        return beta, decay.transpose(1, 2)

    def _forward_gdn_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run main-branch bidirectional GDN and return the (B, N, q_size)
        raw output BEFORE ``output_gate`` and ``proj`` are applied.

        Returns ``(raw_output, (beta, decay))`` so the caller can share the
        beta/decay gates with the camera branch (matches NVlabs' shared
        ``precomputed_gates`` plumbing).
        """
        batch_size, token_count, _hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        if self.conv_k is not None:
            key = self._bidirectional_temporal_short_conv(key, self.conv_k, spatial_shape)
        if precomputed_gates is None:
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
        else:
            beta, decay = precomputed_gates

        if hidden_states.is_cuda:
            qkv = torch.stack(
                (
                    query.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    key.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    value.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                ),
                dim=2,
            )
            try:
                output = triton_bidirectional_gated_delta_net_from_qkv(
                    qkv,
                    beta=beta,
                    decay=decay,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    spatial_tokens=spatial_tokens,
                    rotary_emb=rotary_emb,
                    k_scale=(self.head_dim**-0.5) * (spatial_tokens**-0.5),
                    eps=self.eps,
                )
                output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, q_size)
                return output, (beta, decay)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if os.environ.get(SANA_WM_REQUIRE_TRITON_GDN_ENV, "").lower() in {"1", "true", "yes", "on"}:
                    raise
                # The fused path is an optimization. Keep the reference path as
                # the correctness fallback for unsupported GPUs / Triton builds.
                # Warn once so a silent multi-second regression is observable.
                global _SANA_WM_TRITON_GDN_FALLBACK_WARNED
                if not _SANA_WM_TRITON_GDN_FALLBACK_WARNED:
                    _SANA_WM_TRITON_GDN_FALLBACK_WARNED = True
                    warnings.warn(
                        f"Sana-WM fused Triton GDN kernel raised {type(exc).__name__}: {exc}; "
                        "falling back to the PyTorch reference recurrence (slow). "
                        f"Set {SANA_WM_REQUIRE_TRITON_GDN_ENV}=1 to fail fast instead.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        query = self.q_norm(query).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = self.k_norm(key).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim)

        query = F.relu(query).permute(0, 2, 3, 1)
        key = F.relu(key).permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        key = key * ((self.head_dim**-0.5) * (spatial_tokens**-0.5))
        if rotary_emb is not None:
            query_rot = self._apply_rotary_emb(query, rotary_emb)
            key_rot = self._apply_rotary_emb(key, rotary_emb)
        else:
            query_rot = query
            key_rot = key

        output = reference_bidirectional_gated_delta_net(
            query,
            key,
            value,
            beta=beta,
            decay=decay,
            spatial_tokens=spatial_tokens,
            query_rot=query_rot,
            key_rot=key_rot,
            eps=self.eps,
        )
        output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, q_size)
        return output, (beta, decay)

    def _apply_output_gate_and_proj(
        self,
        combined: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SiLU output gate (driven by hidden_states) + final projection.

        Centralised so the same shared gate/proj path is applied whether or
        not the camera branch contributes to ``combined``.
        """
        gate = F.silu(_linear_output(self.output_gate(hidden_states)).float())
        gated = combined * gate
        return _linear_output(self.proj(gated.to(_linear_weight_dtype(self.proj))))

    @staticmethod
    def _match_local_inner_dim(contrib: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Slice a full TP contribution back to the local inner dimension.

        Some auxiliary projections, notably the camera branch's
        ``out_proj_cam``, naturally all-reduce to the full hidden width.
        The shared GDN raw stream is TP-local before ``output_gate`` and
        ``proj``, so the auxiliary contribution must use the matching local
        slice before the raw streams are added.
        """
        if contrib.shape[-1] == reference.shape[-1]:
            return contrib
        local_width = reference.shape[-1]
        if local_width <= 0 or contrib.shape[-1] % local_width != 0:
            raise ValueError(
                f"Sana-WM TP contribution width mismatch: contrib={contrib.shape[-1]} reference={local_width}."
            )
        tp_rank = _get_tp_rank()
        start = tp_rank * local_width
        end = start + local_width
        if end > contrib.shape[-1]:
            raise ValueError(
                "Sana-WM TP contribution slice is out of range: "
                f"rank={tp_rank} slice=({start}, {end}) width={contrib.shape[-1]}."
            )
        return contrib[..., start:end].contiguous()

    def _forward_gdn(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Main-branch only GDN forward with output_gate + proj applied."""
        raw, _ = self._forward_gdn_raw(hidden_states, spatial_shape, rotary_emb)
        return self._apply_output_gate_and_proj(raw, hidden_states)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _reshape_to_seq_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    @staticmethod
    def _downscale_to_reference_rms(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Downscale ``transformed`` so its per-token channel RMS does not
        exceed ``ref``'s. Mirrors NVlabs ``_downscale_to_reference_rms``.

        Inputs are ``(B, H, N, D)`` (channel dim last). The RMS-scale is
        clamped at 1.0 — the function only ever shrinks, never amplifies.
        """
        ref_rms = ref.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
        return transformed * scale

    @staticmethod
    def _ucpe_rotary_freqs(rotary_emb: torch.Tensor | None) -> torch.Tensor | None:
        if rotary_emb is None:
            return None
        rotary_emb_freqs = rotary_emb.squeeze(0).squeeze(0)
        if rotary_emb_freqs.ndim == 3:
            # vLLM-Omni packs RoPE as (1, 1, N, D//2) complex. Some
            # call-sites may pass an already squeezed (N, D//2) tensor.
            rotary_emb_freqs = rotary_emb_freqs.squeeze(0)
        return rotary_emb_freqs

    def _forward_softmax_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Softmax self-attention raw output before output_gate/proj.

        Mirrors NVlabs ``_forward_softmax_attn(..., apply_output_gate=False)``
        for every-N-th hybrid blocks. GDN-specific gates are intentionally not
        used here.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(f"Sana-WM softmax attention expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.q_norm(query).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = self.k_norm(key).reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)
            key = self._apply_rotary_emb(key.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)

        query = query.transpose(1, 2)  # (B, H, N, D)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        dtype_orig = hidden_states.dtype
        # NVlabs runs SDPA in bf16 even when ``fp32_attention=True`` because
        # FlashAttention only supports bf16/fp16; fp32 falls back to the
        # math backend. Match that semantics: if the upstream code path
        # promoted Q/K/V to fp32 (e.g. q/k_norm in fp32), demote back to
        # bf16 before SDPA. See
        # ``NVlabs-Sana/diffusion/model/nets/sana_gdn_camctrl_blocks.py``
        # ``_forward_softmax_attn_sdpa`` (the comment "SDPA / FlashAttention
        # only supports bf16/fp16; fp32 falls back to math backend.").
        if query.dtype == torch.float32:
            query = query.bfloat16()
            key = key.bfloat16()
            value = value.bfloat16()
        attn = F.scaled_dot_product_attention(query, key, value)
        return attn.transpose(1, 2).reshape(batch_size, token_count, q_size).to(dtype_orig)

    def _forward_softmax_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """UCPE camera branch for softmax hybrid blocks.

        Matches NVlabs ``_forward_cam_branch_softmax``: camera Q/K/V are
        projected from the same hidden stream, transformed by UCPE, run through
        SDPA, inverse-transformed, then returned as raw ``cam_dim`` features.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(f"Sana-WM softmax cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        q_cam = _linear_output(self.q_proj_cam(hidden_states))
        k_cam = _linear_output(self.k_proj_cam(hidden_states))
        v_cam = _linear_output(self.v_proj_cam(hidden_states))

        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        q_cam = self.q_norm_cam(q_cam).reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        k_cam = self.k_norm_cam(k_cam).reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        v_cam = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)

        q_cam = q_cam.transpose(1, 2).contiguous()  # (B, H_cam, N, D)
        k_cam = k_cam.transpose(1, 2).contiguous()
        v_cam = v_cam.transpose(1, 2).contiguous()

        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_fns(
            head_dim=self.cam_head_dim,
            camera_conditions=camera_conditions,
            hw=spatial_shape,
            patch_size=self.patch_size,
            rotary_emb=self._ucpe_rotary_freqs(rotary_emb),
        )
        q_cam_trans = apply_fn_q(q_cam)
        kv_cam_trans = apply_fn_kv(torch.cat([k_cam, v_cam], dim=1))
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        q_cam_trans = self._downscale_to_reference_rms(q_cam, q_cam_trans)
        k_cam_trans = self._downscale_to_reference_rms(k_cam, k_cam_trans)
        v_cam_trans = self._downscale_to_reference_rms(v_cam, v_cam_trans)

        dtype_orig = hidden_states.dtype
        query = q_cam_trans
        key = k_cam_trans
        value = v_cam_trans
        # Match NVlabs ``_forward_softmax_attn_sdpa``: promote to fp32 to honor
        # ``fp32_attention=True``, then demote back to bf16 for SDPA so we
        # get FlashAttention instead of the fp32 math-backend fallback. The
        # net SDPA math is therefore bf16 (matching NVlabs), but Q/K/V are
        # composed in fp32 prior to the demote. See the comment in
        # ``sana_gdn_camctrl_blocks.py::_forward_softmax_attn_sdpa``:
        # "SDPA / FlashAttention only supports bf16/fp16; fp32 falls back to
        # math backend."
        if getattr(self, "fp32_attention", True):
            query = query.float()
            key = key.float()
            value = value.float()
        if query.dtype == torch.float32:
            query = query.bfloat16()
            key = key.bfloat16()
            value = value.bfloat16()

        head_dim = query.shape[-1]
        needs_pad = head_dim not in (32, 64, 128, 256) and head_dim < 256
        if needs_pad:
            pad_to = 128 if head_dim <= 128 else 256
            pad_size = pad_to - head_dim
            query = F.pad(query, (0, pad_size))
            key = F.pad(key, (0, pad_size))
            value = F.pad(value, (0, pad_size))

        out = F.scaled_dot_product_attention(query, key, value)
        if needs_pad:
            out = out[..., :head_dim]
        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)
        out = apply_fn_o(out)
        return out.transpose(1, 2).reshape(batch_size, token_count, self.cam_dim)

    @staticmethod
    def _cam_single_path_delta_scan(
        q_rot: torch.Tensor,  # (B, H, D, N)
        k_rot: torch.Tensor,  # (B, H, D, N)
        value: torch.Tensor,  # (B, H, D, N)
        beta: torch.Tensor,  # (B, H, T, S)
        decay: torch.Tensor,  # (B, H, T)
        *,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Numerator-only delta-rule recurrence used by the SANA-WM cam branch.

        Matches NVlabs ``torch_recurrent_cam_single_path_delta_rule``: no Z
        denominator stream, no final divide. Output is ``state_kv @ q_rot``
        per frame, stacked back into ``(B, H, D, N)`` layout.

        Runs in fp32 internally for numerical stability (matches NVlabs
        ``fp32_attention=True``) and casts the result back to the input dtype.
        """
        dtype_orig = q_rot.dtype
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        value = value.float()
        beta = beta.float()
        decay = decay.float()

        batch_size, num_heads, head_dim, token_count = q_rot.shape
        frames = beta.shape[2]
        if token_count != frames * spatial_tokens:
            raise ValueError(f"single-path scan token_count={token_count} != frames*S={frames * spatial_tokens}.")

        def to_frames(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

        q_rot_f = to_frames(q_rot)
        k_rot_f = to_frames(k_rot)
        value_f = to_frames(value)
        state_kv = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=q_rot.device, dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        for frame_idx in range(frames):
            qrt = q_rot_f[:, :, frame_idx]
            krt = k_rot_f[:, :, frame_idx]
            vt = value_f[:, :, frame_idx]
            bt = beta[:, :, frame_idx].unsqueeze(2)
            gt = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

            state_kv = state_kv * gt
            v_pred = torch.matmul(state_kv, krt)
            delta_v = (vt - v_pred) * bt
            state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
            outputs.append(torch.matmul(state_kv, qrt))

        stacked = torch.stack(outputs, dim=2)
        out = stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)
        return out.to(dtype_orig)

    def _bidi_single_path(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        *,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Bidirectional single-path scan: forward + backward, simple sum.

        Backward direction shifts K/V/beta by one frame (with zero pad) and
        the decay by one frame (with neutral 1.0 pad), matching NVlabs
        ``flip_and_shift`` conventions.

        Set ``SANA_WM_CAM_TRITON=1`` to dispatch to the in-tree Triton port
        ``gdn.cam_scan_bidi_chunkwise``; the default is the Python reference
        path below.
        """
        if q_rot.is_cuda and os.environ.get("SANA_WM_CAM_TRITON", "").lower() in {"1", "true", "yes", "on"}:
            dtype_orig = q_rot.dtype
            B, H, D, N = q_rot.shape
            F = beta.shape[2]
            if N != F * spatial_tokens:
                raise ValueError(f"cam triton path: N={N} != F*S={F * spatial_tokens}")
            # The kernel requires fp32 contiguous q/k/v, beta (B,H,F,S), decay (B,H,F).
            q_f = q_rot.float().contiguous()
            k_f = k_rot.float().contiguous()
            v_f = value.float().contiguous()
            if beta.ndim == 3:
                beta_f = beta.float().unsqueeze(-1).expand(B, H, F, spatial_tokens).contiguous()
            else:
                beta_f = beta.float().contiguous()
            decay_f = decay.float().contiguous()
            out = cam_scan_bidi_chunkwise(q_f, k_f, v_f, beta_f, decay_f)
            return out.to(dtype_orig)
        batch_size, num_heads, head_dim, token_count = q_rot.shape
        frames = beta.shape[2]

        out_fwd = self._cam_single_path_delta_scan(q_rot, k_rot, value, beta, decay, spatial_tokens=spatial_tokens)

        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)

        q_T = to_time(q_rot)
        k_T = to_time(k_rot)
        v_T = to_time(value)
        q_bwd = torch.flip(q_T, dims=[2])
        k_bwd = _flip_and_shift(k_T, dim=2, shift_value=0.0)
        v_bwd = _flip_and_shift(v_T, dim=2, shift_value=0.0)
        beta_bwd = _flip_and_shift(beta, dim=2, shift_value=0.0)
        decay_bwd = _flip_and_shift(decay, dim=2, shift_value=1.0)

        out_bwd_flipped = self._cam_single_path_delta_scan(
            from_time(q_bwd),
            from_time(k_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
            spatial_tokens=spatial_tokens,
        )
        out_bwd = torch.flip(
            out_bwd_flipped.view(batch_size, num_heads, head_dim, frames, spatial_tokens),
            dims=[3],
        ).reshape(batch_size, num_heads, head_dim, token_count)
        return out_fwd + out_bwd

    def _forward_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """UCPE camera branch — matches the NVlabs production
        ``BidirectionalGDNUCPESinglePathLiteLABothTriton`` variant declared
        in the SANA-WM 1600M release config.

        Pipeline (mirrors the released NVlabs
        ``BidirectionalGDNUCPESinglePathLiteLABothTriton`` path):

        1. Project Q/K/V from ``hidden_states`` (all three from x — UCPE is
           NOT cross-attention; camera info enters via per-pixel projection
           matrices).
        2. Short conv on K (and optionally Q/V).
        3. ``q_norm_cam`` / ``k_norm_cam`` (RMSNorm over cam_dim).
        4. ReLU kernel + ``k_scale = D^-0.5 * S^-0.5``.
        5. UCPE per-token block-diagonal transforms via ``apply_fn_q`` and
           ``apply_fn_kv`` (per-ray 4x4 projection on first half + sliced
           complex RoPE on second half).
        6. Dynamic Beta Discounting: β_cam = β / frame_inflation_sq.clamp_min(1).
           The released NVlabs ``BothTriton`` cam path does not apply the
           Python ``PostUCPERenorm`` shrink step before the scan; it uses the
           raw UCPE-transformed Q/K/V and discounts beta from their raw K
           inflation.
        7. **Single-path** delta-rule recurrence (numerator only, no Z
           denominator) — forward + backward, simple sum.
        8. **Inverse output transform**: ``apply_fn_o`` brings the recurrence
           output from the per-ray frame back to the world frame.

        Returns ``(B, N, cam_dim)`` raw — caller applies ``out_proj_cam`` and
        the shared ``output_gate`` + ``proj``.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        # Build the cam-prep context once per call. This mirrors NVlabs'
        # fused ``cam_prep_func`` input contract instead of routing through
        # the Python ``prepare_prope_fns`` Q/K/V closures.
        cam_prep_context = prepare_cam_prep_context(
            camera_conditions=camera_conditions,
            spatial_shape=spatial_shape,
            patch_size=self.patch_size,
            head_dim=self.cam_head_dim,
            rotary_emb=self._ucpe_rotary_freqs(rotary_emb),
        )

        # 1. Projections (all from x).
        q_cam = _linear_output(self.q_proj_cam(hidden_states))
        k_cam = _linear_output(self.k_proj_cam(hidden_states))
        v_cam = _linear_output(self.v_proj_cam(hidden_states))

        # 2. Short conv on K (and optionally Q, V).
        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        # 3-6. Fused cam-prep semantics: RMSNorm/ReLU/K-scale, UCPE
        # projection, RoPE, and K inflation tracking. Outputs already use
        # NVlabs' ``(B, H, D, N)`` scan layout.
        q_raw = q_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).contiguous()
        k_raw = k_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).contiguous()
        v_raw = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).contiguous()
        k_scale = (self.cam_head_dim**-0.5) * (spatial_tokens**-0.5)
        norm_eps_val = float(getattr(self.q_norm_cam, "eps", getattr(self.q_norm_cam, "variance_epsilon", 1e-6)))
        q_rot_bhdn, k_rot_bhdn, v_bhdn, inflation_sq = cam_prep_func(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight=self.q_norm_cam.weight.float().contiguous(),
            k_norm_weight=self.k_norm_cam.weight.float().contiguous(),
            proj_q=cam_prep_context.proj_q,
            proj_kv=cam_prep_context.proj_kv,
            rope_cos=cam_prep_context.rope_cos,
            rope_sin=cam_prep_context.rope_sin,
            k_scale=k_scale,
            norm_eps=norm_eps_val,
        )

        # 7. Dynamic Beta Discounting (per-frame mean over spatial tokens).
        beta, decay = precomputed_gates
        inflation_per_token = inflation_sq.reshape(batch_size, self.cam_heads, frames, spatial_tokens)
        frame_inflation_sq = inflation_per_token.mean(dim=-1)  # (B, H_cam, T)
        if beta.shape[1] != self.cam_heads:
            repeat_factor = self.cam_heads // beta.shape[1]
            if repeat_factor * beta.shape[1] != self.cam_heads:
                raise ValueError(
                    "Sana-WM cam branch expects cam_heads to be a multiple of main heads,"
                    f" got cam_heads={self.cam_heads} main_heads={beta.shape[1]}."
                )
            beta_cam = beta.repeat_interleave(repeat_factor, dim=1)
            decay_cam = decay.repeat_interleave(repeat_factor, dim=1)
        else:
            beta_cam = beta
            decay_cam = decay
        beta_cam = beta_cam / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # 8. Single-path bidirectional delta-rule recurrence on the UCPE-
        # transformed Q/K/V (no Z denominator, no final divide).
        cam_out_bhdn = self._bidi_single_path(
            q_rot_bhdn, k_rot_bhdn, v_bhdn, beta_cam, decay_cam, spatial_tokens=spatial_tokens
        )

        # 9. Apply inverse UCPE transform on the recurrence output.
        # ``apply_fn_o`` expects (B, H, N, D); we currently have
        # (B, H, D, N) — transpose, apply, transpose back.
        cam_out_bhnd = cam_out_bhdn.transpose(-1, -2).contiguous()
        cam_out_bhnd = cam_prep_context.apply_output(cam_out_bhnd)
        cam_out_bhdn = cam_out_bhnd.transpose(-1, -2).contiguous()

        # (B, H_cam, D_cam, N) → (B, N, cam_dim)
        return cam_out_bhdn.permute(0, 3, 1, 2).reshape(batch_size, token_count, self.cam_dim)

    # The GDN/UCPE branch dispatches into a hand-written Triton kernel whose
    # launch grid is computed from runtime shapes; under torch.compile those
    # become symbolic and the grid degrades to a float ("'float' object cannot
    # be interpreted as an integer" at Triton launch). Run the whole attention
    # eagerly (graph break) so the kernel sees concrete int shapes; the rest of
    # the block (MLP / norms / cross-attention) still compiles.
    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with dual-branch GDN+UCPE when ``camera_conditions`` is
        provided, otherwise main-branch-only GDN or softmax fallback.

        ``camera_conditions`` is the raw ``(B, T_latent, 20)`` raymap tensor
        produced by :func:`camera_control._pack_camera_conditions` — NOT the
        embedder output. The cam branch consumes ``hidden_states`` as the
        sole Q/K/V source and uses ``camera_conditions`` to build per-pixel
        4x4 transforms that are applied to the Q/K/V channels via UCPE.
        """
        if spatial_shape is not None and self.use_gdn:
            if camera_conditions is None or self.q_proj_cam is None:
                # Main branch only — preserves legacy path when the request
                # has no camera info or the cam projections were not built.
                return self._forward_gdn(hidden_states, spatial_shape, rotary_emb)
            # Dual-branch: precompute β/decay once, run main raw, cam raw,
            # combine, then apply shared output_gate + proj exactly once.
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
            main_raw, _ = self._forward_gdn_raw(
                hidden_states,
                spatial_shape,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_raw = self._forward_cam_branch(
                hidden_states,
                spatial_shape,
                camera_conditions,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_contrib = _linear_output(self.out_proj_cam(cam_raw))
            cam_contrib = self._match_local_inner_dim(cam_contrib, main_raw)
            combined = main_raw + cam_contrib.to(main_raw.dtype)
            attn_out = self._apply_output_gate_and_proj(combined, hidden_states)
            return attn_out

        if spatial_shape is not None:
            main_raw = self._forward_softmax_raw(hidden_states, spatial_shape, rotary_emb)
            if camera_conditions is not None and self.q_proj_cam is not None:
                cam_raw = self._forward_softmax_cam_branch(
                    hidden_states,
                    spatial_shape,
                    camera_conditions,
                    rotary_emb,
                )
                cam_contrib = _linear_output(self.out_proj_cam(cam_raw))
                cam_contrib = self._match_local_inner_dim(cam_contrib, main_raw)
                main_raw = main_raw + cam_contrib.to(main_raw.dtype)
            return self._apply_output_gate_and_proj(main_raw, hidden_states)

        # Shape-only fallback for legacy callers that do not provide a token-grid shape.
        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        query = self._split_heads(self.q_norm(query))
        key = self._split_heads(self.k_norm(key))
        value = self._split_heads(value)
        if rotary_emb is not None:
            query = self._apply_rotary_emb_to_sdpa(query, rotary_emb)
            key = self._apply_rotary_emb_to_sdpa(key, rotary_emb)
        attn = F.scaled_dot_product_attention(query, key, value)
        return self._apply_output_gate_and_proj(self._merge_heads(attn), hidden_states)


class SanaWmCrossAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        inner = self.total_num_heads * self.head_dim
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.q_linear = ColumnParallelLinear(
                hidden_size,
                inner,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_linear" if prefix else "q_linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.kv_linear = ColumnParallelLinear(
                hidden_size,
                inner * 2,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_linear" if prefix else "kv_linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.num_heads = inner // max(_get_tp_world_size(), 1) // self.head_dim
            self.proj = RowParallelLinear(
                inner,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj" if prefix else "proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.q_linear = nn.Linear(hidden_size, inner)
            self.kv_linear = nn.Linear(hidden_size, inner * 2)
            self.proj = nn.Linear(inner, hidden_size)
        local_inner = self.num_heads * self.head_dim
        norm_cls = _make_sharded_qk_rms_norm if config.cross_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner)
        self.k_norm = norm_cls(local_inner)
        self.softmax_attn = (
            Attention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_heads,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=False,
                role="cross",
                qkv_layout="BSND",
                skip_sequence_parallel=True,
                disable_kv_quant=True,
                prefix=prefix,
            )
            if _vllm_attention_available()
            else None
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _reshape_to_seq_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    @staticmethod
    def _slice_local_inner(tensor: torch.Tensor, local_inner: int) -> torch.Tensor:
        if tensor.shape[-1] == local_inner:
            return tensor
        tp_size = _get_tp_world_size()
        if tp_size <= 1 or tensor.shape[-1] != local_inner * tp_size:
            raise ValueError(
                "Sana-WM cross-attention TP inner width mismatch: "
                f"tensor={tensor.shape[-1]} local_inner={local_inner} tp={tp_size}."
            )
        start = _get_tp_rank() * local_inner
        return tensor[..., start : start + local_inner].contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.q_norm(_linear_output(self.q_linear(hidden_states)))
        key, value = _linear_output(self.kv_linear(encoder_hidden_states)).chunk(2, dim=-1)
        local_inner = self.num_heads * self.head_dim
        key = self._slice_local_inner(key, local_inner)
        value = self._slice_local_inner(value, local_inner)
        key = self.k_norm(key)
        if self.softmax_attn is not None and attention_mask is None:
            query = self._reshape_to_seq_heads(query)
            key = self._reshape_to_seq_heads(key)
            value = self._reshape_to_seq_heads(value)
            attn = self.softmax_attn(query, key, value)
            return _linear_output(self.proj(attn.flatten(2, 3)))
        # vLLM Attention path does not currently take the SANA-WM prompt
        # padding mask; fall back to SDPA for exact NVlabs semantics when a
        # mask is supplied.
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    f"Sana-WM cross-attention mask must be shaped [B, text_len], got {tuple(attention_mask.shape)}."
                )
            attn_mask = (1 - attention_mask.to(query.dtype)) * -10000.0
            attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return _linear_output(self.proj(self._merge_heads(attn)))


class _ConvWrapper(nn.Module):
    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv


class SanaWmMbConvFfn(nn.Module):
    def __init__(self, config: SanaWmConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        expanded = int(hidden_size * config.mlp_ratio) * 2
        t_padding = config.t_kernel_size // 2
        self.glu_act = nn.SiLU()
        self.inverted_conv = _ConvWrapper(nn.Conv2d(hidden_size, expanded, kernel_size=1))
        self.depth_conv = _ConvWrapper(nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded))
        self.point_conv = _ConvWrapper(nn.Conv2d(expanded // 2, hidden_size, kernel_size=1, bias=False))
        self.t_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(config.t_kernel_size, 1),
            padding=(t_padding, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        """Match NVlabs ``GLUMBConvTemp.forward`` exactly:

        1. Reshape ``(B, N=F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)``
           so the spatial MBConv runs PER FRAME with proper 2D (H, W)
           geometry. Our previous reshape ``(B, C, F, H*W)`` made the
           3×3 ``depth_conv`` span the (F, H*W) plane treating F as
           height and the flattened H*W as a 1D width — wrong axis,
           producing garbage (cosine ≈ 0.04 vs NVlabs in the
           block-0 parity probe).
        2. Apply expand → depthwise 3×3 → GLU → contract on per-frame
           (B*F, C, H, W).
        3. Temporal aggregation: reshape back to ``(B, C, F, H*W)``
           and add ``t_conv`` (kernel ``(3, 1)``) along the F axis.
        4. Reshape to ``(B, N, C)``.
        """
        batch, _, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        # (B, F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)
        x = (
            hidden_states.reshape(batch * frames, spatial_tokens, hidden_size)
            .reshape(batch * frames, height, width, hidden_size)
            .permute(0, 3, 1, 2)
        )
        # NVlabs leaves this NHWC-derived NCHW view non-contiguous. On bf16
        # CUDA convs that selects a different kernel/layout than a contiguous
        # copy; preserving the stride is required for late-step parity.
        # NVlabs ``ConvLayer`` wraps Conv2d in `Conv → norm → act`. For SANA-WM
        # `act=("silu", "silu", None)`: inverted_conv applies SiLU, depth_conv
        # and point_conv don't. Our ``_ConvWrapper`` only stores the raw
        # Conv2d to match the checkpoint key (`inverted_conv.conv.weight`),
        # so we need to apply the SiLU explicitly here — skipping it left
        # the expanded features unactivated and inflated the downstream
        # GLU output magnitude by ~12×.
        x_inv = self.glu_act(self.inverted_conv.conv(x))
        x_dep = self.depth_conv.conv(x_inv)
        value, gate = x_dep.chunk(2, dim=1)
        x_glu = value * self.glu_act(gate)
        x_pt = self.point_conv.conv(x_glu)  # (B*F, hidden, H, W)
        x = x_pt
        # Temporal aggregation: (B*F, C, H, W) → (B, F, C, H*W) → (B, C, F, H*W)
        x_temporal = x.reshape(batch, frames, hidden_size, spatial_tokens).permute(0, 2, 1, 3)
        t_conv_out = self.t_conv(x_temporal)
        x_temporal = x_temporal + t_conv_out
        # → (B, F, H*W, C) → (B, N, C)
        final = x_temporal.permute(0, 2, 3, 1).reshape(batch, frames * spatial_tokens, hidden_size).contiguous()
        return final


class SanaWmBlock(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        block_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.block_idx = block_idx
        # Attach block_idx to attn/mlp so per-block probes can read it.
        # (Set after they're created below — see end of __init__.)
        hidden_size = config.hidden_size
        use_gdn = config.softmax_every_n <= 0 or (block_idx + 1) % config.softmax_every_n != 0
        use_plucker_proj = config.use_chunk_plucker_post_attn and (
            config.chunk_plucker_post_attn_blocks < 0 or block_idx < config.chunk_plucker_post_attn_blocks
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SanaWmSelfAttention(
            config,
            use_gdn=use_gdn,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.attn" if prefix else "attn",
        )
        self.cross_attn = SanaWmCrossAttention(
            config,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.cross_attn" if prefix else "cross_attn",
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SanaWmMbConvFfn(config)
        self.mlp.block_idx = block_idx
        self.attn.block_idx = block_idx
        if use_plucker_proj:
            if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
                self.plucker_proj: nn.Module | None = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    gather_output=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.plucker_proj" if prefix else "plucker_proj",
                    disable_tp=_disable_tp_for_vllm_layer(),
                )
            else:
                self.plucker_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.plucker_proj = None
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    @staticmethod
    def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return hidden_states * (1 + scale) + shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NVlabs dispatch convention: when timestep
        # modulation carries a frame axis (ndim > 2), route through the
        # frame-aware path so shift/scale/gate are applied per frame.
        if timestep_modulation.ndim > 2:
            return self._forward_frame_aware(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                camera_conditions,
                encoder_attention_mask,
            )
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        attn_input = self._modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.attn(attn_input, spatial_shape, rotary_emb, camera_conditions)
        if camera_hidden_states is not None and self.plucker_proj is not None:
            attn_output = attn_output + _linear_output(self.plucker_proj(camera_hidden_states))
        hidden_states = hidden_states + gate_msa * attn_output
        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        mlp_input = self._modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = hidden_states + gate_mlp * self.mlp(mlp_input, spatial_shape)
        return hidden_states

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        camera_hidden_states: torch.Tensor | None,
        camera_conditions: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-frame timestep modulation path matching NVlabs
        ``SanaVideoMSCamCtrlBlock.forward_frame_aware``.

        ``timestep_modulation`` is ``(B, 1, F, 6*D)``. We split into the
        six per-frame ``(B, F, 1, D)`` chunks via ``scale_shift_table``
        and apply ``shift/scale/gate`` per frame, broadcasting over the
        spatial tokens within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM frame-aware block expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        if timestep_modulation.shape[2] != frames:
            raise ValueError(
                "Sana-WM frame-aware block: timestep frame axis must match spatial frames, "
                f"got modulation frames={timestep_modulation.shape[2]} vs spatial frames={frames}."
            )

        # (B, 1, F, 6*D) -> (B, F, 6, D), then add the (1, 1, 6, D) table.
        t_per_frame = timestep_modulation.reshape(batch_size, frames, 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t_per_frame
        ).chunk(6, dim=-2)  # each: (B, F, 1, D)

        # Apply per-frame modulation by reshaping x to (B, F, S, D), broadcasting
        # scale/shift over the spatial axis, then flattening back.
        x_norm1 = self.norm1(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_msa_in = (x_norm1 * (1 + scale_msa) + shift_msa).reshape(batch_size, token_count, hidden_size)
        attn_output = self.attn(x_msa_in, spatial_shape, rotary_emb, camera_conditions)
        if camera_hidden_states is not None and self.plucker_proj is not None:
            attn_output = attn_output + _linear_output(self.plucker_proj(camera_hidden_states))
        attn_output_4d = attn_output.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_msa * attn_output_4d).reshape(batch_size, token_count, hidden_size)

        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        x_norm2 = self.norm2(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mlp_in = (x_norm2 * (1 + scale_mlp) + shift_mlp).reshape(batch_size, token_count, hidden_size)
        mlp_out = self.mlp(x_mlp_in, spatial_shape)
        mlp_out_4d = mlp_out.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_mlp * mlp_out_4d).reshape(batch_size, token_count, hidden_size)
        return hidden_states


class SanaWmFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: tuple[int, int, int],
        out_channels: int,
        *,
        quant_config: Any = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.out_channels = out_channels
        out_features = _prod(patch_size) * out_channels
        if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
            self.linear: nn.Module = ColumnParallelLinear(
                hidden_size,
                out_features,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear" if prefix else "linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.linear = nn.Linear(hidden_size, out_features)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        # Per-frame timestep contract dispatch.
        if timestep_embed.ndim > 2:
            return self._forward_frame_aware(hidden_states, timestep_embed, spatial_shape)
        shift, scale = (self.scale_shift_table[None] + timestep_embed[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale) + shift
        return _linear_output(self.linear(hidden_states))

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        """Per-frame final-layer modulation matching NVlabs
        ``T2IFinalLayer.forward_frame_aware``.

        ``timestep_embed`` is ``(B, 1, F, D)``. We transpose to
        ``(B, F, 1, D)`` so it adds correctly to the ``(1, 1, 2, D)``
        ``scale_shift_table`` and produces per-frame ``(B, F, 1, D)``
        ``shift`` / ``scale`` that broadcast over the spatial tokens
        within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames = timestep_embed.shape[2]
        if spatial_shape is not None:
            spatial_tokens = spatial_shape[1] * spatial_shape[2]
        else:
            spatial_tokens = token_count // frames
        if frames * spatial_tokens != token_count:
            raise ValueError(
                "Sana-WM frame-aware final layer: token count mismatch "
                f"(N={token_count}, F={frames}, S={spatial_tokens})."
            )
        # (B, 1, F, D) -> (B, F, 1, D); add (1, 1, 2, D); chunk into shift/scale: each (B, F, 1, D).
        t_per_frame = timestep_embed.transpose(1, 2)
        shift, scale = (self.scale_shift_table[None, None, :, :] + t_per_frame).chunk(2, dim=-2)
        x_norm = self.norm_final(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mod = (x_norm * (1 + scale) + shift).reshape(batch_size, token_count, hidden_size)
        return _linear_output(self.linear(x_mod))


class SanaWmTransformer3DModel(nn.Module):
    """SANA-WM Stage-1 DiT with a runnable pure-PyTorch GDN fallback path."""

    _repeated_blocks: ClassVar[list[str]] = ["SanaWmBlock"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "blocks"
    _hsdp_shard_conditions: ClassVar[list[Any]] = [_is_sana_wm_transformer_block]
    _sp_plan: ClassVar[dict[str, Any] | None] = _build_sana_wm_sp_plan()

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int = SANA_WM_STAGE1_LATENT_CHANNELS,
        prompt_channels: int = SANA_WM_STAGE1_PROMPT_CHANNELS,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config or SanaWmConfig()
        self.quant_config = quant_config
        self.prefix = prefix
        self.use_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        self._latent_channels = latent_channels
        self._prompt_channels = prompt_channels
        self.patch_size = _to_3tuple(self.config.patch_size)
        self.x_embedder = SanaWmPatchEmbedMS3D(self.patch_size, self._latent_channels, self.config.hidden_size)
        self.y_embedder = SanaWmTextEmbedder(
            self._prompt_channels,
            self.config.hidden_size,
            self.config.model_max_length,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.y_embedder" if self.prefix else "y_embedder",
        )
        self.t_embedder = SanaWmTimestepEmbedder(
            SANA_WM_STAGE1_TIMESTEP_CHANNELS,
            self.config.hidden_size,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.t_embedder" if self.prefix else "t_embedder",
        )
        # t_block: SiLU → Linear(hidden, 6*hidden).  Weight key is t_block.1.{weight,bias}
        # to match the Stage-1 checkpoint layout (nn.Sequential index 1).
        if self.use_vllm_parallel_layers and ColumnParallelLinear is not None:
            _t_block_linear: nn.Module = ColumnParallelLinear(
                self.config.hidden_size,
                6 * self.config.hidden_size,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.t_block.1" if self.prefix else "t_block.1",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            _t_block_linear = nn.Linear(self.config.hidden_size, 6 * self.config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), _t_block_linear)
        self.plucker_embedder = SanaWmPatchEmbedMS3D(
            self.patch_size,
            self.config.chunk_plucker_channels,
            self.config.hidden_size,
        )
        self.raymap_embedder = SanaWmPatchEmbedMS3D(self.patch_size, 3, self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SanaWmBlock(
                    self.config,
                    block_idx=i,
                    quant_config=self.quant_config,
                    use_vllm_parallel_layers=self.use_vllm_parallel_layers,
                    prefix=f"{self.prefix}.blocks.{i}" if self.prefix else f"blocks.{i}",
                )
                for i in range(self.config.num_blocks)
            ]
        )
        self.final_layer = SanaWmFinalLayer(
            self.config.hidden_size,
            self.patch_size,
            self._latent_channels,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.final_layer" if self.prefix else "final_layer",
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 484, self.config.hidden_size))
        self.rope = SanaWmWanRotaryPosEmbed(self.config.linear_head_dim)
        self.attention_y_norm = _maybe_make_vllm_rms_norm(self.config.hidden_size)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @staticmethod
    def _local_name(remapped_name: str) -> str:
        return remapped_name.removeprefix("transformer.")

    @staticmethod
    def _maybe_shard_loaded_tensor(
        remapped_name: str,
        tensor: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Shard simple non-vLLM parameters that become TP-local.

        vLLM parallel linear parameters have their own `weight_loader`.
        SANA-WM also has model-local GDN vectors and depthwise temporal
        convolution weights (`A_log`, `dt_bias`, `conv_k`) whose checkpoint
        tensors are full-sized. Narrow those along the single changed dimension
        when TP makes the materialized parameter local.
        """

        if tensor.ndim != target.ndim:
            return tensor
        differing_dims = [
            dim
            for dim, (target_size, loaded_size) in enumerate(zip(target.shape, tensor.shape, strict=True))
            if target_size != loaded_size
        ]
        if len(differing_dims) != 1:
            return tensor
        dim = differing_dims[0]
        target_size = target.shape[dim]
        loaded_size = tensor.shape[dim]
        tp_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()
        if tp_size <= 1 or target_size * tp_size != loaded_size:
            return tensor
        if not any(
            token in remapped_name
            for token in (
                ".A_log",
                ".dt_bias",
                ".conv_k.",
                ".conv_k_cam.",
                ".conv_q_cam.",
                ".conv_v_cam.",
                ".q_norm.",
                ".k_norm.",
                ".q_norm_cam.",
                ".k_norm_cam.",
            )
        ):
            return tensor
        start = tp_rank * target_size
        return tensor.narrow(dim, start, target_size)

    def _positional_embedding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        if pos_embed.shape[1] == token_count:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=token_count, mode="linear", align_corners=False)
        return pos_embed.transpose(1, 2)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def _camera_hidden_states_from_conditions(
        self,
        *,
        plucker: torch.Tensor | None,
        spatial_raymap: torch.Tensor | None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if plucker is None:
            return None
        if plucker.ndim == 4:
            plucker = plucker.unsqueeze(0)
        plucker = plucker.to(device=device, dtype=dtype)
        if plucker.shape[0] == 1 and batch_size > 1:
            plucker = plucker.expand(batch_size, -1, -1, -1, -1)
        camera_hidden_states = self.plucker_embedder(plucker)
        expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        camera_hidden_states = self._match_tokens(camera_hidden_states, expected_tokens)

        # Fuse per-pixel ray-direction map via raymap_embedder only on the
        # non-plucker path. NVlabs skips the absmap/raymap embedder whenever
        # chunk-plucker post-attention injection is enabled.
        if spatial_raymap is not None and not self.config.use_chunk_plucker_post_attn:
            if spatial_raymap.ndim == 4:  # [C, F, H, W] → [1, C, F, H, W]
                spatial_raymap = spatial_raymap.unsqueeze(0)
            spatial_raymap = spatial_raymap.to(device=device, dtype=dtype)
            if spatial_raymap.shape[0] == 1 and batch_size > 1:
                spatial_raymap = spatial_raymap.expand(batch_size, -1, -1, -1, -1)
            ray_hidden = self._match_tokens(self.raymap_embedder(spatial_raymap), expected_tokens)
            camera_hidden_states = camera_hidden_states + ray_hidden

        return camera_hidden_states

    def _unpatchify(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        frames, height, width = spatial_shape
        patch_frames, patch_height, patch_width = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self._latent_channels,
        )
        hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
        return hidden_states.reshape(
            batch_size,
            self._latent_channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        camera_encoder: SanaWmCameraEmbedder | None = None,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_raymap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError("Sana-WM transformer expects latent input shaped [B, C, F, H, W].")
        # Env-gated force-fp32 transformer forward for the
        # bf16-precision-drift upper-bound experiment. Casts inputs to fp32,
        # also upcasts model params (one-shot), and casts noise_pred back to
        # the original input dtype before returning.
        _force_fp32 = os.environ.get("SANA_WM_FORCE_FP32_TRANSFORMER", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        _input_dtype = hidden_states.dtype
        if _force_fp32:
            if not getattr(self, "_fp32_upcasted", False):
                self.to(torch.float32)
                self._fp32_upcasted = True
            hidden_states = hidden_states.float()
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.float()
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device=hidden_states.device)
            if camera_hidden_states is not None:
                camera_hidden_states = camera_hidden_states.float()
            if plucker is not None:
                plucker = plucker.float()
            if raymap is not None:
                raymap = raymap.float()
            if spatial_raymap is not None:
                spatial_raymap = spatial_raymap.float()

        batch_size = hidden_states.shape[0]
        latent_shape = hidden_states.shape[2:]
        hidden_states, spatial_shape = self.x_embedder.project_with_shape(hidden_states)
        rotary_emb = None
        if self.config.pos_embed_type == "wan_rope":
            rotary_emb = self.rope(spatial_shape, hidden_states.device)
        else:
            hidden_states = hidden_states + self._positional_embedding(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device
            )

        encoder_hidden_states = self.y_embedder(
            encoder_hidden_states.to(device=hidden_states.device) if encoder_hidden_states is not None else None,
            batch_size=batch_size,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.attention_y_norm(encoder_hidden_states)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            encoder_attention_mask = encoder_attention_mask.to(device=hidden_states.device)
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError(
                    "Sana-WM encoder_attention_mask must match text tokens "
                    f"{tuple(encoder_hidden_states.shape[:2])}, got {tuple(encoder_attention_mask.shape)}."
                )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=torch.float32)
        timestep = timestep.to(device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        # Per-frame timestep contract: when ``timestep`` is
        # rank > 1 (e.g. ``(B, 1, F)`` from the LTX flow-matching sampler),
        # the per-frame embedding is computed on the flattened token axis
        # and unflattened back to the input rank. ``time_embed`` then has
        # an extra ``hidden_size`` axis appended (``(B, 1, F, D)``) and
        # ``timestep_modulation`` has ``6*hidden_size`` (``(B, 1, F, 6*D)``).
        # SanaWmBlock / SanaWmFinalLayer dispatch on ``ndim > 2`` to the
        # frame-aware paths that broadcast per-frame modulation over the
        # spatial tokens within each frame.
        timestep_shape = tuple(timestep.shape)
        time_embed = self.t_embedder(timestep)  # (numel, D)
        # t_block is Sequential(SiLU, Linear|ColumnParallelLinear); index explicitly
        # so _linear_output can unwrap the (tensor, None) tuple from parallel layers.
        _t_silu = self.t_block[0](time_embed)
        timestep_modulation = _linear_output(self.t_block[1](_t_silu)).to(hidden_states.dtype)
        if len(timestep_shape) > 1:
            time_embed = time_embed.unflatten(0, timestep_shape)
            timestep_modulation = timestep_modulation.unflatten(0, timestep_shape)

        if camera_hidden_states is None:
            camera_hidden_states = self._camera_hidden_states_from_conditions(
                plucker=plucker,
                spatial_raymap=spatial_raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if camera_hidden_states is None and camera_encoder is not None:
            camera_hidden_states = camera_encoder(
                plucker=plucker,
                raymap=raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Build (B, T_latent, 20) camera_conditions from the raw raymap so
        # the per-block UCPE branch can compute its per-pixel apply_fns.
        # When the caller passes pre-embedded camera_hidden_states without
        # raymap, the cam branch becomes a no-op (main-branch only).
        camera_conditions: torch.Tensor | None = None
        if raymap is not None:
            camera_conditions = raymap if raymap.ndim == 3 else raymap.unsqueeze(0)
            if camera_conditions.shape[0] == 1 and batch_size > 1:
                camera_conditions = camera_conditions.expand(batch_size, -1, -1)
            camera_conditions = camera_conditions.to(device=hidden_states.device, dtype=hidden_states.dtype)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                camera_conditions,
                encoder_attention_mask,
            )

        hidden_states = self.final_layer(hidden_states, time_embed.to(hidden_states.dtype), spatial_shape)
        hidden_states = self._unpatchify(hidden_states, spatial_shape)
        if hidden_states.shape[2:] != latent_shape:
            hidden_states = F.interpolate(hidden_states, size=latent_shape, mode="trilinear", align_corners=False)
        if _force_fp32 and hidden_states.dtype != _input_dtype:
            hidden_states = hidden_states.to(_input_dtype)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        """Stream checkpoint tensors into the eagerly-built modules.

        Follows the wan2_2 idiom: one ``named_parameters`` lookup table, the
        per-tensor ``weight_loader`` for vLLM parallel layers (which narrows
        full checkpoint tensors to the TP-local shard at copy time), and a
        manual narrow for the model-local GDN/conv vectors. The source tensor
        is dropped each iteration — no copy of the checkpoint is retained.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()
        unmapped: list[str] = []
        duplicates: list[str] = []

        for source_name, tensor in weights:
            remapped_name = normalize_sana_wm_stage1_weight_name(source_name)
            if remapped_name is None:
                unmapped.append(source_name)
                continue
            if remapped_name in loaded:
                duplicates.append(remapped_name)
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Sana-WM weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
            local_name = self._local_name(remapped_name)
            target = params_dict.get(local_name)
            if target is None:
                target = buffers_dict.get(local_name)
            if target is None:
                raise ValueError(f"Sana-WM weight {remapped_name} was remapped but not consumed by the Stage-1 model.")
            weight_loader = getattr(target, "weight_loader", None)
            if callable(weight_loader):
                weight_loader(target, tensor)
            else:
                if tuple(target.shape) != tuple(tensor.shape):
                    tensor = self._maybe_shard_loaded_tensor(remapped_name, tensor, target)
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Sana-WM weight shape mismatch for {remapped_name}: "
                        f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}."
                    )
                with torch.no_grad():
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(remapped_name)

        if unmapped or duplicates:
            details = []
            if unmapped:
                details.append(f"unmapped={unmapped[:10]}")
            if duplicates:
                details.append(f"duplicates={duplicates[:10]}")
            raise ValueError("Invalid SANA-WM Stage-1 checkpoint keys: " + "; ".join(details))
        return loaded
