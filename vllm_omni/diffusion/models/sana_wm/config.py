# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config parser for the SANA-WM root ``config.yaml`` release file."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_omni.config.yaml_util import load_yaml_config, to_dict

_ARCH_BLOCKS_RE = re.compile(r"(?:^|_)D(?P<num_blocks>\d+)(?:_|$)")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_tuple3(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, list | tuple):
        if len(value) == 2:
            return (1, int(value[0]), int(value[1]))
        if len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
    return default


def _parse_num_blocks(architecture_name: str | None, default: int) -> int:
    if not architecture_name:
        return default
    match = _ARCH_BLOCKS_RE.search(architecture_name)
    return int(match.group("num_blocks")) if match else default


@dataclass(frozen=True)
class SanaWmConfig:
    """SANA-WM Stage-1 architecture and runtime config.

    Defaults mirror the first public HF release. ``from_yaml`` should be the
    normal construction path so future HF config updates flow through here
    instead of being re-hardcoded in the pipeline.
    """

    architecture_name: str | None = "SanaMSVideoCamCtrl_1600M_P1_D20"
    num_blocks: int = 20
    hidden_size: int = 2240
    mlp_ratio: float = 3.0
    attn_type: str = "BidirectionalGDNTriton"
    softmax_every_n: int = 4
    linear_head_dim: int = 112
    conv_kernel_size: int = 4
    t_kernel_size: int = 3
    k_conv_only: bool = True
    ffn_type: str = "GLUMBConvTemp"
    pos_embed_type: str = "wan_rope"
    patch_size: tuple[int, int, int] = (1, 1, 1)
    qk_norm: bool = True
    cross_norm: bool = True
    mixed_precision: str = "bf16"
    fp32_attention: bool = True
    image_size: int = 720
    cam_attn_compress: int = 1
    use_chunk_plucker_post_attn: bool = True
    chunk_plucker_channels: int = 48
    chunk_plucker_post_attn_blocks: int = 20
    inference_flow_shift: float = 9.8
    scheduler_type: str = "flow_dpm-solver"
    chi_prompt: list[str] = field(default_factory=list)
    y_norm_scale_factor: float = 0.01
    model_max_length: int = 300

    @classmethod
    def from_yaml(cls, path: str | Path) -> SanaWmConfig:
        raw = load_yaml_config(path)
        data = to_dict(raw) if raw is not None else {}
        return cls.from_dict(data if isinstance(data, Mapping) else {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SanaWmConfig:
        root = dict(data)
        model_value = root.get("model")
        model_cfg = _as_mapping(model_value)
        architecture_name = (
            model_value
            if isinstance(model_value, str)
            else _first_present(model_cfg.get("name"), model_cfg.get("type"), model_cfg.get("_class_name"))
        )
        scheduler_cfg = _as_mapping(root.get("scheduler"))
        text_encoder_cfg = _as_mapping(root.get("text_encoder"))

        def model_field(name: str, default: Any = None) -> Any:
            return _first_present(model_cfg.get(name), root.get(name), default)

        num_blocks_default = _parse_num_blocks(architecture_name, cls.num_blocks)
        return cls(
            architecture_name=architecture_name,
            num_blocks=int(model_field("num_blocks", model_field("num_layers", num_blocks_default))),
            hidden_size=int(model_field("hidden_size", cls.hidden_size)),
            mlp_ratio=float(model_field("mlp_ratio", cls.mlp_ratio)),
            attn_type=str(model_field("attn_type", cls.attn_type)),
            softmax_every_n=int(model_field("softmax_every_n", cls.softmax_every_n)),
            linear_head_dim=int(model_field("linear_head_dim", cls.linear_head_dim)),
            conv_kernel_size=int(model_field("conv_kernel_size", cls.conv_kernel_size)),
            t_kernel_size=int(model_field("t_kernel_size", cls.t_kernel_size)),
            k_conv_only=_as_bool(model_field("k_conv_only", cls.k_conv_only)),
            ffn_type=str(model_field("ffn_type", cls.ffn_type)),
            pos_embed_type=str(model_field("pos_embed_type", cls.pos_embed_type)),
            patch_size=_as_tuple3(model_field("patch_size", cls.patch_size), cls.patch_size),
            qk_norm=_as_bool(model_field("qk_norm", cls.qk_norm)),
            cross_norm=_as_bool(model_field("cross_norm", cls.cross_norm)),
            mixed_precision=str(model_field("mixed_precision", cls.mixed_precision)),
            fp32_attention=_as_bool(model_field("fp32_attention", cls.fp32_attention)),
            image_size=int(model_field("image_size", cls.image_size)),
            cam_attn_compress=int(model_field("cam_attn_compress", cls.cam_attn_compress)),
            use_chunk_plucker_post_attn=_as_bool(
                model_field("use_chunk_plucker_post_attn", cls.use_chunk_plucker_post_attn)
            ),
            chunk_plucker_channels=int(model_field("chunk_plucker_channels", cls.chunk_plucker_channels)),
            chunk_plucker_post_attn_blocks=int(
                model_field("chunk_plucker_post_attn_blocks", cls.chunk_plucker_post_attn_blocks)
            ),
            inference_flow_shift=float(
                _first_present(
                    scheduler_cfg.get("inference_flow_shift"),
                    root.get("inference_flow_shift"),
                    cls.inference_flow_shift,
                )
            ),
            scheduler_type=str(
                _first_present(scheduler_cfg.get("vis_sampler"), scheduler_cfg.get("type"), cls.scheduler_type)
            ),
            chi_prompt=_as_list_of_str(text_encoder_cfg.get("chi_prompt")),
            y_norm_scale_factor=float(text_encoder_cfg.get("y_norm_scale_factor", cls.y_norm_scale_factor)),
            model_max_length=int(text_encoder_cfg.get("model_max_length", cls.model_max_length)),
        )
