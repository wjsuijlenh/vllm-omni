# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-local Bidirectional Gated DeltaNet (GDN) kernels for SANA-WM.

Merged from three previously separate modules (fused_gdn, fused_gdn_chunkwise,
gated_deltanet_triton). Ported from the Apache-2.0 NVlabs/Sana reference and
kept model-local because SANA-WM's bidirectional video-latent recurrence is a
different contract than vLLM's autoregressive Qwen3-Next GDN cache path.

Contents:
  1. Config / launch helpers (``_resolve_launch_config`` & friends) plus the
     fused Q+K inverse-RMS and RoPE-table kernels.
  2. Chunkwise-parallel forward kernels (phase A/B/C, bidirectional and camera
     scan variants) implementing the recurrence.
  3. High-level GDN entry points: the Triton path
     (``triton_bidirectional_gated_delta_net_from_qkv``) and the pure-PyTorch
     reference fallback (``reference_bidirectional_gated_delta_net``).

Precision knob: env ``FUSED_GDN_PRECISION`` / ``PRECISION_OVERRIDE``:
  0=IEEE fp32 dots, 1=TF32, 2=bf16 TC + fp32 state [default], 3=bf16 TC + bf16 state.
"""

# ruff: noqa: E501

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import nn

# =====================================================================
#  GPU-adaptive kernel config
# =====================================================================


def _get_kernel_config() -> dict:
    """Return optimal kernel parameters for the current GPU.

    STATE_FP32: use fp32 state_prev when SRAM is large enough.
      - bf16 state_prev: ~96KB total SRAM (fits GB10's 101KB).
      - fp32 state_prev: ~128KB total SRAM (needs H100's 228KB+).
    """
    smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    state_fp32 = smem >= 150 * 1024  # H100 (228KB) yes, GB10 (101KB) no
    return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 8, "STATE_FP32": state_fp32}


_KCFG = None


def _kcfg():
    global _KCFG
    if _KCFG is None:
        _KCFG = _get_kernel_config()
    return _KCFG


# precision=0 → IEEE fp32 dots + fp32 state  (DOT_PRECISION=2, STATE_FP32=1)
# precision=1 → TF32  dots   + fp32 state    (DOT_PRECISION=1, STATE_FP32=1)
# precision=2 → bf16  dots   + fp32 state    (DOT_PRECISION=0, STATE_FP32=1) [default]
# precision=3 → bf16  dots   + bf16 state    (DOT_PRECISION=0, STATE_FP32=0)
def _precision_params(precision: int) -> tuple:
    if precision == 0:
        return 2, True
    elif precision == 1:
        return 1, True
    elif precision == 3:
        return 0, False
    else:  # default
        return 0, True


_env_prec = os.environ.get("FUSED_GDN_PRECISION", None)
PRECISION_OVERRIDE: int | None = int(_env_prec) if _env_prec is not None else None


def _resolve_launch_config() -> tuple:
    """Returns (prec, dot_prec, state_fp32, num_warps).

    Uses ``PRECISION_OVERRIDE`` when set; otherwise falls back to ``_kcfg()``
    (which picks ``STATE_FP32`` based on per-GPU SRAM). ``num_warps`` is
    clamped to 4 when dots run on fp32 operands (more registers needed).
    """
    cfg = _kcfg()
    prec = PRECISION_OVERRIDE if PRECISION_OVERRIDE is not None else 2
    dot_prec, state_fp32 = _precision_params(prec)
    if PRECISION_OVERRIDE is None:
        state_fp32 = cfg["STATE_FP32"]
    nw = cfg["num_warps"]
    if dot_prec >= 1:
        nw = min(nw, 4)
    return prec, dot_prec, state_fp32, nw


def prepare_rope_tables(rotary_emb, N: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex rotary_emb `(1, 1, N, D//2)` → expanded (N, D) cos/sin tables.

    Encodes the interleaved-pair rotation
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    as  y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]
    where sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].

    Returns (cos_exp, sin_exp) both (N, D) float32, contiguous.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


# =====================================================================
#  Fused single-pass Q+K inverse-RMS Triton kernel
# =====================================================================
# Single Triton launch that reads each `(b, n)` row of `qkv` once and emits
# both `q_inv_rms[b, n]` and `k_inv_rms[b, n]`. Replaces two separate PyTorch
# scans (cast→square→sum→rsqrt) over `qkv[:, :, 0]` and `qkv[:, :, 1]`.
#
# Layout assumed: `qkv` is (B, N, 3, H, D) contiguous, so the C = H*D channels
# for a given (b, n, qkv_idx) live in a contiguous memory span.


@triton.jit
def _fused_qk_inv_rms_kernel(
    qkv_ptr,  # *T_in     (B, N, 3, H, D), contiguous
    q_inv_rms_ptr,  # *float32  (B, N)
    k_inv_rms_ptr,  # *float32  (B, N)
    N: tl.constexpr,
    C: tl.constexpr,  # H * D
    eps,
    BLOCK_C: tl.constexpr,
):
    bn_id = tl.program_id(0)
    qkv_row_stride = 3 * C
    row_base = bn_id * qkv_row_stride
    q_base = row_base
    k_base = row_base + C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    q_vals = tl.load(qkv_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(qkv_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    q_sq = tl.sum(q_vals * q_vals, axis=0)
    k_sq = tl.sum(k_vals * k_vals, axis=0)

    inv_c = 1.0 / C
    q_inv = tl.rsqrt(q_sq * inv_c + eps)
    k_inv = tl.rsqrt(k_sq * inv_c + eps)

    tl.store(q_inv_rms_ptr + bn_id, q_inv)
    tl.store(k_inv_rms_ptr + bn_id, k_inv)


def fused_qk_inv_rms(
    qkv: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Triton fused Q+K inverse-RMS.

    Replaces ``(_precompute_inv_rms(qkv, 0, C, eps), _precompute_inv_rms(qkv, 1, C, eps))``
    with one launch that reads each ``(b, n)`` row of ``qkv`` exactly once.

    Args:
      qkv: (B, N, 3, H, D) contiguous tensor, any fp dtype.
      eps: RMSNorm epsilon.

    Returns:
      (q_inv_rms, k_inv_rms), each (B, N) float32 contiguous.
    """
    assert qkv.is_contiguous(), "qkv must be contiguous (B, N, 3, H, D)"
    assert qkv.dim() == 5 and qkv.shape[2] == 3, f"expected (B, N, 3, H, D), got {tuple(qkv.shape)}"
    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    k_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_qk_inv_rms_kernel[(B * N,)](
        qkv,
        q_inv_rms,
        k_inv_rms,
        N=N,
        C=C,
        eps=eps,
        BLOCK_C=BLOCK_C,
    )
    return q_inv_rms, k_inv_rms


# =====================================================================
#  Bidirectional GDN entry point (delegates to chunkwise)
# =====================================================================


def fused_bigdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) float32
    k_inv_rms: torch.Tensor,  # (B, N) float32
    q_norm_weight: torch.Tensor,  # (C,) float32
    k_norm_weight: torch.Tensor,  # (C,) float32
    rope_cos: torch.Tensor,  # (N, D) float32
    rope_sin: torch.Tensor,  # (N, D) float32
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional fused GDN. Returns ``(B, N, H, D)``.

    Thin entry point kept for call-site stability; delegates to
    :func:`fused_bigdn_bidi_chunkwise` from ``fused_gdn_chunkwise``.
    """

    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
    )


# Identity RMS/RoPE tables are pure shape-keyed constants (all ones/zeros), so
# they are cached to avoid re-allocating them on every forward. The key varies
# with batch size B and token count N, which are request-dependent in serving,
# so the cache is bounded (LRU) to prevent unbounded GPU-memory growth.
_CAM_IDENTITY_CACHE_MAXSIZE = 32

# ════════════════════════════════════════════════════════════════
#  Per-architecture launch config (auto-selected via compute capability)
# ════════════════════════════════════════════════════════════════
#
# Empirically tuned at production config (B=1..8, T=11, S=920, H=20, D=112) on
# A100 / H100 / GB200. Two effects matter:
#
# 1. **Precision sets BLOCK_S**: fp32 operand fragments are 2× the size of
#    bf16. BLOCK_S=64 + fp32 → register spills (catastrophic, 40-100× slower).
#    BLOCK_S=32 + fp32 → no spills. So fp32 mode forces BLOCK_S=32 everywhere.
#
# 2. **Arch sets BLOCK_S for bf16**: A100 (192 KB SRAM, fewer registers per
#    block) prefers BLOCK_S=32 even at bf16. H100/GB200 (228 KB SRAM) tolerate
#    BLOCK_S=64 cleanly at bf16.
#
# Each entry: (phase_a_warps, phase_a_BLOCK_S,
#              phase_b_warps, phase_b_stages,
#              phase_c_warps, phase_c_BLOCK_S, phase_c_stages)

# ── Launch-config tuning table ─────────────────────────────────────
#
# We tune 8 knobs across 3 phases:
#   Phase A : (nw, BS)               streaming accumulator in registers
#   Phase B : (nw, use_acc, ns)      serial-F scan with persistent M in regs
#   Phase C : (nw, BS, ns)           streams Pass-2 output; loads fp32 M[128,128]
#
# Each arch × precision combination gets a named entry below. Values come from
# empirical sweeps (see commit log: T6 A100/H100 sweep 2026-04-19; Blackwell-DC
# 2026-04-20; Spark GB10 tuning notes in commits 5da52db6 / 3ad104d0) and from
# kernel-structure analysis (Phase B's persistent M[128,128] fp32 is 64 KB → nw
# controls register spread; Phase C's loaded M[128,128] is 64 KB → BS controls
# transient SMEM footprint).
#
# Adding a new arch: pick the closest existing bucket in _CHUNKWISE_TUNING.


@dataclass(frozen=True)
class _PhaseCfg:
    nw: int  # num_warps
    BS: int = 0  # BLOCK_S (Phase A/C only; 0 = N/A for Phase B)
    ns: int = 1  # num_stages
    use_acc: bool = False  # Phase B only: fold A_f via MMA accumulator


@dataclass(frozen=True)
class _ChunkwiseCfg:
    A: _PhaseCfg
    B: _PhaseCfg
    C: _PhaseCfg

    def as_tuple(self) -> tuple:
        """Flatten to the 8-tuple the legacy API returns."""
        return (
            self.A.nw,
            self.A.BS,
            self.B.nw,
            self.B.ns,
            self.B.use_acc,
            self.C.nw,
            self.C.BS,
            self.C.ns,
        )


# ──────────────────────────────────────────────────────────────────
# Primary tuning table: (arch_key, prec_key) → _ChunkwiseCfg.
# Arch keys:
#   "ampere"          sm_80     A100 (164 KB SRAM, no WGMMA)
#   "hopper"          sm_90     H100 (228 KB SRAM, WGMMA)
#   "blackwell_dc"    sm_100    B200 / GB200 (228 KB SRAM, WGMMA v2)
#   "blackwell_spark" sm_120+ with < 150 KB SRAM  5090 / GB10 (~102 KB SRAM)
# Prec keys:
#   "bf16"  dot_prec == 0  (bf16 TC, half-size operand fragments)
#   "fp32"  dot_prec >= 1  (TF32 TC or IEEE Markidis 3-pass; same launch shape)
# ──────────────────────────────────────────────────────────────────
_CHUNKWISE_TUNING: dict[tuple[str, str], _ChunkwiseCfg] = {
    # A100: smaller SRAM than Hopper, no WGMMA → bigger CTAs hide MMA latency.
    # Phase B fp32 needs nw=32 to spread persistent M across warps (no acc-fusion
    # available pre-Hopper, so ns=2 fills the MMA pipeline slot instead).
    ("ampere", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=4, BS=32, ns=1),  # nw=4 bf16 C: 27% faster than nw=8 per T6
    ),
    ("ampere", "fp32"): _ChunkwiseCfg(
        # 2026-04-30 PM retune: Phase A nw=8 → 16 BS=32 yields 8-13× speedup
        # across F ∈ {3, 5, 11, 14, 17, 20} (cos=1.0 verified). Old nw=8 was a
        # legacy default never re-swept; sweep showed nw=16 dominates every F.
        # Closes A100 sink/rolling chunkwise regression where Phase B was
        # already optimal (sub-percent tuning gap) — Phase A was the bottleneck.
        A=_PhaseCfg(nw=16, BS=32),
        B=_PhaseCfg(nw=32, use_acc=False, ns=2),  # ns=2 fills pipe (no acc-fusion)
        C=_PhaseCfg(nw=16, BS=32, ns=1),  # 2026-04-30 retune: nw=16 BS=32 is 2.8x faster (was nw=8 BS=16)
    ),
    # Hopper (H100): WGMMA + 228 KB SRAM → big tiles win at bf16.
    # Phase B fp32 uses acc-fusion (MMA accumulator folds A_f in one op, +12%).
    ("hopper", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),  # small CTAs pack better on WGMMA
        C=_PhaseCfg(nw=8, BS=32, ns=1),
    ),
    ("hopper", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),  # fp32 operand 2× bigger → half BS
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=1
        ),  # 2026-04-29 retune: acc_fusion=False is 3x faster post precision-gate fix
        C=_PhaseCfg(nw=16, BS=32, ns=1),  # 2026-04-30 retune: nw=16 BS=32 is 1.7x faster (was nw=8 BS=16)
    ),
    # Blackwell-DC (B200 / GB200): 228 KB SRAM + improved WGMMA codegen.
    # bf16 likes small CTAs (nw=4); fp32 stays at nw=8 (nw=4 + BS=64 fp32 = 92× regression).
    ("blackwell_dc", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=4, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=64, ns=1),  # 228 KB SRAM leaves room for BS=64 bf16
    ),
    ("blackwell_dc", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(
            nw=8, BS=128
        ),  # 2026-04-30 retune: nw=8 BS=128 ~5% faster at production F=3-6 (sweep across F=3,5,6,11)
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=3
        ),  # 2026-04-29 retune: 14x faster (was nw=8 acc=True 17ms; now nw=32 ns=3 acc=False 1.23ms)
        C=_PhaseCfg(
            nw=4, BS=64, ns=1
        ),  # 2026-04-30 retune: nw=4 BS=64 is 3-5x faster than old nw=8 BS=16 (sweep 2026-04-30)
    ),
    # Blackwell-Spark (5090 / GB10, ~102 KB SRAM): shares SRAM penalty of small
    # chips but not Blackwell-DC's WGMMA-v2 register-spread benefit. Empirically
    # behaves like Hopper at fp32 (Phase B wants nw=32 to spread persistent M
    # across warps, not nw=8 like DC). BS shrunk one step vs DC; Phase A bf16
    # wants nw=8 (nw=4 tested 22× slower per 2026-04-20 sweep).
    # Sweep 2026-04-24 (prod dim F=11 S=920): Phase B nw=32 gives 1.84×/2.65×
    # (GB10/5090) at fp32 over prior nw=8 setting.
    ("blackwell_spark", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),  # nw=8 (not 4) at bf16: ~5% across F=3,6,11
        # 2026-05-06 P1/P2 retune (5090, F=11 S=920): C.nw=4 BS=32 is ~3.5%
        # faster than nw=8 (Phase C is bandwidth-bound, fewer warps schedules
        # better on the small SRAM). BS=64 bf16 on Spark OOMs SRAM.
        C=_PhaseCfg(nw=4, BS=32, ns=1),
    ),
    ("blackwell_spark", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=16),  # fp32 operand 2× bigger → BS=16 (half of DC's 32)
        # 2026-05-06 retune: nw=16 OOMs the 102 KB SRAM cap at TF32 on 5090
        # (131 KB needed). nw=8 fits and is within noise of the prior nw=16
        # benchmark. The Phase B D-tile path (auto-enabled on spark, see
        # `_pick_phase_b_d_splits`) is ~2.6× faster than this baseline at TF32
        # and ~13% faster at IEEE — these baseline params only apply when
        # PHASE_B_D_SPLITS=1 is forced.
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=16, ns=1),  # binding constraint: M.fp32 64 KB + Q stage
    ),
}


def _arch_key(cap: tuple) -> str:
    """Map compute capability → named arch bucket in `_CHUNKWISE_TUNING`.

    Blackwell (cap[0] >= 10) is split into "blackwell_dc" and "blackwell_spark"
    by SRAM size (≥150 KB vs less). Without CUDA or for unknown archs we
    default to the conservative "ampere" bucket.
    """
    if cap[0] == 8:
        return "ampere"
    if cap[0] == 9:
        return "hopper"
    if cap[0] >= 10:
        has_big_sram = True
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
            has_big_sram = smem >= 150 * 1024
        return "blackwell_dc" if has_big_sram else "blackwell_spark"
    return "ampere"


def _prec_key(dot_prec: int) -> str:
    return "fp32" if dot_prec >= 1 else "bf16"


def _auto_config(dot_prec: int, cap: tuple) -> tuple:
    """Look up chunkwise kernel launch params from the tuning table.

    Resolution order:
      1. `_CHUNKWISE_TUNING[(arch, prec)]` — primary per-(arch, prec) table.
      2. Fallback to ("ampere", prec) if the arch is unrecognised.

    Returns the legacy 8-tuple `(a_nw, a_BS, b_nw, b_ns, b_use_acc, c_nw, c_BS, c_ns)`
    for backward compatibility with `_get_arch_config` callers.
    """
    prec = _prec_key(dot_prec)
    cfg = _CHUNKWISE_TUNING.get((_arch_key(cap), prec)) or _CHUNKWISE_TUNING[("ampere", prec)]
    return cfg.as_tuple()


def _get_arch_config(
    dot_precision: int = 0,
    device: torch.device | int | None = None,
):
    """Returns (a_warps, a_BLOCK_S, b_warps, b_stages, b_use_acc_fusion,
                c_warps, c_BLOCK_S, c_stages).

    dot_precision: 0=bf16 TC, 1=TF32 TC, 2=IEEE fp32.
    device:        device whose capability drives the lookup. Defaults to the
                   current CUDA device — pass ``qkv.device`` (or any input
                   tensor's device) when launching kernels in heterogeneous
                   or multi-GPU single-process setups so the right tuning
                   bucket is chosen.
    """
    if not torch.cuda.is_available():
        cap = (9, 0)  # assume modern when querying from CPU
    else:
        if device is None:
            dev_idx = torch.cuda.current_device()
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(dev_idx)
    return _auto_config(dot_precision, cap)


# ════════════════════════════════════════════════════════════════
#  Phase A — split into KV and Z kernels
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_a_kv_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    I_minus_P_kv_ptr,  # output: (I - K_rot^T diag(β) K_rot)
    A_ptr,  # output: K_rot^T diag(β) V
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SKIP_RELU: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_kv_bhf = I_minus_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    A_bhf = A_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    k_nw_pair = tl.load(k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    # KV stream accumulators (in-loop fp32 to avoid bf16 round-off compounding)
    P_kv_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    A_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
        v_ptrs = qkv_bh + n_idx[:, None] * stride_n + 2 * stride_3 + offs_d[None, :] * stride_d
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        if SKIP_RELU:
            K = K_normed * k_scale
        else:
            K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        K_pair_raw = tl.reshape(
            tl.flip(tl.reshape(K_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )
        K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
        if SKIP_RELU:
            K_pair = K_pair_normed * k_scale
        else:
            K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        K_rot = K * Cos + K_pair * Sin

        beta_Krot = beta_t[:, None] * K_rot
        beta_V = beta_t[:, None] * V_raw

        K_rot_T = tl.trans(K_rot)
        P_kv_acc += tl.dot(K_rot_T.to(dot_dtype), beta_Krot.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
        A_acc += tl.dot(K_rot_T.to(dot_dtype), beta_V.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)

    # Store bf16 outputs. Padded positions are 0 by construction (K_rot is 0 outside D).
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    I_minus_P_kv = tl.where(diag_in_range, 1.0 - P_kv_acc, -P_kv_acc)
    if DOT_PRECISION >= 1:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv)
        tl.store(A_bhf + offs_dd, A_acc)
    else:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv.to(tl.bfloat16))
        tl.store(A_bhf + offs_dd, A_acc.to(tl.bfloat16))


@triton.jit
def _phase_a_z_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    I_minus_P_z_ptr,  # output: (I - K^T diag(β) K)
    B_ptr,  # output: K^T β
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Z stream: uses K (no RoPE). Cheaper than KV — no V load, no RoPE compute,
    no K_pair derivation."""
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_z_bhf = I_minus_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    B_bhf = B_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    P_z_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    B_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        # Only K_raw needed (no V, no Cos/Sin)
        k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        beta_K = beta_t[:, None] * K

        K_T = tl.trans(K)
        P_z_acc += tl.dot(K_T.to(dot_dtype), beta_K.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
        B_acc += tl.sum(beta_K, axis=0)

    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    I_minus_P_z = tl.where(diag_in_range, 1.0 - P_z_acc, -P_z_acc)

    if DOT_PRECISION >= 1:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z)
    else:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z.to(tl.bfloat16))
    # B stays fp32 (vector, only 0.5 KB, negligible HBM cost)
    tl.store(B_bhf + offs_d, B_acc)


def phase_a(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    num_warps: int | None = None,
    num_stages: int = 1,
    BLOCK_S: int | None = None,
    dot_precision: int = 0,
    skip_relu: bool = False,
    skip_z: bool = False,
):
    """Compute (I-P_kv), A, (I-P_z), B for all (B, H, F) via 2 kernels (KV + Z).

    `skip_relu=True` makes the K-stream prep a pure linear chain (no ReLU on
    K_normed * k_scale). Used by the camera-branch chunkwise wrapper, where K
    has already been ReLU'd by the cam_prep kernel and subsequently rotated
    by UCPE+RoPE — re-applying ReLU on the rotated values would clobber
    legitimate negatives.

    `skip_z=True` skips the Phase A Z kernel entirely and returns placeholder
    tensors for I_P_z and B_z. Used by NUM_ONLY callers (camera branch) to
    avoid wasted Z-stream prep when the denominator scan won't be used.
    """
    # Auto-pick (num_warps, BLOCK_S) per arch+precision unless overridden
    if num_warps is None or BLOCK_S is None:
        a_w, a_bs, *_ = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = a_w
        if BLOCK_S is None:
            BLOCK_S = a_bs
    B, N, three, H, D = qkv.shape
    assert three == 3 and N == F * S
    BLOCK_D = triton.next_power_of_2(D)
    BH = B * H

    # FAIR-COMPARE PATCH: keep fp32 inter-phase bridge at P0/P1 to match pytorch/fused
    bridge_dtype = torch.float32 if dot_precision >= 1 else torch.bfloat16
    I_P_kv = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    A = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)

    beta_c = beta.contiguous()
    grid = (BH * F,)

    _phase_a_kv_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        rope_cos,
        rope_sin,
        I_P_kv,
        A,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        SKIP_RELU=skip_relu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if skip_z:
        # NUM_ONLY callers (camera branch) do not consume the Z scan. Return
        # placeholders and let Phase B skip all Z loads/stores as well.
        I_P_z = torch.empty(1, device=qkv.device, dtype=bridge_dtype)
        B_z = torch.empty(1, device=qkv.device, dtype=torch.float32)
        return I_P_kv, A, I_P_z, B_z

    I_P_z = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    # B stays fp32 — small vector (0.5 KB/frame), no benefit to downcast
    B_z = torch.empty(BH, F, BLOCK_D, device=qkv.device, dtype=torch.float32)

    _phase_a_z_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        I_P_z,
        B_z,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return I_P_kv, A, I_P_z, B_z


# ════════════════════════════════════════════════════════════════
#  Phase B — serial scan, uses pre-stored (I - P) so MMA folds in M
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_b_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — read when LOAD_INIT_STATE=1
    init_state_z_ptr,  # (BH, BLOCK_D)
    final_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — written when SAVE_FINAL_STATE=1
    final_state_z_ptr,  # (BH, BLOCK_D)
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,  # forward scan seeded with init state (vs zeros)
    SAVE_FINAL_STATE: tl.constexpr,  # write M_{F-1} of forward scan to final_state_*
    DIRECTION: tl.constexpr,  # 0=both, 1=fwd-only, 2=rev-only
    COMBINED_HISTORY: tl.constexpr,  # 1 → rev branch read-add-stores into M_fwd_ptr
    # (M_hist[f] = M_fwd[f] + M_rev[f]); skips the F-1 zero-write so the fwd
    # value at F-1 is preserved (rev contribution there is exactly zero anyway).
    # Only meaningful when DIRECTION=0. Saves one Phase C launch + one M-shaped
    # buffer downstream (Phase C runs once on M_hist instead of twice).
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    bh = pid

    offs_d = tl.arange(0, BLOCK_D)
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]

    # ── Forward scan (skip when DIRECTION=2 i.e. rev-only) ──
    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd).to(tl.float32)
            if not SKIP_Z:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
            if not SKIP_Z:
                z = tl.zeros([BLOCK_D], dtype=tl.float32)
        for f in range(F):
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            # M = g · (I - P_kv) M + A_f
            if USE_ACC_FUSION:
                # Pre-scale (I-P) by g, accumulate A_f directly via the MMA accumulator.
                # Result: A_f + g·(I-P)·M  in one MMA — no separate M_temp tensor.
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            tl.store(M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d)
                # z = g · (I - P_z) z + B_f
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d, z)

        # Save terminal forward state for state-cached inference (autoregressive sampling).
        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d, z)

    # ── Reverse scan (skip when DIRECTION=1 i.e. fwd-only) ──
    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
        # COMBINED_HISTORY mode: rev contributions get read-add-stored into the
        # fwd buffer (which thereby becomes M_hist = M_fwd + M_rev). The F-1
        # zero-write is skipped so M_hist[F-1] keeps the fwd value (rev value
        # there is zero by construction, so no add needed).
        if not COMBINED_HISTORY:
            tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + (F - 1) * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d, z)
        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            if not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                # Read-add-store into the fwd buffer. The fwd loop has already
                # written M_fwd[f_dst] to this slot; we add the rev contribution
                # in place. Stays in L1/L2 since fwd just touched it.
                M_addr = M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd
                tl.store(M_addr, tl.load(M_addr) + M)
                if not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd, M)
                if not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d, z)


def phase_b_triton(
    I_P_kv,
    A,
    I_P_z,
    B,
    decay,
    F,
    num_warps=None,
    num_stages=None,
    use_acc_fusion=None,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
    direction=0,
    combined_history=False,
    skip_z=False,
):
    """Phase B serial-F scan over (B*H,).

    Forward scan can be seeded with `init_state_kv`/`init_state_z` (autoregressive
    sampling chunk > 0) and can write the terminal `M_{F-1}`/`z_{F-1}` to caller-
    provided buffers when `return_final_state=True`.

    `direction`: 0=both (default), 1=forward-only, 2=reverse-only. Forward-only
    skips reverse scan + reverse output buffers; reverse-only skips forward scan
    + state load/save. Used by single-direction state-cached entry points.

    `combined_history` (only meaningful with direction=0): the rev branch
    read-add-stores into the fwd buffer so its contents become
    M_hist[f] = M_fwd[f] + M_rev[f] (and same for z). Lets the caller run
    Phase C exactly once on the combined history, since Phase C is linear in
    M and z (`Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`). When set,
    M_rev/z_rev outputs are placeholder dummies; only M_fwd/z_fwd carry data.

    `skip_z`: skip the denominator/Z recurrence entirely. Used by camera
    numerator-only scans where Phase C runs with `num_only=True`.

    Returns (M_fwd, z_fwd, M_rev, z_rev) — and additionally (final_kv, final_z)
    when return_final_state=True. Skipped-direction outputs are returned as a
    1-element placeholder tensor (kernel never touches them when DIRECTION
    gates them off); callers should always discard the slot they didn't ask
    for. Reverse scan is always seeded with zeros (per upstream's bidi
    state-cache convention — only forward state is cached).
    """
    BH = I_P_kv.shape[0]
    _, _, BLOCK_D, _ = A.shape  # A is always full [BH, F, BLOCK_D, BLOCK_D]
    device, fdtype = I_P_kv.device, torch.float32

    if num_warps is None or num_stages is None or use_acc_fusion is None:
        _, _, b_w, b_s, b_acc, *_ = _get_arch_config(dot_precision, device=device)
        if num_warps is None:
            num_warps = b_w
        if num_stages is None:
            num_stages = b_s
        if use_acc_fusion is None:
            use_acc_fusion = b_acc

    if combined_history and direction != 0:
        raise ValueError("combined_history=True requires direction=0 (bidi)")

    # Phase B kernel is DIRECTION-gated (constexpr); skipped-direction writes
    # never happen, so we can hand it a 1-element placeholder for the inactive
    # buffers and free ~4× M_fwd-shaped allocations per single-direction call.
    decay_flat = decay.reshape(BH, F).contiguous().float()

    load_init = init_state_kv is not None
    dummy = torch.empty(1, device=device, dtype=fdtype)
    full_M = lambda: torch.empty(BH, F, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
    full_z = lambda: torch.empty(BH, F, BLOCK_D, device=device, dtype=fdtype)
    M_fwd = dummy if direction == 2 else full_M()
    z_fwd = dummy if (direction == 2 or skip_z) else full_z()
    # Combined-history mode reuses M_fwd/z_fwd as M_hist/z_hist; rev outputs
    # become placeholders even though DIRECTION!=1.
    M_rev = dummy if (direction == 1 or combined_history) else full_M()
    z_rev = dummy if (direction == 1 or combined_history or skip_z) else full_z()
    if load_init:
        init_kv = init_state_kv.contiguous().view(BH, BLOCK_D, BLOCK_D)
        init_z = dummy if skip_z else init_state_z.contiguous().view(BH, BLOCK_D)
    else:
        init_kv = dummy
        init_z = dummy

    if return_final_state:
        final_kv = torch.empty(BH, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
        final_z = dummy if skip_z else torch.empty(BH, BLOCK_D, device=device, dtype=fdtype)
    else:
        final_kv = dummy
        final_z = dummy

    d_splits, nw_override, ns_override, acc_override = _pick_phase_b_d_splits(BLOCK_D, dot_precision=dot_precision)
    if d_splits > 1:
        D_TILE = BLOCK_D // d_splits
        # Use D-tile-specific tuning if available, else fall back to baseline tuning
        nw_use = nw_override if nw_override is not None else num_warps
        ns_use = ns_override if ns_override is not None else num_stages
        acc_use = acc_override if acc_override is not None else use_acc_fusion
        _phase_b_dtile_kernel[(BH, d_splits)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            D_TILE=D_TILE,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=acc_use,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=nw_use,
            num_stages=ns_use,
        )
    else:
        _phase_b_kernel[(BH,)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=use_acc_fusion,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if return_final_state:
        return M_fwd, z_fwd, M_rev, z_rev, final_kv, final_z
    return M_fwd, z_fwd, M_rev, z_rev


# ════════════════════════════════════════════════════════════════
#  Phase B D-tile — j-axis split for grid parallelism (#118)
# ════════════════════════════════════════════════════════════════
# Same recurrence as _phase_b_kernel but each program owns a D_TILE-wide
# slice of M's output column dim. Grid: (BH, d_splits). M_new[*, j_tile]
# only depends on M_prev[*, j_tile] and full (I-P_kv) — independent across
# j-tiles. z is unsplittable; only `pid_d == 0` updates/writes z.
@triton.jit
def _phase_b_dtile_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    DIRECTION: tl.constexpr,
    COMBINED_HISTORY: tl.constexpr,
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = pid_bh

    offs_d_full = tl.arange(0, BLOCK_D)
    offs_d_tile = pid_d * D_TILE + tl.arange(0, D_TILE)
    offs_dd_full = offs_d_full[:, None] * BLOCK_D + offs_d_full[None, :]
    offs_dd_tile = offs_d_full[:, None] * BLOCK_D + offs_d_tile[None, :]

    is_lead = pid_d == 0

    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
            if is_lead and LOAD_INIT_STATE:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d_full).to(tl.float32)

        for f in range(F):
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_full)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_tile)
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            tl.store(M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_tile, M)

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_full)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full, z)

        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d_full, z)

    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)

        if not COMBINED_HISTORY:
            tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + (F - 1) * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d_full, z)

        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_full)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_tile)
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_full)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                M_addr = M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd_tile
                tl.store(M_addr, tl.load(M_addr) + M)
                if is_lead and not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd_tile, M)
                if is_lead and not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full, z)


_PHASE_B_DTILE_ARCH_CACHE: dict = {}  # (dev, dot_prec) -> (d_splits, nw, ns, acc)


# Per-arch D-tile optimum from 2026-04-29 sweep (T=11 B=1 P0 IEEE):
#   WGMMA-server (A100 sm_80, H100 sm_90): (d=4, nw=32, ns=1, acc=True)
#   Blackwell-family (GB200 sm_100, 5090 sm_120, GB10 sm_121, Ada sm_89):
#                                          (d=8, nw=4,  ns=1, acc=False)
# Both clusters were tested across 96 configs (4 ds × 4 nw × 3 ns × 2 acc).
def _pick_phase_b_d_splits(BLOCK_D: int, dot_precision: int = 0):
    """Returns (d_splits, nw_override, ns_override, acc_override).

    `d_splits=1` → use baseline `_phase_b_kernel` with `_CHUNKWISE_TUNING` config.
    `d_splits>1` → use `_phase_b_dtile_kernel` with overrides for nw/ns/acc.
    Override via env: PHASE_B_D_SPLITS, PHASE_B_DTILE_NW, PHASE_B_DTILE_NS,
    PHASE_B_DTILE_ACC (1=True / 0=False).
    """
    import os

    env_d = os.environ.get("PHASE_B_D_SPLITS", None)
    if env_d is not None:
        d = int(env_d)
        if d < 1 or BLOCK_D % d != 0:
            return (1, None, None, None)
        nw = int(os.environ.get("PHASE_B_DTILE_NW", "0")) or None
        ns = int(os.environ.get("PHASE_B_DTILE_NS", "0")) or None
        acc_env = os.environ.get("PHASE_B_DTILE_ACC", None)
        acc = bool(int(acc_env)) if acc_env is not None else None
        return (d, nw, ns, acc)
    try:
        import torch

        dev = torch.cuda.current_device()
        cache_key = (dev, dot_precision)
        if cache_key not in _PHASE_B_DTILE_ARCH_CACHE:
            cap = torch.cuda.get_device_capability(dev)
            major, minor = cap[0], cap[1]
            if dot_precision == 2:
                # IEEE fp32: D-tile dominates baseline on every arch (96-config sweep).
                if major == 8 and minor == 0:
                    cfg = (4, 32, 1, True)  # A100
                elif major == 9:
                    cfg = (4, 32, 1, True)  # H100 (Hopper)
                elif major == 8 and minor == 9:
                    cfg = (8, 4, 1, False)  # Ada (assume Blackwell-like)
                elif major >= 10:
                    cfg = (8, 4, 1, False)  # GB200/B200, 5090, GB10
                else:
                    cfg = (1, None, None, None)  # unknown — baseline
            else:
                # bf16/TF32: cap-specific dispatch. Multi-arch sweep 2026-05-06
                # (F=11 S=920) determined per-cap whether D-tile beats the
                # baseline _phase_b_kernel:
                #   sm_80  A100:   D-tile WIN 1.09× (P1) / 1.02× (P2) — (4,8,2,F).
                #   sm_90  H100:   D-tile WIN ~10% — P1 (4,8,2,F); P2 (8,8,2,F).
                #                  Use (4,8,2,F) for both (P2 within 0.4%).
                #   sm_100 GB200:  D-tile WIN ~12% — (4,8,2,F) both precisions.
                #   sm_120 5090:   D-tile WIN 2.6× (P1) / 1.13× (P2) — (8,8,1,F).
                #                  TF32 baseline OOMs at 102 KB SRAM cap.
                #   sm_121 GB10:   D-tile LOSS 4% — baseline wins. Despite same
                #                  reported SRAM/SM as sm_120, the baseline
                #                  kernel fits all configs up to nw=16 ns=2 on
                #                  sm_121 (Triton/codegen difference between
                #                  consumer-Blackwell variants), so baseline
                #                  saturates the chip without needing D-tile.
                if major == 8 and minor == 0:
                    cfg = (4, 8, 2, False)  # A100
                elif major == 9:
                    cfg = (4, 8, 2, False)  # H100
                elif major == 10:
                    cfg = (4, 8, 2, False)  # GB200 / B200
                elif major == 12 and minor == 0:
                    cfg = (8, 8, 1, False)  # 5090
                elif major == 12 and minor == 1:
                    cfg = (1, None, None, None)  # GB10 — baseline wins
                else:
                    cfg = (1, None, None, None)  # Ada, unknown
            _PHASE_B_DTILE_ARCH_CACHE[cache_key] = cfg
        return _PHASE_B_DTILE_ARCH_CACHE[cache_key]
    except (RuntimeError, AttributeError):
        return (1, None, None, None)


# ════════════════════════════════════════════════════════════════
#  Phase C — Pass 2 output (per (B, H, F)). Same as v1.
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_c_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    q_inv_rms_ptr,
    q_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    M_ptr,
    z_ptr,
    num_ptr,
    den_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ACCUMULATE: tl.constexpr = False,
    SKIP_LAST_F: tl.constexpr = False,
    SKIP_RELU: tl.constexpr = False,
    NUM_ONLY: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    # Reverse-accumulate callers pass SKIP_LAST_F=True: M_rev[F-1] / z_rev[F-1]
    # are exactly zero (Phase B initializes the reverse scan with zeros and the
    # write loop only fills f<F-1), so the f=F-1 program would only re-write
    # the forward pass's output unchanged. Early-return saves one frame's worth
    # of Q+RoPE HBM reads + dot products per (B, H).
    if SKIP_LAST_F and pid_f == F - 1:
        return

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    num_bh = num_ptr + pid_b * (N * H * D) + pid_h * D
    den_bh = den_ptr + bh * N
    M_bhf = M_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    z_bhf = z_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    nw_offset = pid_h * D
    q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    q_nw_pair = tl.load(q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    M_f = tl.load(M_bhf + offs_dd, mask=mask_dd, other=0.0)
    if NUM_ONLY:
        # NUM_ONLY callers (cam scan) pass a 1-element placeholder for z;
        # never dereference it — the addresses above would be out of bounds.
        z_f = tl.zeros([BLOCK_D], dtype=tl.float32)
    else:
        z_f = tl.load(z_bhf + offs_d, mask=mask_d, other=0.0)

    n_base = pid_f * S
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        q_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d[None, :] * stride_d
        Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_pair_raw = tl.reshape(
            tl.flip(tl.reshape(Q_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )

        q_inv_rms = tl.load(q_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
        Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
        if SKIP_RELU:
            Q = Q_normed
            Q_pair = Q_pair_normed
        else:
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_rot = Q * Cos + Q_pair * Sin

        num = tl.dot(Q_rot.to(dot_dtype), M_f.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
        if not NUM_ONLY:
            den = tl.sum(Q * z_f[None, :], axis=1)

        num_ptrs = num_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
        if not NUM_ONLY:
            den_ptrs = den_bh + n_idx
        if ACCUMULATE:
            # Used by reverse-direction Phase C: add this pass onto forward's
            # already-written buffer instead of allocating a separate one.
            prev_num = tl.load(num_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            num = num + prev_num
            if not NUM_ONLY:
                prev_den = tl.load(den_ptrs, mask=mask_s, other=0.0).to(tl.float32)
                den = den + prev_den
        if DOT_PRECISION >= 1:
            tl.store(num_ptrs, num, mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den, mask=mask_s)
        else:
            tl.store(num_ptrs, num.to(tl.bfloat16), mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den.to(tl.bfloat16), mask=mask_s)


def phase_c(
    qkv,
    q_inv_rms,
    q_norm_w,
    rope_cos,
    rope_sin,
    M,
    z,
    F,
    S,
    num_warps=None,
    num_stages=None,
    BLOCK_S=None,
    dot_precision=0,
    num_out=None,
    den_out=None,
    accumulate=False,
    skip_last_frame=False,
    skip_relu: bool = False,
    num_only: bool = False,
):
    """Phase C Pass-2 output. Optionally accumulates into caller-provided
    ``num_out``/``den_out`` buffers (used to fuse reverse-direction output into
    forward-direction buffer without allocating a separate one — saves ~45 MB
    at B=1 bf16, ~180 MB at B=4).

    ``skip_last_frame=True`` early-returns the f=F-1 programs. Valid for the
    reverse-accumulate call only, where M[F-1]/z[F-1] are guaranteed zero.

    ``skip_relu=True`` matches Phase A KV's flag — used by the camera-branch
    chunkwise wrapper where Q has already been ReLU'd by cam_prep before
    being rotated by UCPE+RoPE; re-applying ReLU on the rotated Q would
    clobber legitimate negatives.

    ``num_only=True`` skips the denominator computation and store entirely
    (kernel writes only ``num_out``; ``den_out`` is allowed to be None /
    unallocated). Used by the camera-branch which has no Z scan.
    """
    if num_warps is None or num_stages is None or BLOCK_S is None:
        *_, c_w, c_bs, c_s = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = c_w
        if num_stages is None:
            num_stages = c_s
        if BLOCK_S is None:
            BLOCK_S = c_bs
    B, N, three, H, D = qkv.shape
    BLOCK_D = triton.next_power_of_2(D)
    if num_out is None:
        num_out = torch.empty(
            B, N, H, D, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16)
        )
    if den_out is None and not num_only:
        den_out = torch.empty(
            B, H, N, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16)
        )
    elif num_only and den_out is None:
        # Pass a 1-element placeholder; kernel guards den loads/stores under NUM_ONLY.
        den_out = torch.empty(1, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16))

    _phase_c_kernel[(B * H * F,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M,
        z,
        num_out,
        den_out,
        H=H,
        F=F,
        S=S,
        D=D,
        NORM_EPS=1e-5,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        ACCUMULATE=1 if accumulate else 0,
        SKIP_LAST_F=skip_last_frame,
        SKIP_RELU=skip_relu,
        NUM_ONLY=num_only,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return num_out, den_out


def fused_bigdn_bidi_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_w,
    k_norm_w,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale=1.0,
    eps=1e-6,
    norm_eps=1e-5,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
):
    """Bidi chunkwise GDN forward, optionally with state-cache for autoregressive
    sampling (chunk 0 = full bidi with state save; chunks > 0 seed forward scan
    from saved state). Reverse always seeds from zero per upstream convention.

    Pipeline (2026-04-25 restructure): Phase A once → Phase B direction=0 with
    combined_history=True (fwd seeded with init_state and saves final state;
    rev zero-seeded; rev output summed into fwd buffer in-kernel via read-
    add-store so on exit M_hist[f] = M_fwd[f] + M_rev[f]) → Phase C ONCE on
    M_hist. Phase C linearity `Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`
    makes the in-kernel sum exact.

    Replaces the prior 2× Phase B + 2× Phase C pattern. Saves one Phase C
    launch + one Q+RoPE HBM pass and one M-shape buffer per call.
    """
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_w,
        k_norm_w,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )

    if return_final_state:
        M_hist, z_hist, _, _, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            return_final_state=True,
            combined_history=True,
        )
    else:
        M_hist, z_hist, _, _ = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            combined_history=True,
        )
    num_out, den_out = phase_c(
        qkv,
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        accumulate=False,
    )
    del M_hist, z_hist, I_P_kv, A, I_P_z, B_z

    # ── Final divide ──
    total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
    out = (num_out.float() / (total_den + eps)).to(qkv.dtype)
    del num_out, den_out, total_den
    if return_final_state:
        B = qkv.shape[0]
        H = qkv.shape[3]
        D = qkv.shape[4]
        BLOCK_D = final_kv.shape[1]
        state_kv = final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
        state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return out, state_kv, state_z
    return out


def _default_dot_prec():
    """Pull dot_precision from `_resolve_launch_config` (honors PRECISION_OVERRIDE)."""

    _, dot_prec, _, _ = _resolve_launch_config()
    return dot_prec


# ─────────────────────────────────────────────────────────────────────────────
#  Camera-branch wrapper — numerator-only single-path delta-rule scan via
#  chunkwise. Drop-in for `diffusion.model.ops.fused_cam_gdn.cam_scan_func`.
#
#  Cam math expanded:
#      state = state * g                               # apply decay
#      state += K^T @ ((V - K @ state) * β)            # delta-rule
#  Equivalently:
#      state_new = g (I - K^T β K) state_old + K^T β V
#                = g (I - P_kv) state_old + A
#  This is bit-identical to chunkwise's Phase B M update, so the scan kernel
#  is reusable. The only differences from main GDN:
#    1. Q/K/V come pre-prepped (cam_prep_kernel did RMSNorm+ReLU+UCPE+RoPE).
#       We disable chunkwise's prep with identity tables (k_inv_rms=1, k_nw=1,
#       k_scale=1, rope_cos=1, rope_sin=0) AND skip_relu=True (because cam
#       applied ReLU BEFORE UCPE; the post-UCPE values can have legitimate
#       negatives that re-applying ReLU would clobber).
#    2. No Z denominator scan; output is num-only (out = Q @ M, no /Z).
#       skip_z=True elides Phase A Z; num_only=True elides Phase C den compute.
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=_CAM_IDENTITY_CACHE_MAXSIZE)
def _cam_identity_tables_cached(
    device_type: str,
    device_index: int | None,
    B: int,
    N: int,
    H: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    ones_inv_rms = torch.ones(B, N, device=device, dtype=torch.float32)
    ones_nw = torch.ones(H * D, device=device, dtype=torch.float32)
    ones_cos = torch.ones(N, D, device=device, dtype=torch.float32)
    zeros_sin = torch.zeros(N, D, device=device, dtype=torch.float32)
    return ones_inv_rms, ones_nw, ones_cos, zeros_sin


def _cam_identity_tables(
    *,
    B: int,
    N: int,
    H: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded-LRU-cached identity RMS/RoPE tables used by ``cam_scan_bidi_chunkwise``.

    Normalizes ``device`` to a hashable ``(type, index)`` pair so equivalent
    devices (e.g. ``cuda`` vs ``cuda:0``) share a cache entry.
    """
    device_index = device.index if device.type == "cuda" else None
    return _cam_identity_tables_cached(device.type, device_index, B, N, H, D)


def cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: int | None = None,
) -> torch.Tensor:
    """Bidirectional camera scan using shared chunkwise phases.

    This is equivalent to a forward + reverse single-direction camera scan
    summed together for full bidirectional attention,
    but it packs QKV once, runs Phase A once, combines forward/reverse histories
    inside Phase B, and runs Phase C once on the summed state.
    """
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert q.dtype == torch.float32, f"cam_scan_bidi_chunkwise requires fp32 q/k/v (got {q.dtype})"

    B, H, D, N = q.shape
    F = beta.shape[2]
    assert N % F == 0
    S = N // F
    assert beta.shape == (B, H, F, S)
    assert decay.shape == (B, H, F)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=q.device)
    I_P_kv, A_, I_P_z, B_z = phase_a(
        qkv,
        beta,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    M_hist, z_hist, _, _ = phase_b_triton(
        I_P_kv,
        A_,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=0,
        combined_history=True,
        skip_z=True,
    )
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )
    return num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)


SANA_WM_DISABLE_TRITON_GDN_ENV = "VLLM_OMNI_SANA_WM_DISABLE_TRITON_GDN"
SANA_WM_REQUIRE_TRITON_GDN_ENV = "VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN"


def _validate_gdn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Sana-WM GDN query/key/value must be shaped [B, H, D, N].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Sana-WM GDN query/key/value shapes must match, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
        )
    if beta.ndim != 4:
        raise ValueError("Sana-WM GDN beta must be shaped [B, H, T, S].")
    if decay.ndim != 3:
        raise ValueError("Sana-WM GDN decay must be shaped [B, H, T].")
    if spatial_tokens <= 0:
        raise ValueError("Sana-WM GDN spatial_tokens must be positive.")

    batch_size, num_heads, head_dim, token_count = query.shape
    if token_count % spatial_tokens != 0:
        raise ValueError(f"Sana-WM GDN token count {token_count} is not divisible by spatial_tokens={spatial_tokens}.")
    frames = token_count // spatial_tokens
    if beta.shape != (batch_size, num_heads, frames, spatial_tokens):
        raise ValueError(
            "Sana-WM GDN beta shape mismatch: expected "
            f"{(batch_size, num_heads, frames, spatial_tokens)}, got {tuple(beta.shape)}."
        )
    if decay.shape != (batch_size, num_heads, frames):
        raise ValueError(
            f"Sana-WM GDN decay shape mismatch: expected {(batch_size, num_heads, frames)}, got {tuple(decay.shape)}."
        )
    return batch_size, num_heads, head_dim, token_count, frames


def _delta_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, head_dim, token_count = query.shape
    frames = beta.shape[2]

    def to_frames(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

    query_f = to_frames(query)
    key_f = to_frames(key)
    value_f = to_frames(value)
    query_rot_f = to_frames(query_rot)
    key_rot_f = to_frames(key_rot)
    state_kv = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=query.device, dtype=query.dtype)
    state_z = torch.zeros(batch_size, num_heads, head_dim, 1, device=query.device, dtype=query.dtype)
    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []

    for frame_idx in range(frames):
        query_t = query_f[:, :, frame_idx]
        key_t = key_f[:, :, frame_idx]
        value_t = value_f[:, :, frame_idx]
        query_rot_t = query_rot_f[:, :, frame_idx]
        key_rot_t = key_rot_f[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx].unsqueeze(2)
        decay_t = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

        state_kv = state_kv * decay_t
        state_z = state_z * decay_t
        value_pred = torch.matmul(state_kv, key_rot_t)
        delta_value = (value_t - value_pred) * beta_t
        state_kv = state_kv + torch.matmul(delta_value, key_rot_t.transpose(-1, -2))

        z_pred = torch.matmul(state_z.transpose(-1, -2), key_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(key_t, delta_z.transpose(-1, -2))

        numerators.append(torch.matmul(state_kv, query_rot_t))
        denominators.append(torch.matmul(state_z.transpose(-1, -2), query_t))

    def restore(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        stacked = torch.stack(tensors, dim=2)
        return stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, dim, token_count)

    return restore(numerators, head_dim), restore(denominators, 1)


def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
    flipped = torch.flip(tensor, dims=[dim])
    shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([padding, shifted], dim=dim)


def reference_bidirectional_gated_delta_net(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
    query_rot: torch.Tensor | None = None,
    key_rot: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the SANA-WM bidirectional gated delta recurrence in PyTorch.

    Inputs follow the layout used by the official Stage-1 operator:
    ``query/key/value/query_rot/key_rot`` are ``[B, H, D, T*S]``, ``beta`` is
    ``[B, H, T, S]``, and ``decay`` is ``[B, H, T]``.
    """

    batch_size, num_heads, head_dim, token_count, frames = _validate_gdn_inputs(
        query, key, value, beta, decay, spatial_tokens
    )
    if query_rot is None:
        query_rot = query
    if key_rot is None:
        key_rot = key
    if query_rot.shape != query.shape or key_rot.shape != key.shape:
        raise ValueError("Sana-WM GDN rotary query/key shapes must match query/key.")

    dtype_orig = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    query_rot = query_rot.float()
    key_rot = key_rot.float()
    beta = beta.float()
    decay = decay.float()

    num_fwd, den_fwd = _delta_scan(
        query,
        key,
        value,
        query_rot,
        key_rot,
        beta,
        decay,
        spatial_tokens=spatial_tokens,
    )

    def to_time(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

    def from_time(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)

    query_t = to_time(query)
    key_t = to_time(key)
    value_t = to_time(value)
    query_rot_t = to_time(query_rot)
    key_rot_t = to_time(key_rot)
    num_bwd_flipped, den_bwd_flipped = _delta_scan(
        from_time(torch.flip(query_t, dims=[2])),
        from_time(_flip_and_shift(key_t, dim=2, shift_value=0.0)),
        from_time(_flip_and_shift(value_t, dim=2, shift_value=0.0)),
        from_time(torch.flip(query_rot_t, dims=[2])),
        from_time(_flip_and_shift(key_rot_t, dim=2, shift_value=0.0)),
        _flip_and_shift(beta, dim=2, shift_value=0.0),
        _flip_and_shift(decay, dim=2, shift_value=1.0),
        spatial_tokens=spatial_tokens,
    )

    def flip_back(tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.shape[2]
        tensor = tensor.view(batch_size, num_heads, dim, frames, spatial_tokens)
        return torch.flip(tensor, dims=[3]).reshape(batch_size, num_heads, dim, token_count)

    output = (num_fwd + flip_back(num_bwd_flipped)) / (den_fwd + flip_back(den_bwd_flipped) + eps)
    return output.to(dtype_orig)


def _triton_disabled() -> bool:
    return os.environ.get(SANA_WM_DISABLE_TRITON_GDN_ENV, "").lower() in {"1", "true", "yes", "on"}


def _rms_norm_weight(module: nn.Module, hidden_size: int, *, device: torch.device) -> torch.Tensor:
    del hidden_size, device
    return module.weight.float().contiguous()


def triton_bidirectional_gated_delta_net_from_qkv(
    qkv: torch.Tensor,
    *,
    beta: torch.Tensor,
    decay: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    spatial_tokens: int,
    rotary_emb: torch.Tensor | None = None,
    k_scale: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the fused NVlabs bidirectional GDN kernel from raw QKV.

    Args:
      qkv: ``[B, N, 3, H, D]`` raw projected Q/K/V after temporal K-conv.
      beta: ``[B, H, T, S]`` token gates.
      decay: ``[B, H, T]`` per-frame decay.

    Returns:
      Tensor shaped ``[B, H, D, N]`` to match
      :func:`reference_bidirectional_gated_delta_net`.
    """

    if _triton_disabled():
        raise RuntimeError(f"{SANA_WM_DISABLE_TRITON_GDN_ENV} disables the fused Sana-WM GDN kernel.")
    if not qkv.is_cuda:
        raise RuntimeError("Sana-WM fused GDN requires CUDA tensors.")
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Sana-WM fused GDN expects qkv [B, N, 3, H, D], got {tuple(qkv.shape)}.")
    if spatial_tokens <= 0:
        raise ValueError("Sana-WM fused GDN spatial_tokens must be positive.")

    batch_size, token_count, _, num_heads, head_dim = qkv.shape
    if token_count % spatial_tokens != 0:
        raise ValueError(
            f"Sana-WM fused GDN token count {token_count} is not divisible by spatial_tokens={spatial_tokens}."
        )
    frames = token_count // spatial_tokens
    if beta.shape != (batch_size, num_heads, frames, spatial_tokens):
        raise ValueError(
            "Sana-WM fused GDN beta shape mismatch: expected "
            f"{(batch_size, num_heads, frames, spatial_tokens)}, got {tuple(beta.shape)}."
        )
    if decay.shape != (batch_size, num_heads, frames):
        raise ValueError(
            "Sana-WM fused GDN decay shape mismatch: expected "
            f"{(batch_size, num_heads, frames)}, got {tuple(decay.shape)}."
        )

    qkv = qkv.contiguous()
    # q_norm is a vLLM RMSNorm on the production path, which stores its epsilon
    # as ``variance_epsilon`` (no ``eps`` attribute); the 1e-5 fallback preserves
    # the validated fused-path numerics for that case (and any norm without eps).
    q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=float(getattr(q_norm, "eps", 1e-5)))
    hidden_size = num_heads * head_dim
    q_norm_weight = _rms_norm_weight(q_norm, hidden_size, device=qkv.device)
    k_norm_weight = _rms_norm_weight(k_norm, hidden_size, device=qkv.device)
    rope_cos, rope_sin = prepare_rope_tables(rotary_emb, token_count, head_dim, qkv.device)
    output = fused_bigdn_func(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        beta=beta.contiguous(),
        decay=decay.contiguous(),
        F=frames,
        S=spatial_tokens,
        k_scale=k_scale,
        eps=eps,
    )
    return output.permute(0, 2, 3, 1).contiguous()


__all__ = [
    "SANA_WM_DISABLE_TRITON_GDN_ENV",
    "SANA_WM_REQUIRE_TRITON_GDN_ENV",
    "reference_bidirectional_gated_delta_net",
    "triton_bidirectional_gated_delta_net_from_qkv",
]
