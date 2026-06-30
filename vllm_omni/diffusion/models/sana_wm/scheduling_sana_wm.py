# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 flow scheduler helpers."""

from __future__ import annotations

import torch

SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT = 9.8


class SanaWmFlowMatchScheduler:
    """Native flow-matching Euler scheduler matching NVlabs ``LTXFlowEuler``.

    Reproduces ``diffusers.FlowMatchEulerDiscreteScheduler(shift=shift)``
    timestep + sigma tables (which NVlabs uses) without depending on
    diffusers at runtime. The schedule has a subtle but load-bearing
    quirk: ``sigma_min`` is set at construction time to the
    shift-transformed ``1/num_train`` (not the raw ``1/num_train``), so
    ``set_timesteps`` ends up applying the shift formula **twice** — once
    implicitly via the shifted ``sigma_min`` linspace endpoint, once
    explicitly. For ``shift=9.8, N=3`` this yields timesteps
    ``[1000, 909.0, 87.7]`` (matches NVlabs) rather than the near-linear
    ``[999, 951, 830]`` that the prior ``DPMSolverMultistep`` wrapper
    produced.
    """

    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        num_inference_steps: int,
        shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")
        self.num_inference_steps = num_inference_steps
        self.shift = float(shift)
        self.num_train_timesteps = self.NUM_TRAIN_TIMESTEPS
        # Shifted sigma_min: raw min is 1/num_train; the FlowMatchEuler
        # __init__ shifts the full sigma array, so the persisted sigma_min
        # is the shifted value. set_timesteps then linspaces from
        # sigma_max=1.0 down to this shifted sigma_min.
        raw_min = 1.0 / self.num_train_timesteps
        self._sigma_min_shifted = self.shift * raw_min / (1.0 + (self.shift - 1.0) * raw_min)
        self._timesteps_tensor: torch.Tensor | None = None  # (N,)
        self._sigmas_tensor: torch.Tensor | None = None  # (N+1,)
        self._timesteps_device: torch.device | None = None

    def _build_schedule(self, device: torch.device) -> None:
        # Replay diffusers FlowMatchEulerDiscreteScheduler.set_timesteps:
        # ts_pre = linspace(num_train * sigma_max, num_train * sigma_min_shifted, N)
        # sig_pre = ts_pre / num_train
        # sigmas = shift * sig_pre / (1 + (shift-1) * sig_pre)   <-- second shift
        # timesteps = sigmas * num_train
        # sigmas = cat([sigmas, 0])
        N = self.num_inference_steps
        ts_pre = torch.linspace(
            float(self.num_train_timesteps),
            float(self.num_train_timesteps) * self._sigma_min_shifted,
            N,
            dtype=torch.float32,
            device=device,
        )
        sig_pre = ts_pre / float(self.num_train_timesteps)
        sigmas = self.shift * sig_pre / (1.0 + (self.shift - 1.0) * sig_pre)
        timesteps = sigmas * float(self.num_train_timesteps)

        sigmas_with_terminal = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32, device=device)])
        self._timesteps_tensor = timesteps
        self._sigmas_tensor = sigmas_with_terminal

    def _ensure_timesteps(self, device: torch.device) -> None:
        if self._timesteps_device != device or self._timesteps_tensor is None:
            self._build_schedule(device)
            self._timesteps_device = device

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        self._ensure_timesteps(device)
        return self._timesteps_tensor

    @property
    def sigmas(self) -> torch.Tensor:
        if self._sigmas_tensor is None:
            self._build_schedule(torch.device("cpu"))
        return self._sigmas_tensor

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token flow-matching Euler step.

        ``prev = latents + (sigma_next - sigma_cur) * (-noise_pred)`` —
        the NVlabs ``LTXFlowEuler`` sign convention. Used by the legacy
        non-per-token path; the per-token path is
        :meth:`step_flow_euler_per_token`.
        """
        self._ensure_timesteps(latents.device)
        sigmas = self._sigmas_tensor
        idx = self._sigma_index_for(timestep)
        sigma_cur = sigmas[idx]
        sigma_next = sigmas[idx + 1]
        dt = (sigma_next - sigma_cur).to(latents.dtype)
        return (latents + dt * (-noise_pred.to(latents.dtype))).to(latents.dtype)

    def _sigma_index_for(self, timestep: torch.Tensor) -> int:
        """Return the index ``i`` such that ``self.timesteps_tensor[i] == timestep``.

        ``argmin(abs(diff))`` so the lookup survives fp16/bf16/fp32
        round-trips of ``timestep``.
        """
        ts = self._timesteps_tensor
        t_val = timestep.to(device=ts.device, dtype=ts.dtype)
        return int((ts - t_val).abs().argmin().item())

    def step_flow_euler_per_token(
        self,
        noise_pred: torch.Tensor,  # (B, C, F, H, W)
        timestep: torch.Tensor,  # scalar current step
        latents: torch.Tensor,  # (B, C, F, H, W)
        per_token_timesteps: torch.Tensor,  # (B, F*H*W) per-token integer timestep
    ) -> torch.Tensor:
        """Per-token flow-matching Euler step matching NVlabs ``LTXFlowEuler.sample``.

        NVlabs invokes diffusers
        ``FlowMatchEulerDiscreteScheduler.step`` with the noise prediction
        sign-flipped and ``per_token_timesteps``. In that branch, diffusers
        does not use ``self.sigmas[step_index]`` as the current sigma; it
        derives the current sigma directly from each token's timestep
        (``per_token_t / num_train_timesteps``), then selects the largest
        scheduler sigma strictly below it as the next sigma. This matters at
        ``t=1000``: the first full-noise step advances the scheduler index,
        but NVlabs' post-step token mask can still discard the update.

        Args:
            noise_pred: ``(B, C, F, H, W)`` model output.
            timestep: scalar integer timestep at this sampling step.
            latents: ``(B, C, F, H, W)`` current latent.
            per_token_timesteps: ``(B, F*H*W)`` per-token current
                timestep, with conditioning tokens set to 0.

        Returns:
            ``(B, C, F, H, W)`` updated latent (``prev_sample``).
        """
        if noise_pred.shape != latents.shape:
            raise ValueError(
                "Sana-WM per-token step: noise_pred / latents shape mismatch "
                f"({tuple(noise_pred.shape)} vs {tuple(latents.shape)})."
            )
        if latents.ndim != 5:
            raise ValueError(f"Sana-WM per-token step expects (B, C, F, H, W); got {tuple(latents.shape)}.")
        self._ensure_timesteps(latents.device)
        sigmas = self._sigmas_tensor.to(device=latents.device, dtype=torch.float32)
        per_token_sigmas = per_token_timesteps.to(device=latents.device, dtype=torch.float32) / float(
            self.num_train_timesteps
        )
        scheduler_sigmas = sigmas[:, None, None]
        lower_mask = scheduler_sigmas < per_token_sigmas[None] - 1e-6
        lower_sigmas = (lower_mask * scheduler_sigmas).max(dim=0).values
        # Matches diffusers' per-token branch: dt is positive and
        # ``model_output`` already carries the sign selected by the caller.
        dt = per_token_sigmas - lower_sigmas  # (B, FHW)

        # Sign convention follows NVlabs `LTXFlowEuler.sample`, which calls
        # diffusers with `model_output=-noise_pred`. Diffusers then applies
        # `prev = sample + dt * model_output`.
        b, c, f, h, w = latents.shape
        latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        noise_flat = noise_pred.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        prev_flat = latents_flat + dt.unsqueeze(-1) * (-noise_flat)
        prev = prev_flat.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return prev.to(latents.dtype)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Linear flow interpolation: sigma*noise + (1-sigma)*sample.

        ``sigma = timestep / num_train_timesteps`` maps the scheduler's integer
        timestep (0–1000 range) back to the [0, 1] noise level.
        """
        num_train = float(self.num_train_timesteps)
        sigma = (timestep.float() / num_train).clamp(0.0, 1.0)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (sigma * noise + (1.0 - sigma) * sample).to(sample.dtype)
