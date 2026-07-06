# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4: refine OUR port's stage-1 latents with the 17B LTX-2 refiner, then
decode with the causal VAE. Runs as a SEPARATE job from the stage-1 rollout (the
17B refiner + the DiT together would OOM the 80GB card).

Input:  outputs/sana_stream_video/demo_0_stream_causal_latents.pt  (from
        sana_stream_video.py with SANA_AR_SAVE_LATENTS=1)
Output: outputs/sana_stream_video/demo_0_refined.mp4 + key-frame PNGs.

The refiner (`DiffusersLTX2Refiner.refine_latents`) encodes the prompt with
gemma3-12B and refines block-by-block (chunk-causal AR, sink anchor, distilled
3-step sigmas) internally; we just hand it the stage-1 latent + prompt.
Run via gpuq; PYTHONPATH incl NVlabs-Sana.
"""

from __future__ import annotations

import gc
import glob
import os
from pathlib import Path

os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
from PIL import Image

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

HUB = Path.home() / ".cache/huggingface/hub"
STREAM_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_streaming/snapshots/*/"))[0])
ASSETS = Path(os.environ.get("SANA_WM_ASSETS", "NVlabs-Sana/asset/sana_wm"))
DEMO = os.environ.get("SANA_AR_DEMO", "demo_0")
OUT_DIR = Path(os.environ.get("SANA_AR_OUT_DIR", "outputs/sana_stream_video"))
FPS = float(os.environ.get("SANA_AR_FPS", "16"))
LAT_PATH = Path(os.environ.get("SANA_AR_LATENTS", str(OUT_DIR / f"{DEMO}_stream_causal_latents.pt")))

prompt_text = (ASSETS / f"{DEMO}.txt").read_text().strip()
latents = torch.load(str(LAT_PATH), map_location="cpu").to(DEVICE, DTYPE)
print(f"loaded stage-1 latents {tuple(latents.shape)} from {LAT_PATH}")
F = latents.shape[2]
active = F - 1  # sink_size=1
print(f"latent frames={F}  active(after sink)={active}  block_size=3 -> {active // 3} blocks "
      f"(divisible={active % 3 == 0})")

# --- build the 17B refiner + refine ------------------------------------------
from diffusion.refiner.diffusers_ltx2_refiner import DiffusersLTX2Refiner  # noqa: E402

print("building 17B LTX-2 refiner ...")
refiner = DiffusersLTX2Refiner(
    refiner_root=STREAM_SNAP / "refiner_diffusers",
    gemma_root=STREAM_SNAP / "gemma3_12b",
    dtype=DTYPE,
    device=DEVICE,
)
print("refining (block_size=3, sink_size=1, distilled 3-step) ...")
with torch.no_grad():
    refined = refiner.refine_latents(
        latents, prompt_text, fps=FPS, sink_size=1, seed=42, block_size=3, kv_max_frames=11,
    )
print(f"refined latents shape={tuple(refined.shape)}  peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
del refiner
gc.collect()
torch.cuda.empty_cache()

# --- decode refined latents with the causal VAE ------------------------------
from diffusion.model.ltx2.causal_vae import AutoencoderKLCausalLTX2Video  # noqa: E402
from diffusion.model.ltx2.streaming_decoder import CausalVaeStreamingDecoder  # noqa: E402

with torch.device("cpu"):
    causal_vae = AutoencoderKLCausalLTX2Video.from_pretrained(
        str(STREAM_SNAP / "ltx2_causal_vae"), torch_dtype=DTYPE, local_files_only=True
    ).to(DEVICE).eval()
dec = CausalVaeStreamingDecoder(causal_vae)
dec.reset()
with torch.no_grad():
    pix = dec.decode_chunk(refined.to(DEVICE, causal_vae.dtype))  # (B,3,T,H,W) [-1,1]
pix = pix[0].permute(1, 2, 3, 0)
frames = ((pix.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8).cpu().numpy()
print(f"decoded refined frames: shape={frames.shape} min={frames.min()} max={frames.max()} mean={frames.mean():.1f}")

# --- write outputs -----------------------------------------------------------
import cv2  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_DIR / f"{DEMO}_refined_frames.npy", frames)
mp4 = OUT_DIR / f"{DEMO}_refined.mp4"
h, w = frames.shape[1:3]
writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
try:
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
finally:
    writer.release()
n = len(frames)
for tag, idx in (("start", 0), ("5s", min(80, n - 1)), ("10s", min(160, n - 1)), ("last", n - 1)):
    Image.fromarray(frames[idx]).save(OUT_DIR / f"{DEMO}_refined_{tag}.png")
print(f"wrote {mp4} ({n} frames, {w}x{h} @ {FPS}fps) + start/5s/10s/last PNGs")
