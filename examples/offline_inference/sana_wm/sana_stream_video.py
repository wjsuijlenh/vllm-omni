# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1: run OUR SANA-WM AR pipeline with the STREAMING *student* weights.

Same driver shape as ``sana_ar_video.py``, but instead of the chunk_causal teacher
it loads the distilled self-forcing student from ``SANA-WM_streaming/sana_dit/model.pt``
and runs the existing ``_run_native_backend_ar`` rollout with the student's recipe:

  * weights   -- ``ckpt["generator"]``, ``model.`` prefix stripped, ``pos_embed``
                 dropped -> an EXACT key match to the teacher safetensors the port
                 already loads (Phase-0 finding), so no remapping.
  * schedule  -- the distilled 4-step ``denoising_step_list`` [1000,960,889,727,0]
                 (``sana_wm_ar_denoising_step_list``) instead of uniform steps.
  * sink      -- the WHOLE first chunk (``sana_wm_ar_sink_first_chunk=True``),
                 matching upstream ``SelfForcingFlowEulerCamCtrl``.
  * cfg       -- 1.0 (the distilled student is trained for cfg=1).
  * VAE       -- still the bidirectional LTX-2 VAE (Phase 3 swaps the causal VAE);
                 the encoder latent contract matches, so decode is coherent here.

The bet: same architecture + student weights + distilled schedule + chunk-0 sink
== the streaming behaviour, produced by the PORT. If the car stays coherent past
the ~5-10s point where the teacher melts, Phase 1 is validated.

Env knobs mirror sana_ar_video.py (SANA_AR_FRAMES/HEIGHT/WIDTH/CHUNK/CFG/
KV_WINDOW/SEED/OUT_DIR/FPS/DEMO). Run through gpuq with the Triton CPATH set.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import SanaWmPipeline
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmTransformer3DModel

torch.manual_seed(int(os.environ.get("SANA_AR_SEED", "42")))

_VLLM_CFG_CTX = set_current_vllm_config(VllmConfig())
_VLLM_CFG_CTX.__enter__()

DEVICE = torch.device("cuda")
DTYPE = torch.float32 if os.environ.get("SANA_AR_DTYPE", "bf16").lower() in ("fp32", "float32") else torch.bfloat16

HUB = Path.home() / ".cache/huggingface/hub"
CC_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_chunk_causal/snapshots/*/"))[0])
BIDI_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_bidirectional/snapshots/*/"))[0])
STREAM_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_streaming/snapshots/*/"))[0])
STREAM_DIT_PT = STREAM_SNAP / "sana_dit/model.pt"
ASSETS = Path(os.environ.get("SANA_WM_ASSETS", "NVlabs-Sana/asset/sana_wm"))
DEMO = os.environ.get("SANA_AR_DEMO", "demo_0")

FRAMES = int(os.environ.get("SANA_AR_FRAMES", "161"))
HEIGHT = int(os.environ.get("SANA_AR_HEIGHT", "704"))
WIDTH = int(os.environ.get("SANA_AR_WIDTH", "1280"))
CHUNK = int(os.environ.get("SANA_AR_CHUNK", "3"))
CFG = float(os.environ.get("SANA_AR_CFG", "1.0"))
# Match upstream's stage-1 sliding window. SelfForcingFlowEulerCamCtrl keeps
# ``num_cached_blocks`` previous chunks (streaming default 2) plus the chunk-0
# sink. With uniform chunk size that is num_cached_blocks * CHUNK latent frames,
# and because chunk 0 absorbs the remainder (sink = 4, later chunks = 3), a raw
# frame window of that size reproduces upstream's chunk-granular eviction
# exactly. Earlier we ran a longer window (10) which is off-recipe: the student
# was distilled at num_cached_blocks=2, so 10 feeds it a longer-than-trained
# context and shifts the morph pace away from the reference.
NUM_CACHED_BLOCKS = int(os.environ.get("SANA_AR_NUM_CACHED_BLOCKS", "2"))
KV_WINDOW = int(os.environ.get("SANA_AR_KV_WINDOW", str(NUM_CACHED_BLOCKS * CHUNK)))
SEED = int(os.environ.get("SANA_AR_SEED", "42"))
STEP_LIST = [int(t) for t in os.environ.get("SANA_AR_STEP_LIST", "1000,960,889,727,0").split(",")]
OUT_DIR = Path(os.environ.get("SANA_AR_OUT_DIR", "outputs/sana_stream_video"))
FPS = int(os.environ.get("SANA_AR_FPS", "16"))
# Phase 3: SANA_AR_VAE=causal decodes the rollout latents with the streaming
# causal VAE (fixes the bidirectional VAE's temporal-tile banding). The causal
# and bidirectional VAEs share latents_mean/std/scaling (verified max|Δ|=0), so
# the bidirectional first-frame encode stays valid and no re-normalization is
# needed. SANA_AR_VAE=bidi keeps the Phase-1 bidirectional decode.
VAE_MODE = os.environ.get("SANA_AR_VAE", "causal").lower()
SUFFIX = "_stream" + ("_causal" if VAE_MODE == "causal" else "_bidi")

print(f"=== SANA-WM STREAMING student: {DEMO} frames={FRAMES} {WIDTH}x{HEIGHT} chunk={CHUNK} "
      f"cfg={CFG} kv_window={KV_WINDOW} steps={STEP_LIST} ===")
latent_frames = (FRAMES - 1) // 8 + 1
print(f"latent_frames={latent_frames}")

cfg = SanaWmConfig.from_yaml(CC_SNAP / "config.yaml")
print(f"config: num_blocks={cfg.num_blocks} hidden={cfg.hidden_size} softmax_every_n={cfg.softmax_every_n} "
      f"flow_shift={cfg.inference_flow_shift} (bypassed by explicit step list)")

pipe = SanaWmPipeline(od_config=None)
pipe.sana_wm_config = cfg
pipe.od_config = SimpleNamespace(dtype=DTYPE, model=None, revision=None, quantization_config=None)

# --- load the STREAMING student weights ---------------------------------------
pipe.transformer = SanaWmTransformer3DModel(config=cfg, quant_config=None, prefix="transformer")
print(f"loading student generator from {STREAM_DIT_PT} ...")
_ck = torch.load(str(STREAM_DIT_PT), map_location="cpu", mmap=True, weights_only=False)
_gen = _ck["generator"]
student_sd = {}
for k, v in _gen.items():
    if "pos_embed" in k:
        continue
    student_sd[k[len("model."):] if k.startswith("model.") else k] = v
print(f"student state_dict: {len(student_sd)} tensors (pos_embed dropped)")
loaded = pipe.transformer.load_weights(student_sd.items())
want = set(dict(pipe.transformer.named_parameters()).keys())
got = {n.removeprefix("transformer.") for n in loaded}
missing = sorted(p for p in want if p not in got and p not in loaded)
print(f"transformer weights: loaded={len(loaded)}  missing={len(missing)}")
for m in missing[:10]:
    print("   MISSING", m)
pipe.transformer = pipe.transformer.to(device=DEVICE, dtype=DTYPE).eval()
pipe.transformer.config = cfg

# --- VAE (bidirectional for Phase 1; Phase 3 swaps causal) --------------------
from diffusers import AutoencoderKLLTX2Video  # noqa: E402

with torch.device("cpu"):
    vae = AutoencoderKLLTX2Video.from_pretrained(
        str(BIDI_SNAP), subfolder="vae", torch_dtype=DTYPE, local_files_only=True
    ).to(DEVICE)
vae.enable_tiling()
vae.use_framewise_encoding = True
vae.use_framewise_decoding = True
vae.tile_sample_stride_num_frames = int(getattr(vae.config, "tile_sample_stride_num_frames", 64))
vae.tile_sample_min_num_frames = int(getattr(vae.config, "tile_sample_min_num_frames", 96))
pipe.vae = vae
print("VAE preloaded from bidirectional snapshot")

# Phase 3: causal VAE for streaming-correct decode (loaded from the streaming
# snapshot's ltx2_causal_vae/). Kept separate from pipe.vae so the first-frame
# encode still uses the (shared-normalization) bidirectional encoder.
causal_dec = None
if VAE_MODE == "causal":
    from diffusion.model.ltx2.causal_vae import AutoencoderKLCausalLTX2Video  # noqa: E402
    from diffusion.model.ltx2.streaming_decoder import CausalVaeStreamingDecoder  # noqa: E402

    causal_vae_dir = STREAM_SNAP / "ltx2_causal_vae"
    with torch.device("cpu"):
        causal_vae = AutoencoderKLCausalLTX2Video.from_pretrained(
            str(causal_vae_dir), torch_dtype=DTYPE, local_files_only=True
        ).to(DEVICE).eval()
    causal_dec = CausalVaeStreamingDecoder(causal_vae)
    print(f"causal VAE loaded from {causal_vae_dir} (decoder.is_causal={causal_vae.decoder.is_causal})")

# --- inputs -------------------------------------------------------------------
image = Image.open(ASSETS / f"{DEMO}.png").convert("RGB")
prompt_text = (ASSETS / f"{DEMO}.txt").read_text().strip()
poses = np.load(ASSETS / f"{DEMO}_pose.npy")
intrinsics = np.load(ASSETS / f"{DEMO}_intrinsics.npy")
print(f"assets: image={image.size} poses={poses.shape} K={intrinsics.shape}")

prompt = {"prompt": prompt_text, "multi_modal_data": {"image": image}}
payload = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_frames": FRAMES,
    "camera": {"poses": poses[:FRAMES]},
    "intrinsics": intrinsics[:FRAMES],
    "session_id": f"sana-wm-stream-{DEMO}",
}
sampling_params = SimpleNamespace(
    height=HEIGHT,
    width=WIDTH,
    num_frames=FRAMES,
    num_inference_steps=len(STEP_LIST) - 1,
    seed=SEED,
    guidance_scale_provided=CFG > 1.0,
    guidance_scale=CFG,
    extra_args={
        "sana_wm_ar_chunk_size": CHUNK,
        "sana_wm_ar_kv_window_frames": KV_WINDOW,
        "sana_wm_ar_denoising_step_list": STEP_LIST,
        "sana_wm_ar_sink_first_chunk": True,
        "sana_wm_ar_conv_carry": True,
        # causal mode returns latents; we decode them with the causal VAE below.
        "sana_wm_output_type": "latent" if VAE_MODE == "causal" else "np",
        "sana_wm_native_max_tokens": 10_000_000,
    },
)

torch.cuda.reset_peak_memory_stats()
print("\n=== running _run_native_backend_ar (streaming student) ===")
with torch.no_grad():
    out = pipe._run_native_backend_ar(prompt=prompt, payload=payload, sampling_params=sampling_params)
print("custom_output:", out.custom_output)
print(f"peak CUDA allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB  "
      f"reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

if VAE_MODE == "causal":
    # out.output is the normalized latent tensor (1, 128, F, H, W). Decode it
    # causally frame-by-frame (single call == chunk-by-chunk, bit-identical).
    latents = out.output
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    latents = torch.as_tensor(latents, device=DEVICE, dtype=causal_dec.vae.dtype)
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if os.environ.get("SANA_AR_SAVE_LATENTS") == "1":
        # Persist the normalized stage-1 latents so a separate job can refine them
        # (the 17B refiner + this DiT together would OOM the card).
        lat_path = OUT_DIR / f"{DEMO}{SUFFIX}_latents.pt"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(latents.cpu(), lat_path)
        print(f"saved stage-1 latents -> {lat_path}")
    print(f"causal-decoding latents shape={tuple(latents.shape)} ...")
    causal_dec.reset()
    with torch.no_grad():
        pix = causal_dec.decode_chunk(latents)  # (B, 3, T, H, W) in [-1, 1]
    pix = pix[0].permute(1, 2, 3, 0)  # (T, H, W, 3)
    frames = ((pix.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8).cpu().numpy()
else:
    video = out.output
    if isinstance(video, (list, tuple)):
        video = video[0]
    frames = np.asarray(video)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
print(f"decoded frames: shape={frames.shape} dtype={frames.dtype} "
      f"min={frames.min()} max={frames.max()} mean={frames.mean():.1f}")

import cv2  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_DIR / f"{DEMO}{SUFFIX}_frames.npy", frames)
mp4 = OUT_DIR / f"{DEMO}{SUFFIX}.mp4"
h, w = frames.shape[1:3]
writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (w, h))
if not writer.isOpened():
    raise RuntimeError(f"cannot open video writer for {mp4}")
try:
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
finally:
    writer.release()
# key frames across the clip for a quick eyeball
n = len(frames)
for tag, idx in (("start", 0), ("5s", min(80, n - 1)), ("10s", min(160, n - 1)), ("last", n - 1)):
    Image.fromarray(frames[idx]).save(OUT_DIR / f"{DEMO}{SUFFIX}_{tag}.png")
print(f"\nwrote {mp4} ({n} frames, {w}x{h} @ {FPS}fps) + start/5s/10s/last PNGs")
