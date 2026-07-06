# SANA-WM streaming (autoregressive) offline examples

Drivers for the SANA-WM **streaming / autoregressive (AR)** world-model path: generate a
video chunk-by-chunk, refine it, and profile the per-session GPU memory the AR rollout uses.
The model port and the session-memory manager these scripts exercise live in
`vllm_omni/diffusion/models/sana_wm/` and `vllm_omni/experimental/world_models/`.

| Script | What it does |
|---|---|
| `sana_stream_video.py` | Stage-1 AR rollout with the distilled **streaming student**, decode with the causal VAE → `*_stream_causal.mp4` (and, with `SANA_AR_SAVE_LATENTS=1`, dumps `*_stream_causal_latents.pt`). |
| `sana_refine_decode.py` | Runs the 17B LTX-2 refiner on the dumped stage-1 latents → `*_refined.mp4`. Separate job: the DiT and the refiner do not fit on one card together. |
| `sana_ar_memprofile.py` | Drives one AR session chunk-by-chunk and records per-chunk `torch.cuda` memory stats + the session's per-object byte sizes → `*_timeline.json`. Compares the central `SessionMemoryManager` against a `NullSessionManager` baseline (`SANA_AR_MANAGER=on|off`). |
| `sana_ar_memfig.py` | CPU-only. Reads the `*_timeline.json` files and renders the memory-profile figure. |

## Prerequisites

- vLLM pinned to the version vllm-omni targets (`pip install vllm==0.23.0`).
- Hugging Face checkpoints (gated — set `HF_TOKEN`, then `hf download`):
  `Efficient-Large-Model/SANA-WM_streaming` (student DiT, causal VAE, 17B refiner, text
  encoder). `SANA-WM_bidirectional` is only needed for the non-AR reference video (upstream
  `NVlabs/Sana` script, not these drivers).
- Demo assets (first frame / prompt / camera / intrinsics): the five `demo_{0..4}.*` bundles
  from `NVlabs/Sana` at `asset/sana_wm/`. Point the scripts at them with
  **`SANA_WM_ASSETS=/path/to/Sana/asset/sana_wm`** (default: `NVlabs-Sana/asset/sana_wm`
  relative to your working directory).
- The Gated-DeltaNet block compiles a Triton kernel that needs the Python headers:
  `export CPATH=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")`.

## Quickstart

```bash
export SANA_WM_ASSETS=/path/to/Sana/asset/sana_wm
export CPATH=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")

# stage-1 AR video (+ dump latents for the refiner)
SANA_AR_DEMO=demo_0 SANA_AR_FRAMES=313 SANA_AR_VAE=causal SANA_AR_SAVE_LATENTS=1 \
  python sana_stream_video.py

# refine the dumped latents
SANA_AR_DEMO=demo_0 python sana_refine_decode.py

# profile the AR session's GPU memory (validated recipe: streaming, cfg 1.0, KV window 6)
SANA_AR_STREAMING=1 SANA_AR_CFG=1.0 SANA_AR_KV_WINDOW=6 SANA_AR_FRAMES=313 \
  SANA_AR_MANAGER=on python sana_ar_memprofile.py

# render the figure from the timeline JSONs (no GPU)
python sana_ar_memfig.py
```

Key knobs: `SANA_AR_DEMO`, `SANA_AR_FRAMES` (`24k+1`; 313/961/4801 = 13/40/200 chunks),
`SANA_AR_KV_WINDOW` (6 = bounded sliding window, 0 = unbounded), `SANA_AR_MANAGER=on|off`,
`SANA_AR_CFG`, `SANA_AR_MEM_LIMIT_GB` (hard cap), `SANA_AR_OUT_DIR` / `SANA_AR_PROFILE_DIR`.
