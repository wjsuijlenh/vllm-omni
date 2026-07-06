# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step 2 deliverable: profile GPU memory-allocation patterns across a growing
SANA-WM AR session, and break the per-session state down by component.

Rides the same real AR sampler vehicle as ``sana_ar_video.py``
(``SanaWmPipeline._run_native_backend_ar``), but instrumented:

* Per-chunk timeline -- a probe on ``SanaWmArStateAdapter.chunk_index`` (set once
  per chunk, after all state commits) records ``torch.cuda.memory_stats`` (current
  + peak allocated/reserved, ``inactive_split_bytes`` = fragmentation,
  ``num_alloc_retries``, ``active_bytes``) and resets the peak so each row is the
  peak *within that chunk*.
* Per-state-component breakdown -- from the session's own objects
  (``session.names()`` + ``obj.nbytes``), bucketed into GDN ``FixedState``, softmax
  ``PagedKV`` (main + camera streams), text ``EncodeOnceKV``, warm-up
  ``LatentBuffer``, conv ``FixedState``. Shows which components are FIXED (GDN,
  text), which are BOUNDED-then-plateau (softmax window), which are tiny (warmup).
* Allocation snapshot -- ``_record_memory_history`` around the run,
  ``_dump_snapshot`` for the PyTorch memory visualizer.

Output_type=latent (no VAE decode) so the timeline isolates the rollout's session
state from the one-shot decode spike; we already have pixels from Step 1.

The bounded-vs-unbounded contrast (RFC #4480 byte-budget motivation) is driven by
``SANA_AR_KV_WINDOW``: 10 (bounded sink+window) vs 0 (unbounded full history).

Env: SANA_AR_FRAMES (default 169 -> ~7 chunks @ chunk 3), SANA_AR_STEPS (default 8,
memory pattern is step-independent), SANA_AR_KV_WINDOW (default 10), plus the
SANA_AR_HEIGHT/WIDTH/CHUNK/CFG/SEED/DEMO knobs from sana_ar_video.py. Run via gpuq
with the Triton CPATH env.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import SanaWmPipeline
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmTransformer3DModel
from vllm_omni.experimental.world_models.adapters.state_sana_wm_ar_adapter import SanaWmArStateAdapter
from vllm_omni.experimental.world_models.memory.manager import SessionMemory, SessionMemoryManager


class NullSessionManager:
    """Ad-hoc 'manager-off' baseline: bare per-session stores, no central budget.

    Represents the status quo where each model owns its per-session state directly
    with no cross-session LRU table, byte budget, or accounting. It still returns
    the *same* ``SessionMemory`` for a given id (so the pipeline's adapter and our
    final-snapshot adapter see one store), because that is what any per-model store
    does too -- the point being measured is whether routing through the central
    ``SessionMemoryManager`` costs any GPU bytes over this bare storage (it does
    not: both hold the identical tensors, in place).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: str | None) -> SessionMemory:
        key = str(session_id or "default")
        session = self._sessions.get(key)
        if session is None:
            session = SessionMemory()
            self._sessions[key] = session
        return session

    def stats(self) -> dict:
        return {
            "mode": "off_adhoc",
            "sessions": len(self._sessions),
            "total_nbytes": sum(s.nbytes for s in self._sessions.values()),
            "note": "no central byte budget / LRU (ad-hoc per-model storage)",
        }

torch.manual_seed(int(os.environ.get("SANA_AR_SEED", "42")))
_VLLM_CFG_CTX = set_current_vllm_config(VllmConfig())
_VLLM_CFG_CTX.__enter__()

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Optional HARD GPU-memory cap: forces the caching allocator to OOM rather than
# reserve past the cap, so it must release its free/cached slack under pressure.
# Proves whether the run's true requirement is the peak-LIVE allocation (not the
# inflated reserved). SANA_AR_MEM_LIMIT_GB is interpreted in GiB.
_MEM_LIMIT_GB = os.environ.get("SANA_AR_MEM_LIMIT_GB")
if _MEM_LIMIT_GB:
    _limit_bytes = float(_MEM_LIMIT_GB) * 1024**3
    _total = torch.cuda.get_device_properties(0).total_memory
    _frac = _limit_bytes / _total
    torch.cuda.set_per_process_memory_fraction(_frac, 0)
    torch.cuda.empty_cache()
    print(f"=== HARD GPU MEMORY CAP: {float(_MEM_LIMIT_GB):.2f} GiB "
          f"({_limit_bytes/1e9:.2f} GB) = fraction {_frac:.4f} of {_total/1024**3:.1f} GiB total ===", flush=True)

HUB = Path.home() / ".cache/huggingface/hub"
CC_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_chunk_causal/snapshots/*/"))[0])
BIDI_SNAP = Path(glob.glob(str(HUB / "models--Efficient-Large-Model--SANA-WM_bidirectional/snapshots/*/"))[0])
DIT_SF = next(CC_SNAP.glob("dit/*.safetensors"))
ASSETS = Path(os.environ.get("SANA_WM_ASSETS", "NVlabs-Sana/asset/sana_wm"))
DEMO = os.environ.get("SANA_AR_DEMO", "demo_0")

FRAMES = int(os.environ.get("SANA_AR_FRAMES", "169"))
HEIGHT = int(os.environ.get("SANA_AR_HEIGHT", "704"))
WIDTH = int(os.environ.get("SANA_AR_WIDTH", "1280"))
CHUNK = int(os.environ.get("SANA_AR_CHUNK", "3"))
STEPS = int(os.environ.get("SANA_AR_STEPS", "8"))
CFG = float(os.environ.get("SANA_AR_CFG", "5.0"))
KV_WINDOW = int(os.environ.get("SANA_AR_KV_WINDOW", "10"))
SEED = int(os.environ.get("SANA_AR_SEED", "42"))
OUT_DIR = Path(os.environ.get("SANA_AR_PROFILE_DIR", "artifacts/sana-wm-memprofile"))
MANAGER_MODE = os.environ.get("SANA_AR_MANAGER", "on").strip().lower()  # on | off
CONV_CARRY = os.environ.get("SANA_AR_CONV_CARRY", "1") != "0"  # temporal-conv cross-chunk carry (working AR = on)
TAG = os.environ.get("SANA_AR_TAG", f"window{KV_WINDOW}")

latent_frames = (FRAMES - 1) // 8 + 1
print(f"=== SANA-WM AR memprofile [{TAG}]: {DEMO} frames={FRAMES} ({latent_frames} latent) "
      f"{WIDTH}x{HEIGHT} chunk={CHUNK} steps={STEPS} cfg={CFG} kv_window={KV_WINDOW} conv_carry={CONV_CARRY} ===")

# ---- per-state-component byte breakdown from the live session ----------------
def _component_breakdown(session: object) -> dict[str, dict[str, int]]:
    """Bucket the session's MemoryObjects by kind -> {bytes, count}."""
    buckets: dict[str, dict[str, int]] = {}
    for name in session.names():  # type: ignore[attr-defined]
        obj = session.get(name)  # type: ignore[attr-defined]
        nbytes = int(getattr(obj, "nbytes", 0))
        if name.startswith("gdn/"):
            key = "gdn_fixedstate"
        elif name.startswith("softmax_kv/main/"):
            key = "softmax_kv_main"
        elif name.startswith("softmax_kv/cam/"):
            key = "softmax_kv_cam"
        elif name.startswith("text/"):
            key = "text_encodeonce"
        elif name.startswith("conv/"):
            key = "conv_fixedstate"
        elif name == "warmup_latents":
            key = "warmup_latentbuffer"
        else:
            key = "other"
        b = buckets.setdefault(key, {"bytes": 0, "count": 0})
        b["bytes"] += nbytes
        b["count"] += 1
    return buckets


# ---- per-chunk probe on the adapter's chunk_index setter ---------------------
TIMELINE: list[dict] = []
_MB = 1024 * 1024


def _snapshot(adapter: SanaWmArStateAdapter, chunk_boundary: int) -> None:
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    session = adapter._session  # noqa: SLF001 -- profiling the live session by design
    comp = _component_breakdown(session)
    row = {
        # chunk_boundary == k means "state after committing chunk k-1"; 0 = initial
        # (post create_state, GDN zeros allocated, softmax windows empty).
        "chunk_boundary": int(chunk_boundary),
        "alloc_mb": round(torch.cuda.memory_allocated() / _MB, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / _MB, 2),
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / _MB, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / _MB, 2),
        # fragmentation: reserved-but-unusable split blocks; retries: alloc pressure
        "inactive_split_mb": round(stats.get("inactive_split_bytes.all.current", 0) / _MB, 2),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "active_mb": round(stats.get("active_bytes.all.current", 0) / _MB, 2),
        "session_total_mb": round(sum(v["bytes"] for v in comp.values()) / _MB, 4),
        "components_kb": {k: round(v["bytes"] / 1024, 2) for k, v in sorted(comp.items())},
        "component_counts": {k: v["count"] for k, v in sorted(comp.items())},
    }
    TIMELINE.append(row)
    print(f"  [chunk boundary {chunk_boundary:>2}] alloc={row['alloc_mb']:.0f}MB "
          f"reserved={row['reserved_mb']:.0f}MB peak_chunk={row['peak_alloc_mb']:.0f}MB "
          f"frag={row['inactive_split_mb']:.0f}MB  session_state={row['session_total_mb']:.3f}MB "
          f"[sm_main={row['components_kb'].get('softmax_kv_main', 0):.0f}KB "
          f"sm_cam={row['components_kb'].get('softmax_kv_cam', 0):.0f}KB "
          f"gdn={row['components_kb'].get('gdn_fixedstate', 0):.0f}KB]")
    # Reset peak so the next row's peak is the peak *within the coming chunk*.
    torch.cuda.reset_peak_memory_stats()


# Patch the class property so the setter also snapshots (fires once per chunk).
_orig_prop = SanaWmArStateAdapter.chunk_index


def _probed_setter(self: SanaWmArStateAdapter, value: int) -> None:
    _orig_prop.fset(self, value)
    _snapshot(self, value)


SanaWmArStateAdapter.chunk_index = property(_orig_prop.fget, _probed_setter)  # type: ignore[assignment]

# ---- build the pipeline (same assembly as sana_ar_video.py) ------------------
cfg = SanaWmConfig.from_yaml(CC_SNAP / "config.yaml")
num_heads = max(cfg.hidden_size // max(cfg.linear_head_dim, 1), 1)
head_dim = cfg.hidden_size // num_heads
gdn_layers = [i for i in range(cfg.num_blocks) if not (cfg.softmax_every_n > 0 and (i + 1) % cfg.softmax_every_n == 0)]
softmax_layers = [i for i in range(cfg.num_blocks) if i not in gdn_layers]
print(f"config: num_blocks={cfg.num_blocks} hidden={cfg.hidden_size} softmax_every_n={cfg.softmax_every_n} "
      f"num_heads={num_heads} head_dim={head_dim} | {len(gdn_layers)} GDN, {len(softmax_layers)} softmax blocks")

pipe = SanaWmPipeline(od_config=None)
pipe.sana_wm_config = cfg
pipe.transformer = SanaWmTransformer3DModel(config=cfg, quant_config=None, prefix="transformer")
# SANA_AR_STREAMING=1 profiles the distilled streaming STUDENT (same architecture,
# so state shapes are identical to the teacher) instead of the chunk_causal teacher.
STREAMING = os.environ.get("SANA_AR_STREAMING") == "1"
if STREAMING:
    _spt = next((HUB / "models--Efficient-Large-Model--SANA-WM_streaming").glob("snapshots/*/sana_dit/model.pt"))
    _ck = torch.load(str(_spt), map_location="cpu", mmap=True, weights_only=False)
    _sd = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in _ck["generator"].items()
        if "pos_embed" not in k
    }
    loaded = pipe.transformer.load_weights(_sd.items())
    print(f"transformer weights (STREAMING student): loaded={len(loaded)}")
else:
    loaded = pipe.transformer.load_weights(load_file(str(DIT_SF)).items())
    print(f"transformer weights (chunk_causal teacher): loaded={len(loaded)}")
pipe.transformer = pipe.transformer.to(device=DEVICE, dtype=DTYPE).eval()
pipe.transformer.config = cfg

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
pipe.vae = vae  # still needed for the first-frame encode (not for output decode)

# Inject a store we retain a handle to, for a final breakdown after the run.
# on  -> the central SessionMemoryManager (RFC #4480)
# off -> a NullSessionManager (ad-hoc per-model storage, no central budget/LRU)
MANAGER = SessionMemoryManager() if MANAGER_MODE == "on" else NullSessionManager()
pipe._ar_session_manager_obj = MANAGER
print(f"session store: {'SessionMemoryManager (on)' if MANAGER_MODE == 'on' else 'NullSessionManager (off/ad-hoc)'}")

# ---- inputs ------------------------------------------------------------------
image = Image.open(ASSETS / f"{DEMO}.png").convert("RGB")
prompt_text = (ASSETS / f"{DEMO}.txt").read_text().strip()
poses = np.load(ASSETS / f"{DEMO}_pose.npy")
intrinsics = np.load(ASSETS / f"{DEMO}_intrinsics.npy")
if FRAMES > len(poses):
    # The shipped demo trajectories are only ~321 frames; tile them to reach
    # minute-long clip lengths (961f/40 chunks). The GPU-memory allocation
    # pattern is content-independent (depends on sequence length + chunk
    # structure, not the camera values), so tiling is valid for this profile.
    n0 = len(poses)
    reps = -(-FRAMES // n0)  # ceil
    poses = np.concatenate([poses] * reps, axis=0)
    intrinsics = np.concatenate([intrinsics] * reps, axis=0)
    print(f"tiled camera trajectory {n0} -> {len(poses)} frames (reps={reps}) for FRAMES={FRAMES} "
          f"[content-independent memory profile]", flush=True)
prompt = {"prompt": prompt_text, "multi_modal_data": {"image": image}}
payload = {
    "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
    "camera": {"poses": poses[:FRAMES]}, "intrinsics": intrinsics[:FRAMES],
    "session_id": f"sana-wm-memprofile-{DEMO}-{TAG}",
}
sampling_params = SimpleNamespace(
    height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=STEPS, seed=SEED,
    guidance_scale_provided=CFG > 1.0, guidance_scale=CFG,
    extra_args={
        "sana_wm_ar_chunk_size": CHUNK,
        "sana_wm_ar_kv_window_frames": KV_WINDOW,
        "sana_wm_output_type": "latent",  # skip VAE decode: profile the rollout only
        "sana_wm_hash_prompt_fallback": False,
        "sana_wm_ar_conv_carry": CONV_CARRY,  # working AR carries the temporal-conv state
        "sana_wm_native_max_tokens": 100_000_000,
        # streaming student: distilled 4-step schedule + whole-first-chunk sink
        **(
            {
                "sana_wm_ar_denoising_step_list": [1000, 960, 889, 727, 0],
                "sana_wm_ar_sink_first_chunk": True,
            }
            if STREAMING
            else {}
        ),
    },
)

# ---- run under memory-history recording --------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
_hist = False
for _kw in ({"enabled": "all", "max_entries": 200_000}, {"max_entries": 200_000}, {"enabled": True}):
    try:
        torch.cuda.memory._record_memory_history(**_kw)
        _hist = True
        break
    except Exception as exc:  # noqa: BLE001
        _last_hist_exc = exc
if not _hist:
    print(f"(memory-history recording unavailable: {_last_hist_exc})")

torch.cuda.reset_peak_memory_stats()
print("\n=== running instrumented _run_native_backend_ar ===")
with torch.no_grad():
    out = pipe._run_native_backend_ar(prompt=prompt, payload=payload, sampling_params=sampling_params)
print("custom_output:", out.custom_output)

# Final snapshot after the last chunk (the last chunk sets chunk_index but does not
# commit, so this captures the settled end-of-session state explicitly).
_final_adapter = SanaWmArStateAdapter.from_config(payload["session_id"], MANAGER, cfg)
_snapshot(_final_adapter, chunk_boundary=999)

pickle_path = OUT_DIR / f"{DEMO}_{TAG}_mem.pickle"
if _hist:
    try:
        torch.cuda.memory._dump_snapshot(str(pickle_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"wrote allocation snapshot {pickle_path} (open in https://pytorch.org/memory_viz)")
    except Exception as exc:  # noqa: BLE001
        print(f"(snapshot dump failed: {exc})")

# ---- persist the timeline + a compact summary --------------------------------
spans_note = out.custom_output.get("sana_wm_ar_chunks")
summary = {
    "tag": TAG,
    "manager_mode": MANAGER_MODE,
    "conv_carry": CONV_CARRY,
    "config": {
        "demo": DEMO, "frames": FRAMES, "latent_frames": latent_frames, "height": HEIGHT, "width": WIDTH,
        "chunk_size": CHUNK, "steps": STEPS, "cfg": CFG, "kv_window_frames": KV_WINDOW,
        "num_blocks": cfg.num_blocks, "gdn_blocks": len(gdn_layers), "softmax_blocks": len(softmax_layers),
        "num_heads": num_heads, "head_dim": head_dim, "ar_chunks": spans_note,
        "spatial_tokens_per_frame": (HEIGHT // 32) * (WIDTH // 32),
    },
    "peak_alloc_mb_overall": round(torch.cuda.max_memory_allocated() / _MB, 2),
    "manager_stats": MANAGER.stats(),
    "timeline": TIMELINE,
}
json_path = OUT_DIR / f"{DEMO}_{TAG}_timeline.json"
json_path.write_text(json.dumps(summary, indent=2))
print(f"\nwrote timeline {json_path} ({len(TIMELINE)} snapshots)")

# ---- console summary: growth of each session-state component -----------------
print(f"\n=== session-state component growth [{TAG}] (KB) ===")
comp_keys = sorted({k for row in TIMELINE for k in row["components_kb"]})
header = "chunk  " + "  ".join(f"{k:>18}" for k in comp_keys) + "   total_MB"
print(header)
for row in TIMELINE:
    cells = "  ".join(f"{row['components_kb'].get(k, 0):>18.2f}" for k in comp_keys)
    label = "init" if row["chunk_boundary"] == 0 else ("final" if row["chunk_boundary"] == 999 else str(row["chunk_boundary"]))
    print(f"{label:>5}  {cells}   {row['session_total_mb']:>8.3f}")
print("\nGDN FixedState + text are FLAT (constant per block); softmax_kv grows then "
      "plateaus at the sink+window bound (or grows unbounded if kv_window=0).")
