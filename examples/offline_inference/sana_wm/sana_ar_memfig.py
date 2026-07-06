# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the SANA-WM AR memory profile as a figure from the timeline JSONs.

Reads every ``*_timeline.json`` under the profile dir and draws a 3x2 panel:
  (A) total per-session state (MB) vs chunk -- bounded plateaus, unbounded grows
  (B) softmax windowed KV vs chunk, with the flat GDN/conv FixedState references
  (C) CUDA reserved vs allocated vs chunk -- the reserved-over-allocated gap
  (D) fragmentation (inactive_split) vs chunk
  (E) end-state composition of RESERVED (from the *_mem.pickle allocator
      snapshots): weights + text-encoder floor, tracked session state, and the
      caching-allocator reserved-but-free slack -- shows panel A is a thin slice
  (F) unbounded run: reserved = live-allocated + allocator slack, per chunk,
      with the (tiny) tracked session state overlaid

Pure post-processing (no GPU / model); safe to run on the host anytime.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROFILE_DIR = Path(os.environ.get("SANA_AR_PROFILE_DIR", "artifacts/sana-wm-memprofile"))
_GB = 2**30


def _load() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(str(PROFILE_DIR / "*_timeline.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        mode = d.get("manager_mode", "on")
        d["_label"] = f"w{cfg['kv_window_frames']}·{cfg['frames']}f·{mode}"
        d["_window"] = cfg["kv_window_frames"]
        d["_mode"] = mode
        d["_rows"] = [r for r in d["timeline"] if r["chunk_boundary"] != 999]
        runs.append(d)
    return runs


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = [r["chunk_boundary"] for r in rows]
    if key == "softmax_kv":
        ys = [(r["components_kb"].get("softmax_kv_main", 0) + r["components_kb"].get("softmax_kv_cam", 0)) / 1024
              for r in rows]
    elif key == "gdn":
        ys = [r["components_kb"].get("gdn_fixedstate", 0) / 1024 for r in rows]
    elif key == "conv":
        ys = [r["components_kb"].get("conv_fixedstate", 0) / 1024 for r in rows]
    else:
        ys = [r[key] for r in rows]
    return xs, ys


def _style(run: dict) -> dict:
    color = "tab:red" if run["_window"] == 0 else ("tab:green" if run["_mode"] == "off" else "tab:blue")
    ls = "--" if run["_mode"] == "off" else "-"
    lw = 2.4 if run["config"]["frames"] >= 300 else 1.6
    return {"color": color, "linestyle": ls, "linewidth": lw, "marker": "o", "markersize": 3}


def _decomp(tag: str) -> dict[str, float] | None:
    """Bucket a *_mem.pickle allocator snapshot into GB by allocation site."""
    path = PROFILE_DIR / f"demo_0_{tag}_mem.pickle"
    if not path.exists():
        return None
    snap = pickle.load(open(path, "rb"))
    reserved = sum(s["total_size"] for s in snap["segments"])
    b: dict[str, float] = defaultdict(float)
    live = 0
    for s in snap["segments"]:
        for blk in s["blocks"]:
            if blk.get("state") != "active_allocated":
                continue
            sz = blk["size"]
            live += sz
            frames = blk.get("frames") or []
            sig = " ".join(f"{f.get('filename', '')}:{f.get('name', '')}" for f in frames)
            if not frames:
                k = "weights"
            elif "_ensure_stage1_text_encoder" in sig:
                k = "text_encoder"
            elif "commit_softmax_kv" in sig:
                k = "softmax_kv"
            elif "commit_conv_state" in sig:
                k = "conv"
            elif "create_state" in sig:
                k = "gdn"
            else:
                k = "other"
            b[k] += sz
    return {k: v / _GB for k, v in b.items()} | {"reserved": reserved / _GB, "slack": (reserved - live) / _GB}


def main() -> None:
    runs = _load()
    if not runs:
        raise SystemExit(f"no *_timeline.json under {PROFILE_DIR}")
    print(f"loaded {len(runs)} runs: {[r['_label'] for r in runs]}")

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("SANA-WM autoregressive session: GPU memory-allocation profile (streaming student, conv-carry on)\n"
                 "(demo_0, 704x1280, cfg 1.0, window 6 = num_cached_blocks 2, output_type=latent)", fontsize=12)
    axA, axB, axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), \
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axE, axF = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    for run in runs:
        st = _style(run)
        xs, ys = _series(run["_rows"], "session_total_mb")
        axA.plot(xs, ys, label=run["_label"], **st)
        xs, ys = _series(run["_rows"], "softmax_kv")
        axB.plot(xs, ys, label=run["_label"], **st)
        xs, yf = _series(run["_rows"], "inactive_split_mb")
        axD.plot(xs, yf, label=run["_label"], **st)

    ref = max(runs, key=lambda r: len(r["_rows"]))["_rows"]
    gx, gy = _series(ref, "gdn")
    axB.plot(gx, gy, color="0.4", linestyle=":", linewidth=1.6, label="GDN FixedState (flat, 14.5 MB)")
    cx, cy = _series(ref, "conv")
    axB.plot(cx, cy, color="tab:purple", linestyle=":", linewidth=1.8, label="conv FixedState (flat, 413.6 MB)")

    on_runs = sorted([r for r in runs if r["_mode"] == "on"], key=lambda r: -r["config"]["frames"])
    for run in on_runs[:2] if on_runs else []:
        c = "tab:red" if run["_window"] == 0 else "tab:blue"
        xs, yr = _series(run["_rows"], "reserved_mb")
        _, ya = _series(run["_rows"], "alloc_mb")
        tag = "unbounded" if run["_window"] == 0 else "bounded"
        axC.plot(xs, yr, color=c, linestyle="-", marker="o", markersize=3, label=f"{tag} reserved")
        axC.plot(xs, ya, color=c, linestyle="--", marker="x", markersize=4, label=f"{tag} allocated")

    axA.set(title="(A) per-session state total", xlabel="chunk boundary", ylabel="MB")
    axB.set(title="(B) growing softmax KV vs fixed GDN/conv components", xlabel="chunk boundary", ylabel="MB")
    axC.set(title="(C) CUDA reserved vs allocated", xlabel="chunk boundary", ylabel="MB")
    axD.set(title="(D) fragmentation (inactive_split)", xlabel="chunk boundary", ylabel="MB")

    # (E) end-state composition of reserved, bounded vs unbounded (from pickles).
    bd, ud = _decomp("stream_cfg1_w6_on"), _decomp("stream_cfg1_w0_on")
    if bd and ud:
        # stack order (bottom->top): weights, text encoder, session-fixed, session-KV, slack
        order = [("weights", "DiT weights", "0.55"),
                 ("text_encoder", "Gemma text encoder", "tab:gray"),
                 ("_sessfix", "conv+GDN state (tracked)", "tab:purple"),
                 ("softmax_kv", "softmax KV window (tracked, panel A)", "tab:blue"),
                 ("slack", "allocator reserved-but-free", "tab:red")]
        labels = ["bounded\n(w6)", "unbounded\n(w0)"]
        for d in (bd, ud):
            d["_sessfix"] = d.get("conv", 0) + d.get("gdn", 0)
        bottoms = [0.0, 0.0]
        for key, lab, col in order:
            vals = [bd.get(key, 0.0), ud.get(key, 0.0)]
            axE.bar(labels, vals, bottom=bottoms, color=col, label=lab, edgecolor="white", linewidth=0.5)
            bottoms = [bottoms[i] + vals[i] for i in range(2)]
        for i, d in enumerate((bd, ud)):
            axE.text(i, d["reserved"] + 0.6, f"{d['reserved']:.1f} GB\nreserved", ha="center", fontsize=8)
        axE.set(title="(E) what the reserved GB actually are (end state)", ylabel="GB")
        axE.set_ylim(0, max(bd["reserved"], ud["reserved"]) * 1.18)
        axE.legend(fontsize=6.5, loc="upper left")
    else:
        axE.text(0.5, 0.5, "pickles not found", ha="center")

    # (F) unbounded: reserved = live-allocated + allocator slack, per chunk.
    ub = next((r for r in runs if r["_window"] == 0 and r["_mode"] == "on"), None)
    if ub:
        xs = [r["chunk_boundary"] for r in ub["_rows"]]
        alloc = [r["alloc_mb"] / 1024 for r in ub["_rows"]]
        slack = [(r["reserved_mb"] - r["alloc_mb"]) / 1024 for r in ub["_rows"]]
        sess = [r["session_total_mb"] / 1024 for r in ub["_rows"]]
        axF.stackplot(xs, alloc, slack, labels=["live allocated (weights+TE+state+transient)",
                                                 "allocator reserved-but-free"],
                      colors=["tab:blue", "tab:red"], alpha=0.65)
        axF.plot(xs, sess, color="k", lw=2, marker="o", markersize=3, label="tracked session state (panel A)")
        axF.set(title="(F) unbounded: reserved = live + allocator slack (GB)", xlabel="chunk boundary", ylabel="GB")
        axF.legend(fontsize=6.5, loc="upper left")

    for ax in (axA, axB, axC, axD, axF):
        ax.grid(True, alpha=0.3)
    for ax in (axA, axB, axC, axD):
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PROFILE_DIR / "memprofile_figure.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
