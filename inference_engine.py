"""
inference_engine.py
───────────────────
Drives ProPainter inference via subprocess.

ProPainter is expected to be cloned at <project>/propainter/.
Model weights are loaded from <project>/model/ via a per-job symlink
(or directory junction on Windows) placed at propainter/weights before
each run, so the upstream code requires no changes.
"""

import os
import re
import sys
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OPTIONS = {
    "height":           -1,       # -1 = keep original
    "width":            -1,       # -1 = keep original
    "mask_dilation":    4,
    "ref_stride":       10,
    "neighbor_length":  10,
    "subvideo_length":  80,
    "raft_iter":        20,
    "fp16":             False,
    "resize_ratio":     1.0,
    "save_fps":         -1,       # -1 = use source fps
}


# ── Public API ────────────────────────────────────────────────────────────────

def run_propainter(
    video_path:      Path,
    mask_path:       Path,
    output_dir:      Path,
    model_dir:       Path,
    propainter_dir:  Path,
    options:         dict,
    progress_cb:     Optional[Callable[[int, str], None]] = None,
) -> Path:
    """
    Run ProPainter inference and return the path to the output video.

    Parameters
    ----------
    video_path      : input video file
    mask_path       : single-frame mask PNG (white = inpaint)
    output_dir      : where to write the result
    model_dir       : directory that holds the three .pth weight files
    propainter_dir  : root of the cloned ProPainter repository
    options         : dict with optional overrides (see DEFAULT_OPTIONS)
    progress_cb     : callback(percent: int, message: str)
    """
    opts = {**DEFAULT_OPTIONS, **options}

    inference_script = propainter_dir / "inference_propainter.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"ProPainter inference script not found at {inference_script}.\n"
            "Please clone the ProPainter repository into the 'propainter/' folder.\n"
            "See README.md for instructions."
        )

    _verify_weights(model_dir)
    _link_weights(model_dir, propainter_dir)

    # Auto-tune options based on video dimensions to prevent OOM
    opts = _autotune_options(video_path, opts, progress_cb)

    cmd = _build_command(
        inference_script, video_path, mask_path, output_dir, opts
    )

    _emit(progress_cb, 0, "Launching ProPainter…")

    return_code, output_video = _execute(cmd, propainter_dir, output_dir,
                                         video_path.stem, progress_cb)

    if return_code != 0:
        raise RuntimeError(
            f"ProPainter exited with code {return_code}. "
            "Check server logs for details."
        )
    if not output_video.exists():
        raise RuntimeError(
            f"ProPainter finished but output video not found at {output_video}."
        )

    return output_video


# ── Helpers ───────────────────────────────────────────────────────────────────

def _autotune_options(video_path: Path, opts: dict,
                      progress_cb: Optional[Callable]) -> dict:
    """
    Inspect video resolution and auto-lower subvideo_length / resize_ratio
    if the user left them at defaults, to avoid out-of-memory crashes.

    ProPainter reads ALL frames into RAM before chunked processing begins.
    Rule of thumb (FP16):
        pixels > 1280×720  → force resize_ratio 0.5 unless user set it
        subvideo_length     → cap at 50 for high-res, 80 for lower-res
    """
    import cv2 as _cv2

    cap = _cv2.VideoCapture(str(video_path))
    w   = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w <= 0 or h <= 0:
        return opts  # can't determine, leave unchanged

    opts = opts.copy()
    pixels = w * h

    # Only adjust values the user left at their defaults
    if pixels > 1280 * 720:
        if opts.get("resize_ratio", 1.0) >= 1.0:
            opts["resize_ratio"] = 0.5
            _emit(progress_cb, 0,
                  f"High resolution ({w}×{h}) detected — auto-set resize_ratio=0.5 to save RAM")
        if opts.get("subvideo_length", 80) > 50:
            opts["subvideo_length"] = 50
    elif pixels > 720 * 480:
        if opts.get("subvideo_length", 80) > 80:
            opts["subvideo_length"] = 80

    # Always enable fp16 unless user explicitly disabled it
    if "fp16" not in opts:
        opts["fp16"] = True

    return opts


REQUIRED_WEIGHTS = [
    "ProPainter.pth",
    "recurrent_flow_completion.pth",
    "raft-things.pth",
]


def _verify_weights(model_dir: Path):
    missing = [w for w in REQUIRED_WEIGHTS if not (model_dir / w).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model weight(s) in '{model_dir}':\n  "
            + "\n  ".join(missing)
            + "\nSee README.md → 'Download Model Weights' for download links."
        )


def _link_weights(model_dir: Path, propainter_dir: Path):
    """
    Make propainter/weights point to our model_dir so ProPainter's
    load_file_from_url finds the files without trying to download them.

    Strategy (cross-platform):
      1. Prefer a symlink (Linux/macOS and Windows with developer mode).
      2. Fall back to a hard-link for each individual file.
      3. Fall back to a plain file copy as last resort.
    """
    weights_dir = propainter_dir / "weights"

    # Already points to the right place?
    if weights_dir.is_symlink():
        if weights_dir.resolve() == model_dir.resolve():
            return
        weights_dir.unlink()

    if weights_dir.exists() and not weights_dir.is_symlink():
        # It is a real directory – just make sure the files are inside it.
        _copy_or_hardlink_weights(model_dir, weights_dir)
        return

    # Try symlink first
    try:
        os.symlink(str(model_dir), str(weights_dir))
        return
    except (OSError, NotImplementedError):
        pass

    # Fall back: create the dir and link/copy individual files
    weights_dir.mkdir(exist_ok=True)
    _copy_or_hardlink_weights(model_dir, weights_dir)


def _copy_or_hardlink_weights(src_dir: Path, dst_dir: Path):
    for name in REQUIRED_WEIGHTS:
        src = src_dir / name
        dst = dst_dir / name
        if dst.exists():
            continue
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))


def _build_command(
    script:     Path,
    video:      Path,
    mask:       Path,
    output_dir: Path,
    opts:       dict,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--video",  str(video),
        "--mask",   str(mask),
        "--output", str(output_dir),
        "--save_frames",                    # needed for progress counting
        "--mask_dilation",   str(opts["mask_dilation"]),
        "--ref_stride",      str(opts["ref_stride"]),
        "--neighbor_length", str(opts["neighbor_length"]),
        "--subvideo_length", str(opts["subvideo_length"]),
        "--raft_iter",       str(opts["raft_iter"]),
        "--resize_ratio",    str(opts["resize_ratio"]),
    ]

    if opts["height"] > 0:
        cmd += ["--height", str(opts["height"])]
    if opts["width"] > 0:
        cmd += ["--width", str(opts["width"])]
    if opts["save_fps"] > 0:
        cmd += ["--save_fps", str(opts["save_fps"])]
    if opts["fp16"]:
        cmd.append("--fp16")

    return cmd


# ── Progress estimation ───────────────────────────────────────────────────────
# ProPainter prints tqdm bars like:
#   inpainting: 100%|████| 120/120 [01:23<00:00,  1.44it/s]
# We parse these to report progress.

_TQDM_RE = re.compile(r"(\d+)/(\d+)")

# Rough weights for the three pipeline stages visible in stdout:
#   1. flow completion   (~20 %)
#   2. image propagation (~20 %)
#   3. inpainting        (~60 %)
_STAGE_WEIGHTS = [0.20, 0.20, 0.60]


def _execute(
    cmd:         list[str],
    cwd:         Path,
    output_dir:  Path,
    video_stem:  str,
    progress_cb: Optional[Callable[[int, str], None]],
) -> tuple[int, Path]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    stage_idx   = 0
    stage_names = ["Completing optical flow", "Propagating frames", "Inpainting"]
    accumulated = [0.0, 0.0, 0.0]   # fraction complete per stage

    def read_output():
        nonlocal stage_idx
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            print(f"[ProPainter] {line}", flush=True)

            # Detect stage transitions by keywords in the tqdm description
            low = line.lower()
            if "flow completion" in low and stage_idx < 1:
                stage_idx = 0
            elif ("propagat" in low or "img prop" in low) and stage_idx < 1:
                stage_idx = 1
            elif "inpaint" in low and stage_idx < 2:
                stage_idx = 2

            m = _TQDM_RE.search(line)
            if m:
                done  = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    accumulated[stage_idx] = done / total

            # Weighted overall progress
            overall = sum(
                accumulated[i] * _STAGE_WEIGHTS[i]
                for i in range(3)
            )
            pct = int(overall * 100)
            msg = stage_names[min(stage_idx, 2)]
            _emit(progress_cb, pct, msg)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    proc.wait()
    reader.join()

    # ProPainter saves to: <output_dir>/<video_stem>/inpaint_out.mp4
    output_video = output_dir / video_stem / "inpaint_out.mp4"
    return proc.returncode, output_video


def _emit(cb, pct, msg):
    if cb:
        try:
            cb(pct, msg)
        except Exception:
            pass
