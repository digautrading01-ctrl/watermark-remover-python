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

    cap2 = _cv2.VideoCapture(str(video_path))
    total_frames = int(cap2.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap2.release()

    opts = opts.copy()
    pixels = w * h

    # High resolution: reduce to half to stay within VRAM budget
    if pixels > 1280 * 720:
        if opts.get("resize_ratio", 1.0) >= 1.0:
            opts["resize_ratio"] = 0.5
            _emit(progress_cb, 0,
                  f"High resolution ({w}×{h}) — auto-set resize_ratio=0.5 to save RAM")
        if opts.get("subvideo_length", 80) > 50:
            opts["subvideo_length"] = 50

    # Long video (>2000 frames): reduce chunk size to keep memory bounded
    if total_frames > 2000 and opts.get("subvideo_length", 80) > 40:
        opts["subvideo_length"] = 40
        _emit(progress_cb, 0,
              f"Long video ({total_frames} frames) — auto-set subvideo_length=40")

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
# tqdm writes updates with \r (carriage return), not \n, so we cannot rely on
# line-by-line iteration. We read the raw byte stream and split on both \r and
# \n to catch every tqdm update as it arrives.
#
# ProPainter pipeline stages and their approximate share of total work:
#   Stage 0 – optical flow estimation  (RAFT)         ~15 %
#   Stage 1 – flow completion                         ~15 %
#   Stage 2 – image propagation                       ~20 %
#   Stage 3 – inpainting (transformer)                ~50 %

_TQDM_RE = re.compile(r"(\d+)/(\d+)")

_STAGE_NAMES   = ["Estimating optical flow", "Completing optical flow",
                  "Propagating frames",       "Inpainting"]
_STAGE_WEIGHTS = [0.15, 0.15, 0.20, 0.50]

# Keywords that appear in tqdm bar descriptions for each stage
_STAGE_KEYWORDS = [
    ["raft", "flow estimation", "completing flows"],   # stage 0
    ["flow completion", "flow_completion"],             # stage 1
    ["propagat", "img prop", "image prop"],             # stage 2
    ["inpaint"],                                        # stage 3
]


def _execute(
    cmd:         list[str],
    cwd:         Path,
    output_dir:  Path,
    video_stem:  str,
    progress_cb: Optional[Callable[[int, str], None]],
) -> tuple[int, Path]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force tqdm to emit plain-text progress without ANSI colour codes and
    # with \r updates so we can parse them from a non-TTY pipe.
    env["TQDM_NCOLS"]        = "120"
    env["TQDM_DISABLE"]      = "0"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,          # unbuffered bytes
        env=env,
    )

    stage_idx   = 0
    accumulated = [0.0] * len(_STAGE_NAMES)

    def _parse_chunk(chunk: str):
        nonlocal stage_idx
        # Split on both CR and LF so we catch every tqdm update
        for raw in re.split(r"[\r\n]+", chunk):
            line = raw.strip()
            if not line:
                continue
            print(f"[ProPainter] {line}", flush=True)

            low = line.lower()

            # Advance stage: scan forward from current stage only
            for si in range(stage_idx, len(_STAGE_KEYWORDS)):
                if any(kw in low for kw in _STAGE_KEYWORDS[si]):
                    stage_idx = si
                    break

            m = _TQDM_RE.search(line)
            if m:
                done  = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    accumulated[stage_idx] = done / total

            overall = sum(
                accumulated[i] * _STAGE_WEIGHTS[i]
                for i in range(len(_STAGE_NAMES))
            )
            pct = min(99, int(overall * 100))
            _emit(progress_cb, pct, _STAGE_NAMES[stage_idx])

    def read_output():
        buf = b""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            # Decode and flush whenever we see a line-ending character
            try:
                text = buf.decode("utf-8", errors="replace")
            except Exception:
                continue
            if "\r" in text or "\n" in text:
                _parse_chunk(text)
                buf = b""
        # Flush any remaining buffered bytes
        if buf:
            try:
                _parse_chunk(buf.decode("utf-8", errors="replace"))
            except Exception:
                pass

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
