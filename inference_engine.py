"""
inference_engine.py
───────────────────
Drives ProPainter inference via subprocess.

ProPainter is expected to be cloned at <project>/propainter/.
Model weights are loaded from <project>/model/ via a per-job symlink
(or directory junction on Windows) placed at propainter/weights before
each run, so the upstream code requires no changes.

Segmentation
────────────
For long videos the engine splits the input into overlapping segments,
processes each through ProPainter independently, trims the overlap frames,
and concatenates the results via ffmpeg (sourced from imageio-ffmpeg).
This keeps peak VRAM proportional to the segment length rather than the
full video length.

Overlap diagram (seg_frames=150, overlap=15):

  ┌──────────── seg 0 (165 fr) ────────────┐
                              ┌──────────── seg 1 (180 fr) ────────────┐
  ↑ keep 150 fr ↑             ↑ skip 15 ↑  keep 150 fr (or remainder)
               overlap        overlap

Progress allocation:
  0 –  3 %  splitting
  3 – 97 %  ProPainter (distributed across segments proportionally)
 97 – 99 %  concatenation + audio remux
 99 –100 %  reported by the caller
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
    # ── Segmentation ──────────────────────────────────────────────────────────
    "segment_frames":   150,      # effective frames per segment (0 = disable)
    "overlap_frames":   15,       # frames of temporal context shared between segments
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
    opts = _autotune_options(video_path, opts, progress_cb)

    # Video metadata
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(_cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    seg_frames = int(opts.get("segment_frames", 150))
    overlap    = int(opts.get("overlap_frames",  15))

    # Use segmented path when video is long enough to warrant splitting
    if seg_frames > 0 and total_frames > seg_frames:
        return _run_segmented(
            video_path, mask_path, output_dir, model_dir, propainter_dir,
            opts, progress_cb, fps, total_frames, seg_frames, overlap,
        )

    # ── Single-pass (short videos) ────────────────────────────────────────────
    _emit(progress_cb, 0, "Launching ProPainter…")
    cmd = _build_command(inference_script, video_path, mask_path, output_dir, opts)
    return_code, output_video = _execute(
        cmd, propainter_dir, output_dir, video_path.stem, progress_cb
    )
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


# ── Segmented processing ──────────────────────────────────────────────────────

def _plan_segments(
    total_frames: int,
    seg_frames:   int,
    overlap:      int,
) -> list[tuple[int, int, int, int]]:
    """
    Plan overlapping segments for the full video.

    Returns a list of (global_start, global_end, keep_start, keep_count):
      global_start  – first frame to extract from the original video (inclusive)
      global_end    – last  frame to extract from the original video (exclusive)
      keep_start    – index within the ProPainter output to start keeping
      keep_count    – number of output frames to keep for this segment
    """
    segments: list[tuple[int, int, int, int]] = []
    effective_start = 0   # global frame where kept output begins

    while effective_start < total_frames:
        is_first = (len(segments) == 0)

        # Add pre-overlap (except for the very first segment)
        global_start = effective_start if is_first else effective_start - overlap

        # Add post-overlap (clamped to video length)
        global_end = min(total_frames, effective_start + seg_frames + overlap)

        keep_start = 0 if is_first else overlap
        keep_count = min(seg_frames, total_frames - effective_start)

        segments.append((global_start, global_end, keep_start, keep_count))
        effective_start += seg_frames

        if global_end >= total_frames:
            break

    return segments


def _run_segmented(
    video_path:     Path,
    mask_path:      Path,
    output_dir:     Path,
    model_dir:      Path,
    propainter_dir: Path,
    opts:           dict,
    progress_cb:    Optional[Callable[[int, str], None]],
    fps:            float,
    total_frames:   int,
    seg_frames:     int,
    overlap:        int,
) -> Path:
    ffmpeg_exe       = _get_ffmpeg()
    inference_script = propainter_dir / "inference_propainter.py"

    plan = _plan_segments(total_frames, seg_frames, overlap)
    n    = len(plan)

    _emit(progress_cb, 0, f"Splitting video into {n} segment(s)…")

    seg_dir     = output_dir / "_segments"
    trimmed_dir = output_dir / "_trimmed"
    seg_dir.mkdir(parents=True, exist_ok=True)
    trimmed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Split video into segments (0 – 3 %) ───────────────────────────────
    seg_paths: list[Path] = []
    for i, (gs, ge, _ks, _kc) in enumerate(plan):
        seg_path = seg_dir / f"seg_{i:04d}.mp4"
        _ffmpeg_extract(video_path, seg_path, gs, ge, fps, ffmpeg_exe)
        seg_paths.append(seg_path)

    _emit(progress_cb, 3, f"Split into {n} segment(s). Starting inference…")

    # ── 2. Process each segment (3 – 97 %) ───────────────────────────────────
    trimmed_paths: list[Path] = []

    for i, ((gs, ge, ks, kc), seg_path) in enumerate(zip(plan, seg_paths)):
        seg_out_dir = output_dir / f"_seg_{i:04d}_out"
        seg_out_dir.mkdir(parents=True, exist_ok=True)

        # Map [0,100] progress of this segment to the global [3, 97] band
        seg_base = 3 + int((i / n) * 94)
        seg_span = max(1, int((1 / n) * 94))

        seg_cb = _make_seg_cb(progress_cb, seg_base, seg_span, i + 1, n)
        _emit(seg_cb, 0, "Launching ProPainter…")

        cmd = _build_command(inference_script, seg_path, mask_path, seg_out_dir, opts)
        rc, seg_output = _execute(cmd, propainter_dir, seg_out_dir, seg_path.stem, seg_cb)

        if rc != 0:
            raise RuntimeError(
                f"ProPainter failed on segment {i + 1}/{n} (exit code {rc}). "
                "Check server logs for details."
            )
        if not seg_output.exists():
            raise RuntimeError(
                f"ProPainter output missing for segment {i + 1}/{n}: {seg_output}"
            )

        # Trim overlap frames from this segment's output
        trimmed = trimmed_dir / f"trimmed_{i:04d}.mp4"
        _ffmpeg_trim(seg_output, trimmed, ks, kc, fps, ffmpeg_exe)
        trimmed_paths.append(trimmed)

    # ── 3. Concatenate trimmed segments (97 – 98 %) ──────────────────────────
    _emit(progress_cb, 97, "Concatenating segments…")
    concat_no_audio = output_dir / "_concat_no_audio.mp4"
    _ffmpeg_concat(trimmed_paths, concat_no_audio, ffmpeg_exe)

    # ── 4. Remux original audio (98 – 99 %) ──────────────────────────────────
    _emit(progress_cb, 98, "Remuxing audio…")
    final_output = output_dir / "inpaint_out.mp4"
    _ffmpeg_remux_audio(concat_no_audio, video_path, final_output, ffmpeg_exe)

    # ── 5. Cleanup temporary files ────────────────────────────────────────────
    shutil.rmtree(seg_dir,     ignore_errors=True)
    shutil.rmtree(trimmed_dir, ignore_errors=True)
    concat_no_audio.unlink(missing_ok=True)
    # Remove per-segment ProPainter output dirs
    for i in range(n):
        shutil.rmtree(output_dir / f"_seg_{i:04d}_out", ignore_errors=True)

    _emit(progress_cb, 99, "Done")
    return final_output


def _make_seg_cb(
    progress_cb: Optional[Callable[[int, str], None]],
    base: int,
    span: int,
    seg_idx: int,
    n_segs: int,
) -> Optional[Callable[[int, str], None]]:
    """Return a progress callback that maps [0,100] into [base, base+span]."""
    if progress_cb is None:
        return None

    def cb(pct: int, msg: str):
        overall = base + int((pct / 100) * span)
        _emit(progress_cb, overall, f"[{seg_idx}/{n_segs}] {msg}")

    return cb


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _get_ffmpeg() -> str:
    """Return path to an ffmpeg executable (prefers imageio-ffmpeg bundle)."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise RuntimeError(
        "ffmpeg not found. Install imageio-ffmpeg (pip install imageio-ffmpeg) "
        "or add ffmpeg to your PATH."
    )


def _ffmpeg_run(cmd: list[str], label: str):
    """Run an ffmpeg command; raise RuntimeError on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg {label} failed:\n{stderr}")


def _ffmpeg_extract(
    src: Path, dst: Path,
    start_frame: int, end_frame: int,
    fps: float, ffmpeg: str,
):
    """Extract frames [start_frame, end_frame) from src into dst."""
    start_t  = start_frame / fps
    duration = (end_frame - start_frame) / fps
    _ffmpeg_run([
        ffmpeg, "-y",
        "-ss", f"{start_t:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        # Re-encode so the output starts on a proper keyframe
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
        "-an",           # strip audio – ProPainter doesn't need it
        str(dst),
    ], f"extract {dst.name}")


def _ffmpeg_trim(
    src: Path, dst: Path,
    keep_start: int, keep_count: int,
    fps: float, ffmpeg: str,
):
    """Keep keep_count frames starting at keep_start from src."""
    start_t  = keep_start / fps
    duration = keep_count / fps
    _ffmpeg_run([
        ffmpeg, "-y",
        "-ss", f"{start_t:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        str(dst),
    ], f"trim {dst.name}")


def _ffmpeg_concat(parts: list[Path], dst: Path, ffmpeg: str):
    """Concatenate video files in order using the concat demuxer."""
    list_file = dst.parent / "_concat_list.txt"
    # Use POSIX-style paths; ffmpeg accepts forward slashes on all platforms
    list_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in parts),
        encoding="utf-8",
    )
    try:
        _ffmpeg_run([
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(dst),
        ], "concat")
    finally:
        list_file.unlink(missing_ok=True)


def _ffmpeg_remux_audio(video: Path, audio_src: Path, dst: Path, ffmpeg: str):
    """
    Combine video stream from `video` with the audio stream (if any) from
    `audio_src`.  The `?` suffix on the audio map makes it optional so the
    command succeeds even when the source video has no audio track.
    """
    _ffmpeg_run([
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(audio_src),
        "-map", "0:v",
        "-map", "1:a?",
        "-c", "copy",
        str(dst),
    ], "audio remux")


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
    # (Only applies in single-pass mode; segmented mode never sees >segment_frames.)
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
