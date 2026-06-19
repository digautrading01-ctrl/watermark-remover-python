"""
subtitle_detector.py
────────────────────
Heuristic detection of hard-coded subtitle regions in a video frame.

Algorithm
─────────
1. Convert to grayscale and apply adaptive thresholding to find bright-on-dark
   (or dark-on-bright) text-like blobs.
2. Focus on the bottom 30 % of the frame where subtitles typically live.
3. Optionally also scan the top 10 % for header watermarks.
4. Dilate connected components horizontally to merge letter clusters into word
   groups, then filter by aspect ratio and minimum width to reject noise.
5. Union all detected boxes into horizontal subtitle bands and return:
   - a full-frame binary mask (255 = region to inpaint)
   - a list of bounding-box dicts  {x, y, w, h}  in pixel coordinates

No external model is required – only OpenCV.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


# ── Public API ────────────────────────────────────────────────────────────────

def detect_subtitle_regions(
    frame: np.ndarray,
    *,
    bottom_fraction: float = 0.30,
    top_fraction:    float = 0.10,
    min_width_ratio: float = 0.10,   # min box width as fraction of frame width
    min_height:      int   = 8,
    max_height:      int   = 80,
    dilation_iters:  int   = 5,
    padding:         int   = 6,
) -> Tuple[np.ndarray, list[dict]]:
    """
    Parameters
    ----------
    frame            : BGR image (H×W×3 uint8)
    bottom_fraction  : fraction of frame height to search at the bottom
    top_fraction     : fraction of frame height to search at the top
    min_width_ratio  : discard boxes narrower than this × frame_width
    min_height       : discard boxes shorter than this many pixels
    max_height       : discard boxes taller than this many pixels
    dilation_iters   : horizontal dilation strength to merge letters
    padding          : extra pixels added around each detected box

    Returns
    -------
    mask   : binary uint8 ndarray (H×W), 255 inside detected regions
    boxes  : list of {"x":, "y":, "w":, "h":} dicts (full-frame coords)
    """
    H, W = frame.shape[:2]
    min_width = int(W * min_width_ratio)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes: list[dict] = []

    # Define search zones: (y_start, y_end)
    zones = [
        (int(H * (1.0 - bottom_fraction)), H),          # bottom
        (0, int(H * top_fraction)),                      # top
    ]

    for (y0, y1) in zones:
        if y1 <= y0:
            continue

        roi = gray[y0:y1, :]
        zone_boxes = _detect_in_zone(
            roi,
            min_width=min_width,
            min_height=min_height,
            max_height=max_height,
            dilation_iters=dilation_iters,
        )

        for (x, y, w, h) in zone_boxes:
            # Convert back to full-frame coordinates
            fy  = y0 + y
            fx  = x
            fw  = w
            fh  = h

            # Add padding (clamped)
            px1 = max(0,  fx - padding)
            py1 = max(0,  fy - padding)
            px2 = min(W,  fx + fw + padding)
            py2 = min(H,  fy + fh + padding)

            mask[py1:py2, px1:px2] = 255
            boxes.append({"x": px1, "y": py1, "w": px2 - px1, "h": py2 - py1})

    # Merge overlapping boxes
    boxes = _merge_boxes(boxes)
    # Rebuild clean mask from merged boxes
    mask[:] = 0
    for b in boxes:
        mask[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = 255

    return mask, boxes


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_in_zone(
    roi:           np.ndarray,
    min_width:     int,
    min_height:    int,
    max_height:    int,
    dilation_iters: int,
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes inside the ROI (zone-local coordinates)."""
    results: list[tuple[int, int, int, int]] = []

    # ── Method 1: adaptive threshold (works for subtitles on varying bg) ──
    thresh = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=8,
    )

    # ── Method 2: simple global threshold (catches bright hard subs) ──
    _, bright = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(thresh, bright)

    # Horizontal dilation to merge letters into words
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, dilation_iters * 4), max(3, dilation_iters)),
    )
    dilated = cv2.dilate(combined, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_width:
            continue
        if h < min_height or h > max_height:
            continue
        # Aspect ratio guard: subtitles are wide, not tall
        if h > w * 0.8:
            continue
        results.append((x, y, w, h))

    return results


def _merge_boxes(boxes: list[dict]) -> list[dict]:
    """Merge overlapping or vertically close bounding boxes."""
    if not boxes:
        return boxes

    # Sort by y
    boxes = sorted(boxes, key=lambda b: b["y"])
    merged = [boxes[0].copy()]

    for b in boxes[1:]:
        last = merged[-1]
        # Overlap or within 10 px gap vertically?
        if b["y"] <= last["y"] + last["h"] + 10:
            x1 = min(last["x"], b["x"])
            y1 = min(last["y"], b["y"])
            x2 = max(last["x"] + last["w"], b["x"] + b["w"])
            y2 = max(last["y"] + last["h"], b["y"] + b["h"])
            merged[-1] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        else:
            merged.append(b.copy())

    return merged
