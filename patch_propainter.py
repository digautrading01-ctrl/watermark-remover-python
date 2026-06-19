"""
patch_propainter.py
───────────────────
Patches propainter/inference_propainter.py to fix the
    AttributeError: module 'torchvision.io' has no attribute 'read_video'
error that occurs with torchvision >= 0.16.

Compatible with macOS, Linux, and Windows.

Run once from the watermark-remover project root:
    python patch_propainter.py
"""

import re
import sys
import shutil
from pathlib import Path

TARGET = Path(__file__).parent / "propainter" / "inference_propainter.py"

if not TARGET.exists():
    sys.exit(
        f"[ERROR] File not found: {TARGET}\n"
        "Make sure you cloned ProPainter into the 'propainter/' folder."
    )

src = TARGET.read_text(encoding="utf-8")

# ── Already patched? ──────────────────────────────────────────────────────
if "cv2.VideoCapture" in src and "read_video" not in src:
    print("[OK] inference_propainter.py is already patched.")
    sys.exit(0)

# ── Backup ────────────────────────────────────────────────────────────────
# Use .bak suffix appended to the full filename, e.g. inference_propainter.py.bak
backup = TARGET.parent / (TARGET.name + ".bak")
shutil.copy2(TARGET, backup)
print(f"[INFO] Backup saved to {backup}")

# ── Replacement for read_frame_from_videos ────────────────────────────────
# The original function uses torchvision.io.read_video which was removed in
# torchvision >= 0.16. We replace the entire function with a cv2-based
# implementation that returns the same tuple: (frames, fps, size, video_name)
#
# Windows notes:
#   - frame_root may be a str or a pathlib.Path; we normalise to str early.
#   - Path separators (/ vs \) are handled by os.path throughout.
#   - cv2.VideoCapture accepts both forward- and back-slash paths on Windows.
#   - os.listdir + endswith covers case-insensitive filesystems because we
#     check both upper- and lower-case extensions explicitly.

OLD_FUNC = re.search(
    r"(def read_frame_from_videos\(frame_root\):.*?)(?=\ndef |\nclass |\nif __name__)",
    src,
    re.DOTALL,
)

if not OLD_FUNC:
    sys.exit(
        "[ERROR] Could not locate read_frame_from_videos() in the source.\n"
        "The ProPainter version you have may differ from what this patch expects.\n"
        "Please apply the fix manually — see README.md for guidance."
    )

NEW_FUNC = '''\
def read_frame_from_videos(frame_root):
    """
    Read frames from a video file or a directory of images.
    Patched to use cv2 instead of torchvision.io.read_video
    (compatible with torchvision >= 0.16, macOS / Linux / Windows).
    """
    import cv2

    # Normalise to str so both str and pathlib.Path inputs work on all OSes
    frame_root = str(frame_root)

    VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm',
                  '.MP4', '.MOV', '.AVI', '.MKV', '.WEBM')
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png',
                  '.JPG', '.JPEG', '.PNG')

    if frame_root.lower().endswith(tuple(e.lower() for e in VIDEO_EXTS)):
        # ── Video file ────────────────────────────────────────────────────
        video_name = os.path.splitext(os.path.basename(frame_root))[0]
        # On Windows, forward-slash paths are fine for cv2
        cap = cv2.VideoCapture(frame_root)
        if not cap.isOpened():
            raise ValueError(f"cv2 could not open video: {frame_root}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        if not frames:
            raise ValueError(f"No frames could be decoded from: {frame_root}")
        size = frames[0].size  # PIL size: (width, height)
    else:
        # ── Directory of images ───────────────────────────────────────────
        video_name = os.path.basename(frame_root.rstrip("\\/"))
        frame_files = sorted([
            f for f in os.listdir(frame_root)
            if os.path.splitext(f)[1] in IMAGE_EXTS
        ])
        if not frame_files:
            raise ValueError(f"No image frames found in: {frame_root}")
        frames = [
            Image.open(os.path.join(frame_root, f)).convert('RGB')
            for f in frame_files
        ]
        fps = 24.0
        size = frames[0].size

    return frames, fps, size, video_name

'''

patched = src[:OLD_FUNC.start()] + NEW_FUNC + src[OLD_FUNC.end():]

# ── Comment-out the now-unused torchvision.io import ──────────────────────
patched = re.sub(
    r"^(import torchvision\.io\b.*|from torchvision\.io import.*)$",
    r"# \1  # patched: replaced by cv2",
    patched,
    flags=re.MULTILINE,
)

TARGET.write_text(patched, encoding="utf-8")
print("[OK] inference_propainter.py patched successfully.")
print("     torchvision.io.read_video → cv2.VideoCapture")
print(f"     Original backed up to: {backup}")
