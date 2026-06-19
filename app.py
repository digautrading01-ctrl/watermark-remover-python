"""
Watermark & Subtitle Remover – Flask application
Wraps ProPainter for offline video inpainting.
"""

import os
import sys
import uuid
import json
import time
import queue
import threading
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, Response, stream_with_context
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
UPLOAD_DIR    = BASE_DIR / "uploads"
RESULTS_DIR   = BASE_DIR / "results"
MODEL_DIR     = BASE_DIR / "model"
PROPAINTER_DIR = BASE_DIR / "propainter"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── In-memory job store ───────────────────────────────────────────────────────
#  jobs[job_id] = {
#    "status": "pending"|"running"|"done"|"error",
#    "progress": 0-100,
#    "message": str,
#    "output_file": str | None,
#    "queue": Queue  (for SSE)
#  }
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": total, "width": w, "height": h}


def extract_first_frame(video_path: Path, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.imwrite(str(out_path), frame)
    return True


def push_event(job_id: str, event: str, data: dict):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["queue"].put(json.dumps({"event": event, **data}))


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    ext = f.filename.rsplit(".", 1)[1].lower()
    video_path = job_dir / f"input.{ext}"
    f.save(str(video_path))

    # Extract preview frame
    preview_path = job_dir / "preview.jpg"
    ok = extract_first_frame(video_path, preview_path)
    if not ok:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": "Cannot read video file"}), 400

    meta = get_video_meta(video_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Waiting",
            "output_file": None,
            "video_path": str(video_path),
            "queue": queue.Queue(),
        }

    return jsonify({
        "job_id": job_id,
        "meta": meta,
        "preview_url": f"/preview/{job_id}",
    })


@app.route("/preview/<job_id>")
def serve_preview(job_id: str):
    preview = UPLOAD_DIR / job_id / "preview.jpg"
    if not preview.exists():
        return "Not found", 404
    return send_from_directory(str(preview.parent), preview.name)


@app.route("/detect_subtitles/<job_id>")
def detect_subtitles(job_id: str):
    """
    Analyse the first frame and return a mask PNG representing subtitle regions.
    Returns JSON with base64-encoded mask + bounding boxes.
    """
    from subtitle_detector import detect_subtitle_regions

    preview = UPLOAD_DIR / job_id / "preview.jpg"
    if not preview.exists():
        return jsonify({"error": "Job not found"}), 404

    frame = cv2.imread(str(preview))
    if frame is None:
        return jsonify({"error": "Cannot read preview frame"}), 500

    mask, boxes = detect_subtitle_regions(frame)

    import base64
    _, buf = cv2.imencode(".png", mask)
    b64 = base64.b64encode(buf.tobytes()).decode()

    return jsonify({"mask_png": b64, "boxes": boxes})


@app.route("/process", methods=["POST"])
def start_processing():
    data = request.get_json(force=True)
    job_id  = data.get("job_id")
    mask_b64 = data.get("mask_png")          # base64-encoded PNG mask drawn by user
    options  = data.get("options", {})

    if not job_id or not mask_b64:
        return jsonify({"error": "job_id and mask_png required"}), 400

    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Unknown job"}), 404
        if jobs[job_id]["status"] == "running":
            return jsonify({"error": "Job already running"}), 409

    # Save mask image
    import base64
    mask_bytes = base64.b64decode(mask_b64)
    mask_np    = np.frombuffer(mask_bytes, np.uint8)
    mask_img   = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return jsonify({"error": "Invalid mask image"}), 400

    mask_path = UPLOAD_DIR / job_id / "mask.png"
    cv2.imwrite(str(mask_path), mask_img)

    update_job(job_id, status="running", progress=0, message="Starting…")

    thread = threading.Thread(
        target=_run_inference,
        args=(job_id, options),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route("/progress/<job_id>")
def progress_stream(job_id: str):
    """Server-Sent Events stream for job progress."""
    with jobs_lock:
        if job_id not in jobs:
            return "Unknown job", 404
        q = jobs[job_id]["queue"]

    def generate():
        while True:
            try:
                payload = q.get(timeout=30)
                yield f"data: {payload}\n\n"
                parsed = json.loads(payload)
                if parsed.get("event") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"event\": \"ping\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/status/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Unknown job"}), 404
        j = jobs[job_id]
        return jsonify({
            "status": j["status"],
            "progress": j["progress"],
            "message": j["message"],
            "output_url": f"/download/{job_id}" if j["status"] == "done" else None,
        })


@app.route("/download/<job_id>")
def download_result(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return "Unknown job", 404
        j = jobs[job_id]

    if j["status"] != "done" or not j["output_file"]:
        return "Not ready", 404

    out = Path(j["output_file"])
    return send_from_directory(str(out.parent), out.name, as_attachment=True)


# ── Inference worker ──────────────────────────────────────────────────────────

def _run_inference(job_id: str, options: dict):
    from inference_engine import run_propainter

    with jobs_lock:
        video_path = Path(jobs[job_id]["video_path"])
    mask_path  = UPLOAD_DIR / job_id / "mask.png"
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    def progress_cb(pct: int, msg: str):
        update_job(job_id, progress=pct, message=msg)
        push_event(job_id, "progress", {"progress": pct, "message": msg})

    try:
        output_path = run_propainter(
            video_path  = video_path,
            mask_path   = mask_path,
            output_dir  = result_dir,
            model_dir   = MODEL_DIR,
            propainter_dir = PROPAINTER_DIR,
            options     = options,
            progress_cb = progress_cb,
        )
        update_job(job_id, status="done", progress=100,
                   message="Done", output_file=str(output_path))
        push_event(job_id, "done", {
            "progress": 100,
            "message": "Done",
            "download_url": f"/download/{job_id}",
        })
    except Exception as exc:
        msg = str(exc)
        update_job(job_id, status="error", message=msg)
        push_event(job_id, "error", {"message": msg})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
