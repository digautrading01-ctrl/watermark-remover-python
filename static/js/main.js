/* ═══════════════════════════════════════════════════════════════════════════
   main.js  –  Watermark & Subtitle Remover front-end logic
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

/* ── Section references ────────────────────────────────────────────────── */
const secUpload   = document.getElementById("sec-upload");
const secMask     = document.getElementById("sec-mask");
const secOptions  = document.getElementById("sec-options");
const secProgress = document.getElementById("sec-progress");
const secDone     = document.getElementById("sec-done");
const secError    = document.getElementById("sec-error");

function showSection(sec) {
  [secUpload, secMask, secOptions, secProgress, secDone, secError]
    .forEach(s => s.classList.add("hidden"));
  sec.classList.remove("hidden");
}

/* ── Upload ────────────────────────────────────────────────────────────── */
const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const uploadInfo  = document.getElementById("upload-info");
const uploadName  = document.getElementById("upload-filename");
const uploadMeta  = document.getElementById("upload-meta");

let jobId        = null;
let videoMeta    = null;   // { fps, total_frames, width, height }

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter") fileInput.click(); });

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const allowed = ["video/mp4","video/quicktime","video/x-msvideo",
                   "video/x-matroska","video/webm"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
    alert("Unsupported file type. Please upload MP4, MOV, AVI, MKV or WEBM.");
    return;
  }

  uploadName.textContent = file.name;
  uploadMeta.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  uploadInfo.classList.remove("hidden");
  dropZone.querySelector("p").textContent = "Uploading…";

  const fd = new FormData();
  fd.append("video", file);

  fetch("/upload", { method: "POST", body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      jobId     = data.job_id;
      videoMeta = data.meta;
      dropZone.querySelector("p").textContent =
        `${file.name} uploaded successfully.`;
      initMaskEditor(data.preview_url, data.meta);
    })
    .catch(err => showError(err.message));
}

/* ── Mask editor ───────────────────────────────────────────────────────── */
const canvasVideo  = document.getElementById("canvas-video");
const canvasMask   = document.getElementById("canvas-mask");
const ctxVideo     = canvasVideo.getContext("2d");
const ctxMask      = canvasMask.getContext("2d");

const toolBrush    = document.getElementById("tool-brush");
const toolRect     = document.getElementById("tool-rect");
const toolErase    = document.getElementById("tool-erase");
const brushSizeEl  = document.getElementById("brush-size");
const brushSizeVal = document.getElementById("brush-size-val");
const maskOpacity  = document.getElementById("mask-opacity");
const btnAutoDetect = document.getElementById("btn-auto-detect");
const btnClearMask  = document.getElementById("btn-clear-mask");

let currentTool    = "brush";   // brush | rect | erase
let isDrawing      = false;
let rectStart      = null;
let rectSnapshot   = null;      // imageData snapshot before rect preview
let DISPLAY_W      = 0;
let DISPLAY_H      = 0;

function initMaskEditor(previewUrl, meta) {
  const img = new Image();
  img.onload = () => {
    // Scale to fit max 820 px wide
    const maxW = Math.min(820, window.innerWidth - 40);
    const scale = Math.min(1, maxW / img.naturalWidth);
    DISPLAY_W = Math.round(img.naturalWidth  * scale);
    DISPLAY_H = Math.round(img.naturalHeight * scale);

    canvasVideo.width  = DISPLAY_W;
    canvasVideo.height = DISPLAY_H;
    canvasMask.width   = DISPLAY_W;
    canvasMask.height  = DISPLAY_H;

    ctxVideo.drawImage(img, 0, 0, DISPLAY_W, DISPLAY_H);
    ctxMask.clearRect(0, 0, DISPLAY_W, DISPLAY_H);

    showSection(secMask);
    secOptions.classList.remove("hidden");   // show options below mask
  };
  img.src = previewUrl;
}

/* Tool selection */
[toolBrush, toolRect, toolErase].forEach(btn => {
  btn.addEventListener("click", () => {
    [toolBrush, toolRect, toolErase].forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentTool = btn.id.replace("tool-", "");
  });
});

brushSizeEl.addEventListener("input", () => {
  brushSizeVal.textContent = brushSizeEl.value;
});

/* Drawing */
canvasMask.addEventListener("mousedown",  e => startDraw(e));
canvasMask.addEventListener("mousemove",  e => draw(e));
canvasMask.addEventListener("mouseup",    e => endDraw(e));
canvasMask.addEventListener("mouseleave", e => endDraw(e));

canvasMask.addEventListener("touchstart",  e => { e.preventDefault(); startDraw(e.touches[0]); }, { passive: false });
canvasMask.addEventListener("touchmove",   e => { e.preventDefault(); draw(e.touches[0]); },      { passive: false });
canvasMask.addEventListener("touchend",    e => { e.preventDefault(); endDraw(e.changedTouches[0]); }, { passive: false });

function canvasPos(e) {
  const r = canvasMask.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvasMask.width  / r.width),
    y: (e.clientY - r.top)  * (canvasMask.height / r.height),
  };
}

function startDraw(e) {
  isDrawing = true;
  const { x, y } = canvasPos(e);

  if (currentTool === "rect") {
    rectStart    = { x, y };
    rectSnapshot = ctxMask.getImageData(0, 0, DISPLAY_W, DISPLAY_H);
    return;
  }

  ctxMask.beginPath();
  ctxMask.moveTo(x, y);
  paintAt(x, y);
}

function draw(e) {
  if (!isDrawing) return;
  const { x, y } = canvasPos(e);

  if (currentTool === "rect") {
    // Live preview
    ctxMask.putImageData(rectSnapshot, 0, 0);
    drawRectPreview(rectStart.x, rectStart.y, x, y);
    return;
  }

  paintAt(x, y);
}

function endDraw(e) {
  if (!isDrawing) return;
  isDrawing = false;

  if (currentTool === "rect" && rectStart && e) {
    const { x, y } = canvasPos(e);
    ctxMask.putImageData(rectSnapshot, 0, 0);
    commitRect(rectStart.x, rectStart.y, x, y);
    rectStart = null;
    rectSnapshot = null;
    return;
  }

  ctxMask.closePath();
}

function paintAt(x, y) {
  const size = parseInt(brushSizeEl.value);
  if (currentTool === "erase") {
    ctxMask.globalCompositeOperation = "destination-out";
    ctxMask.fillStyle = "rgba(0,0,0,1)";
  } else {
    ctxMask.globalCompositeOperation = "source-over";
    ctxMask.fillStyle = "rgba(255,80,80,0.85)";
  }
  ctxMask.beginPath();
  ctxMask.arc(x, y, size / 2, 0, Math.PI * 2);
  ctxMask.fill();
  ctxMask.globalCompositeOperation = "source-over";
}

function drawRectPreview(x1, y1, x2, y2) {
  ctxMask.globalCompositeOperation = "source-over";
  ctxMask.fillStyle = "rgba(255,80,80,0.6)";
  ctxMask.fillRect(
    Math.min(x1, x2), Math.min(y1, y2),
    Math.abs(x2 - x1), Math.abs(y2 - y1)
  );
}

function commitRect(x1, y1, x2, y2) {
  ctxMask.globalCompositeOperation = "source-over";
  ctxMask.fillStyle = "rgba(255,80,80,0.85)";
  ctxMask.fillRect(
    Math.min(x1, x2), Math.min(y1, y2),
    Math.abs(x2 - x1), Math.abs(y2 - y1)
  );
}

btnClearMask.addEventListener("click", () => {
  ctxMask.clearRect(0, 0, DISPLAY_W, DISPLAY_H);
});

/* ── Auto-detect subtitles ─────────────────────────────────────────────── */
btnAutoDetect.addEventListener("click", () => {
  if (!jobId) return;
  btnAutoDetect.textContent = "Detecting…";
  btnAutoDetect.disabled = true;

  fetch(`/detect_subtitles/${jobId}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      applyDetectedMask(data.mask_png, data.boxes);
    })
    .catch(err => alert("Auto-detect failed: " + err.message))
    .finally(() => {
      btnAutoDetect.textContent = "Auto-Detect Subtitles";
      btnAutoDetect.disabled = false;
    });
});

function applyDetectedMask(maskPng, boxes) {
  const img = new Image();
  img.onload = () => {
    // Scale mask from original resolution to display resolution
    const scaleX = DISPLAY_W / img.naturalWidth;
    const scaleY = DISPLAY_H / img.naturalHeight;

    ctxMask.globalCompositeOperation = "source-over";
    ctxMask.drawImage(img, 0, 0, img.naturalWidth * scaleX, img.naturalHeight * scaleY);

    // Tint the drawn mask red for visibility
    ctxMask.globalCompositeOperation = "source-atop";
    ctxMask.fillStyle = "rgba(255,80,80,0.7)";
    ctxMask.fillRect(0, 0, DISPLAY_W, DISPLAY_H);
    ctxMask.globalCompositeOperation = "source-over";
  };
  img.src = "data:image/png;base64," + maskPng;
}

/* ── Process ───────────────────────────────────────────────────────────── */
const btnProcess = document.getElementById("btn-process");
btnProcess.addEventListener("click", startProcessing);

function buildMaskPng() {
  // Create a clean B&W mask: painted areas → white, rest → black
  const offscreen = document.createElement("canvas");
  offscreen.width  = DISPLAY_W;
  offscreen.height = DISPLAY_H;
  const ctx = offscreen.getContext("2d");

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);

  // Composite the mask channel: use alpha as white pixels
  const data = ctxMask.getImageData(0, 0, DISPLAY_W, DISPLAY_H);
  const bw   = ctx.getImageData(0, 0, DISPLAY_W, DISPLAY_H);

  for (let i = 0; i < data.data.length; i += 4) {
    const alpha = data.data[i + 3];
    if (alpha > 30) {
      bw.data[i]     = 255;
      bw.data[i + 1] = 255;
      bw.data[i + 2] = 255;
      bw.data[i + 3] = 255;
    }
  }

  ctx.putImageData(bw, 0, 0);

  return new Promise((resolve) => {
    offscreen.toBlob(blob => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.readAsDataURL(blob);
    }, "image/png");
  });
}

async function startProcessing() {
  if (!jobId) { alert("Please upload a video first."); return; }

  // Check mask is not empty
  const maskData = ctxMask.getImageData(0, 0, DISPLAY_W, DISPLAY_H);
  const hasContent = maskData.data.some((v, i) => i % 4 === 3 && v > 30);
  if (!hasContent) {
    alert("The mask is empty. Please paint over the areas you want to remove first.");
    return;
  }

  const maskPng = await buildMaskPng();

  const options = {
    fp16:             document.getElementById("opt-fp16").checked,
    subvideo_length:  parseInt(document.getElementById("opt-subvideo").value),
    neighbor_length:  parseInt(document.getElementById("opt-neighbor").value),
    mask_dilation:    parseInt(document.getElementById("opt-dilation").value),
    resize_ratio:     parseFloat(document.getElementById("opt-resize").value),
    height:           parseInt(document.getElementById("opt-height").value),
    width:            parseInt(document.getElementById("opt-width").value),
  };

  btnProcess.textContent = "Starting…";
  btnProcess.disabled = true;

  fetch("/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, mask_png: maskPng, options }),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      showSection(secProgress);
      secOptions.classList.add("hidden");
      listenProgress();
    })
    .catch(err => {
      btnProcess.textContent = "Start Processing";
      btnProcess.disabled = false;
      showError(err.message);
    });
}

/* ── Progress SSE ──────────────────────────────────────────────────────── */
const progressBar  = document.getElementById("progress-bar");
const progressPct  = document.getElementById("progress-pct");
const progressMsg  = document.getElementById("progress-msg");
const progressLog  = document.getElementById("progress-log");
const downloadLink = document.getElementById("download-link");

function listenProgress() {
  const src = new EventSource(`/progress/${jobId}`);

  src.onmessage = (e) => {
    const d = JSON.parse(e.data);

    if (d.event === "ping") return;

    if (d.event === "progress") {
      const pct = d.progress || 0;
      progressBar.style.width = pct + "%";
      progressPct.textContent = pct + "%";
      progressMsg.textContent = d.message || "";
      progressLog.textContent += `[${pct}%] ${d.message}\n`;
      progressLog.scrollTop = progressLog.scrollHeight;
    }

    if (d.event === "done") {
      src.close();
      progressBar.style.width = "100%";
      progressPct.textContent = "100%";
      progressMsg.textContent = "Finished!";
      progressLog.textContent += "[100%] Done!\n";

      downloadLink.href = d.download_url;
      showSection(secDone);
    }

    if (d.event === "error") {
      src.close();
      showError(d.message);
    }
  };

  src.onerror = () => {
    src.close();
    showError("Connection to server lost. Check server logs.");
  };
}

/* ── Done / reset ──────────────────────────────────────────────────────── */
document.getElementById("btn-new").addEventListener("click", resetApp);
document.getElementById("btn-retry").addEventListener("click", resetApp);

function resetApp() {
  jobId     = null;
  videoMeta = null;
  ctxMask.clearRect(0, 0, DISPLAY_W, DISPLAY_H);
  ctxVideo.clearRect(0, 0, DISPLAY_W, DISPLAY_H);

  progressBar.style.width  = "0%";
  progressPct.textContent  = "0%";
  progressMsg.textContent  = "Initialising…";
  progressLog.textContent  = "";
  uploadInfo.classList.add("hidden");
  dropZone.querySelector("p").textContent =
    "Drag & drop a video here, or <span class=\"link\">click to browse</span>";

  document.getElementById("btn-process").textContent = "Start Processing";
  document.getElementById("btn-process").disabled    = false;

  fileInput.value = "";
  showSection(secUpload);
  secOptions.classList.add("hidden");
}

/* ── Error helper ──────────────────────────────────────────────────────── */
function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  showSection(secError);
}
