# Watermark & Subtitle Remover

A **100% offline** Flask web application for removing moving/static watermarks and hard-coded subtitles from videos using [ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023).

---

## Features

- Drag-and-drop video upload (MP4, MOV, AVI, MKV, WEBM)
- Interactive canvas mask editor — freehand brush and rectangle tools
- One-click **Auto-Detect Subtitles** (heuristic OpenCV detection, no model needed)
- Real-time progress bar during inference
- FP16 half-precision mode to cut VRAM usage
- Configurable resolution, chunk size, and dilation
- All processing runs locally — no network calls

---

## Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.9 +   |
| PyTorch   | 2.0 +   |
| CUDA      | 11.8 + (recommended) or CPU (slow) |
| RAM       | 8 GB+   |
| VRAM      | 4 GB+ GPU recommended (see table below) |

**VRAM guidance (FP16):**

| Resolution | ~50 frames | ~80 frames |
|------------|-----------|-----------|
| 1280×720   | ~19 GB    | ~25 GB    |
| 720×480    | ~7 GB     | ~8 GB     |
| 640×480    | ~6 GB     | ~7 GB     |
| 320×240    | ~2 GB     | ~3 GB     |

Use the **Resize ratio** option (e.g. `0.5`) to process at half resolution if VRAM is limited.

---

## Project Structure

```
watermark-remover/
├── app.py                  # Flask application (routes, job management, SSE)
├── inference_engine.py     # ProPainter subprocess wrapper + progress tracking
├── subtitle_detector.py    # OpenCV heuristic subtitle region detector
├── requirements.txt        # Python dependencies
│
├── propainter/             # ← Clone ProPainter here (see step 2 below)
│   ├── inference_propainter.py
│   ├── model/
│   ├── core/
│   ├── utils/
│   └── RAFT/
│
├── model/                  # ← Place downloaded weight files here (step 3)
│   ├── ProPainter.pth
│   ├── recurrent_flow_completion.pth
│   └── raft-things.pth
│
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   └── index.html
├── uploads/                # Temporary upload storage (auto-created)
└── results/                # Processed video output (auto-created)
```

---

## Installation

### Step 1 — Clone this repository

```bash
git clone <this-repo-url> watermark-remover
cd watermark-remover
```

### Step 2 — Clone ProPainter

```bash
git clone https://github.com/sczhou/ProPainter.git propainter
```

The ProPainter source code must be present at `propainter/` inside this project folder.

### Step 3 — Download Model Weights

Download the three required `.pth` files and place them in the `model/` folder:

| File | Size | Download URL |
|------|------|-------------|
| `ProPainter.pth` | ~170 MB | [GitHub Release v0.1.0](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth) |
| `recurrent_flow_completion.pth` | ~37 MB | [GitHub Release v0.1.0](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth) |
| `raft-things.pth` | ~21 MB | [GitHub Release v0.1.0](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth) |

```
model/
├── ProPainter.pth
├── recurrent_flow_completion.pth
└── raft-things.pth
```

> **Offline use:** If you are on an air-gapped machine, download the files on another machine first and transfer them manually.

### Step 4 — Install Python dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install PyTorch first (choose the right CUDA version from https://pytorch.org)
# Example: CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Also install ProPainter's own requirements
pip install -r propainter/requirements.txt
```

---

## Running the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Usage Workflow

1. **Upload Video** — drag & drop or click to browse
2. **Draw Mask** — paint over the watermark/subtitle area on the preview frame
   - Use **Brush** for freehand painting
   - Use **Rectangle** for logos and banners
   - Use **Auto-Detect Subtitles** to automatically highlight subtitle regions
3. **Configure Options** — adjust VRAM/quality trade-offs if needed
4. **Start Processing** — watch the real-time progress bar
5. **Download** — click the download button when complete

---

## Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| FP16 | ✓ | Half-precision — halves VRAM use with minimal quality loss |
| Sub-video length | 80 | Frames per processing chunk. Lower = less VRAM |
| Neighbor length | 10 | Local temporal context. Lower = less VRAM |
| Mask dilation | 4 | Expands mask edges to prevent fringing |
| Resize ratio | 1.0 | 0.5 = process at half resolution (much less VRAM) |
| Output height/width | -1 | Override output resolution (-1 = keep original) |

---

## License

ProPainter is released under the [NTU S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE) — **non-commercial use only**.

This Flask application wrapper is MIT licensed.

---

## Credits

- **ProPainter** — [sczhou/ProPainter](https://github.com/sczhou/ProPainter)
  *"ProPainter: Improving Propagation and Transformer for Video Inpainting"*, ICCV 2023
