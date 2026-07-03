# Vision Toolkit — Desktop Edition

A native desktop Computer Vision application built with **Python, OpenCV, and PyQt5**.
Developed as part of the Visibility Bots AI Summer Internship 2026 — Computer Vision Track.

## Features

- **Image Upload** — open any JPG/PNG/BMP/WEBP image via a native file dialog
- **Image Information** — width, height, resolution, file size, color channels
- **Grayscale Conversion**
- **Edge Detection** — Canny with adjustable lower/upper threshold sliders
- **Blur Filters** — Gaussian Blur, Median Blur (adjustable kernel size)
- **Thresholding** — Binary Threshold, Adaptive Threshold
- **Drawing Tools** — Rectangle, Circle, Line, Text (custom color, thickness, position)
- **Image Saving** — save the processed image via a native save dialog

### Bonus Features
- Brightness / Contrast adjustment
- Rotate / Resize
- Histogram visualization (per-channel)

All tools update the **Processed** preview live as you move the sliders. Drawing
tools require clicking **"Apply Drawing to Image"** to commit the shape onto the
working image (so you can layer multiple shapes before saving).

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd vision-toolkit-desktop

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

A native application window will open — no browser required.

## Project Structure

```
vision-toolkit-desktop/
├── app.py              # Main PyQt5 application
├── build.py             # Packages app.py into a standalone executable
├── requirements.txt    # Python dependencies
└── README.md
```

## Packaging into a Standalone Executable

By default you run this with `python app.py` every time. To turn it into a
real installable app that launches with a double-click (no terminal, no
"is Python installed?" step), package it with **PyInstaller**:

```bash
python build.py
```

This will:
1. Install PyInstaller if you don't already have it
2. Bundle `app.py` and all its dependencies into a single executable
3. Output the result to `dist/VisionToolkit.exe` (Windows) or `dist/VisionToolkit` (macOS/Linux)

You can then:
- **Windows:** double-click `VisionToolkit.exe` directly, or right-click it →
  "Send to → Desktop (create shortcut)" to pin it to your desktop.
- **macOS:** double-click `VisionToolkit` in `dist/`, or drag it into
  `/Applications` for it to show up like a normal app.
- **Linux:** run `chmod +x dist/VisionToolkit` once, then double-click it or
  run `./dist/VisionToolkit` from anywhere.

**Notes:**
- Build the executable **on the same OS you want to run it on** — a Windows
  build won't run on macOS/Linux and vice versa (PyInstaller doesn't cross-compile).
- The first build can take 1–3 minutes and the resulting file may be 100–200MB,
  since it bundles the Python interpreter and all libraries (OpenCV, PyQt5, etc.)
  inside it. This is normal.
- If Windows SmartScreen or macOS Gatekeeper warns that the app is from an
  "unknown publisher," that's expected for unsigned executables — click
  "More info → Run anyway" (Windows) or right-click → Open (macOS).
- You do **not** need to commit the `dist/` or `build/` folders to GitHub —
  add them to `.gitignore`. Only commit `app.py`, `build.py`,
  `requirements.txt`, and `README.md`; anyone can regenerate the executable
  by running `python build.py`.

## How It Works

- **UI Layer (PyQt5):** `QMainWindow` with a two-panel layout — image previews
  (Original / Processed) on the left, a tool selector and parameter controls
  (`QStackedWidget`) on the right.
- **Processing Layer (OpenCV):** every tool maps to a small block of OpenCV code
  in the `process()` method — e.g. `cv2.Canny`, `cv2.GaussianBlur`,
  `cv2.adaptiveThreshold`, `cv2.rectangle`, etc.
- **Image Conversion:** OpenCV images (BGR NumPy arrays) are converted to `QPixmap`
  for display via a small `cv_to_qpixmap()` helper.
- **State:** `working_img` holds the base image that drawing tools commit onto;
  `original_img` is kept untouched so you can always **Reset to Original**.

## Tech Stack

- **Python 3.11+**
- **OpenCV** — all image processing operations
- **PyQt5** — native desktop GUI framework
- **NumPy** — array operations
- **Matplotlib** — histogram rendering (embedded as an image, not a separate window)

## Author

Ahsan Rehman — The University of Faisalabad
Computer Vision Fellowship 2026, Visibility Bots Innovation Lab
