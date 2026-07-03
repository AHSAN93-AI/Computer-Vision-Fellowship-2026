# Vision Toolkit — Installation Guide

*How to run Vision Toolkit — from source or as a standalone executable*

## What's Included in This Repository

- `app.py` — main PyQt5 application (source code)
- `build.py` — script that packages `app.py` into a standalone `.exe` using PyInstaller
- `VisionToolkit.spec` — PyInstaller build specification
- `requirements.txt` — Python dependencies
- `README.md` — project overview and documentation
- `dist/VisionToolkit.exe` — the prebuilt, double-clickable Windows executable

## Option 1: Run from Source Code

Use this method if you have Python installed and want to run or modify the code directly.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd vision-toolkit-desktop

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

A native application window will open — no browser required.

## Option 2: Run the Standalone Executable (No Python Required)

Use this method if you just want to use the app without installing Python or any dependencies.

- Download `VisionToolkit.exe` from the `dist/` folder (or the GitHub Release, if uploaded there instead).
- Double-click `VisionToolkit.exe` to launch the app directly.
- If Windows SmartScreen shows an "unknown publisher" warning, click **More info → Run anyway**. This is expected for unsigned executables and is safe for this project.
- No installation, Python, or extra setup is needed — everything is bundled inside the `.exe`.

## Rebuilding the Executable Yourself (Optional)

If you modify `app.py` and want to regenerate the `.exe`:

```bash
python build.py
```

This installs PyInstaller if missing, then rebuilds `dist/VisionToolkit.exe` using the settings in `VisionToolkit.spec`.

## Adding These Files + the .exe to GitHub

Your current `.gitignore` excludes `*.spec` and the `dist/` folder, which would hide both `VisionToolkit.spec` and the `.exe` from the repository. Since this project requires both to be included, update `.gitignore` first, then commit everything:

```bash
# In .gitignore, remove or comment out these two lines:
# *.spec
# dist/
```

```bash
git add app.py build.py requirements.txt README.md VisionToolkit.spec
git add dist/VisionToolkit.exe
git commit -m "Add Vision Toolkit source code and executable"
git push origin main
```

> **Note:** GitHub blocks any single file over 100 MB when pushed directly. PyInstaller `--onefile` builds are typically 100–200 MB. If your `.exe` is rejected on push:
> - Use **Git LFS** (Large File Storage): `git lfs install`, then `git lfs track "*.exe"` before committing, **or**
> - Upload `VisionToolkit.exe` as a **GitHub Release asset** instead (Releases tab → "Draft a new release" → attach the `.exe`), and link to it from the README's Demo/Download section.

## Troubleshooting

- **"Python not found"** — install Python 3.11+ from python.org and ensure it's added to PATH.
- **Missing module errors** — re-run `pip install -r requirements.txt` inside your virtual environment.
- **App won't launch from `.exe`** — confirm you downloaded the full file (not corrupted/partial) and that you're on Windows, since this build is Windows-only.
- **Antivirus flags the `.exe`** — common false positive for PyInstaller-built apps; safe to allow/whitelist.
