"""
Build script — packages Vision Toolkit into a standalone executable
using PyInstaller.

Usage:
    python build.py

Output:
    dist/VisionToolkit(.exe on Windows)  -> a single double-clickable file
"""

import subprocess
import sys

def main():
    # Make sure PyInstaller is installed
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--name=VisionToolkit",
        "--onefile",          # bundle everything into a single executable
        "--windowed",         # no console window pops up behind the GUI
        "--noconfirm",
        "app.py",
    ], check=True)

    print("\nDone! Your app is at: dist/VisionToolkit(.exe)")

if __name__ == "__main__":
    main()
