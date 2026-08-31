"""
Sentinel Pose Intel — Flask Application Entry Point

This module is the single entry point for the backend server.
Run with:
    python -m app.main
or:
    flask --app app.main run --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, render_template

from app import setup_logging
from app.config import get_settings
from app.dashboard.routes import bp as dashboard_bp

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Bootstrap logging ───────────────────────────────────
_settings = get_settings()
setup_logging(level=_settings.log_level, log_file=str(_settings.log_file_abs_path))
logger = logging.getLogger(__name__)

# ── Create Flask app ─────────────────────────────────────
app = Flask(
    __name__,
    static_folder=str(_PROJECT_ROOT / "static"),
    template_folder=str(_PROJECT_ROOT / "templates"),
)

# ── Register API routes (REST + MJPEG stream) ────────────
app.register_blueprint(dashboard_bp)

# ── Serve evidence images ────────────────────────────────
_evidence_dir = _settings.evidence_abs_path
_evidence_dir.mkdir(parents=True, exist_ok=True)

# ── Ensure the uploaded-video directory exists ───────────
_settings.upload_abs_path.mkdir(parents=True, exist_ok=True)

# ── Cap request size to match the configured upload limit ──
app.config["MAX_CONTENT_LENGTH"] = _settings.upload_max_bytes + (5 * 1024 * 1024)


@app.route("/evidence/<path:filename>")
def serve_evidence(filename: str):
    from flask import send_from_directory
    return send_from_directory(str(_evidence_dir), filename)


# ── Serve the plain HTML/CSS/JS dashboard ────────────────
@app.route("/")
def index():
    return render_template("index.html")


logger.info("Sentinel Pose Intel Flask app ready")


def main() -> None:
    settings = get_settings()
    app.run(host=settings.api_host, port=settings.api_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
