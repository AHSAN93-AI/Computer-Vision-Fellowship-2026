"""
Sentinel Pose Intel — Application Package

Sets up structured logging on import so every module that does
``import logging; logger = logging.getLogger(__name__)``
automatically gets console + file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger with console and (optionally) file handlers.

    Called once at application startup from ``main.py``.  Safe to call
    multiple times — clears existing handlers first.

    Parameters
    ----------
    level:
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file:
        If provided, logs are also written to this file path.
        Parent directories are created automatically.
    """
    root = logging.getLogger()

    # Clear any handlers left over from previous calls or library defaults
    root.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── File handler (optional) ─────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Quieten noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    root.info("Logging initialised - level=%s, file=%s", level, log_file or "(none)")
