"""
app.timeutil — monotonic → wall-clock conversion helper.

Several modules (activity state machine, alert engine, pipeline) time
durations using ``time.monotonic()`` because it can't jump backwards
and is safe for measuring elapsed intervals. That's the right choice
for *durations*, but the dashboard and the SQLite tables need real
(wall-clock) timestamps to display "when did this happen" to a human.

This module computes a single offset at import time and exposes
``monotonic_to_wall()`` to convert a monotonic reading into an
approximate epoch timestamp, without disturbing any of the existing
duration math (which keeps using time.monotonic() directly).
"""

from __future__ import annotations

import time

_OFFSET = time.time() - time.monotonic()


def monotonic_to_wall(monotonic_ts: float) -> float:
    """Convert a ``time.monotonic()`` reading to an epoch timestamp."""
    return _OFFSET + monotonic_ts
