"""
Requirement 20: Evidence Capture.

Saves original / heatmap / annotated images to evidence/<inspection_id>/
and returns the relative path stored on the inspection record.
"""

import os
import cv2
import numpy as np

from app.config import EVIDENCE_DIR


def evidence_dir_for(inspection_id: str) -> str:
    """Deterministic, unique-per-inspection-id path (Requirement: path generation)."""
    return os.path.join(EVIDENCE_DIR, inspection_id)


def save_evidence(
    inspection_id: str,
    original_gray: np.ndarray,
    error_map: np.ndarray,
    status: str,
    max_severity: str,
    predicted_class: str,
) -> str:
    """
    Writes original.png, heatmap.png, annotated.png into evidence/<id>/.
    Returns the evidence directory path (relative to project root, for
    storage in the DB / JSON result).
    """
    out_dir = evidence_dir_for(inspection_id)
    os.makedirs(out_dir, exist_ok=True)

    # 1. original
    cv2.imwrite(os.path.join(out_dir, "original.png"), original_gray)

    # 2. heatmap -- normalize the reconstruction-error map to 0-255 and
    #    apply a color map so hot regions (high error) pop visually.
    norm = error_map - error_map.min()
    denom = norm.max() if norm.max() > 0 else 1.0
    norm = (norm / denom * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(
        heatmap, (original_gray.shape[1], original_gray.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(os.path.join(out_dir, "heatmap.png"), heatmap_resized)

    # 3. annotated -- original + a small stamped label/severity overlay,
    #    and the heatmap blended in translucently so a reviewer can see
    #    both the verdict and where it came from at a glance.
    bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(bgr, 0.65, heatmap_resized, 0.35, 0)

    color = (60, 200, 60) if status == "PASS" else (50, 50, 230)
    label = f"{status}"
    if status == "FAIL":
        label += f" | {predicted_class} | {max_severity}"
    cv2.rectangle(blended, (0, 0), (blended.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        blended, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(os.path.join(out_dir, "annotated.png"), blended)

    return os.path.relpath(out_dir, os.path.dirname(EVIDENCE_DIR))
