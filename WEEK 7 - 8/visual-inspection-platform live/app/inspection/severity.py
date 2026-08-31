"""
Requirement 9: Defect Severity.

Rule-based, and deliberately kept independent of the model code so the
bands/floors can be retuned from Settings without retraining anything.
"""

from app.config import runtime_settings

SEVERITY_ORDER = ["None", "Minor", "Major", "Critical"]


def _rank(severity: str) -> int:
    return SEVERITY_ORDER.index(severity)


def classify_severity(defect_class: str, defect_area_ratio: float) -> str:
    """
    Combines an area-ratio band with a per-class floor (e.g. 'hole' can
    never be rated below 'Major', however small it measures).
    """
    settings = runtime_settings["severity"]

    if defect_area_ratio <= 0:
        band = "None"
    elif defect_area_ratio <= settings["minor_max_area_ratio"]:
        band = "Minor"
    elif defect_area_ratio <= settings["major_max_area_ratio"]:
        band = "Major"
    else:
        band = "Critical"

    floor = settings["minimum_severity_by_class"].get(defect_class)
    if floor and _rank(floor) > _rank(band):
        return floor
    return band


def max_severity(severities: list) -> str:
    if not severities:
        return "None"
    return max(severities, key=_rank)
