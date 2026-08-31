"""
Requirement 19: structured inspection result.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class InspectionResult:
    inspection_id: str
    timestamp: str
    product_type: str
    model_version: str
    status: str  # PASS / FAIL / INVALID
    defects_detected: List[str] = field(default_factory=list)
    defect_count: int = 0
    max_severity: str = "None"
    classifier_confidence: float = 0.0
    anomaly_score: float = 0.0
    defect_area_ratio: float = 0.0
    processing_time_ms: float = 0.0
    evidence_path: str = ""
    invalid_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
