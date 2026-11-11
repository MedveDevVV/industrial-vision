from dataclasses import dataclass
from enum import Enum


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None
    in_roi: bool = False

@dataclass
class PPEDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    person_track_id: int | None = None
    
class PPEStatus(Enum):
    VIOLATION = "violation"
    COMPLIANT = "compliant"
    UNKNOWN = "unknown"


@dataclass
class PPEAssessment:
    helmet: PPEStatus
    vest: PPEStatus