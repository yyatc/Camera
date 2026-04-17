from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    source: str


@dataclass
class TrackedPerson:
    track_id: int
    detection: Detection
    last_seen_ts: float


@dataclass
class FrameContext:
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int


@dataclass
class PTZCommand:
    pan_speed: float
    tilt_speed: float
    zoom_speed: float


@dataclass
class TrackingSnapshot:
    mode: str
    current_target_id: Optional[int]
    total_people_count: int
    per_person_seconds: Dict[int, float]
    total_observation_seconds: float
