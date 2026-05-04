from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.common.types import TrackedPerson

class OverlayRenderer:
    def render(
        self,
        frame: np.ndarray,
        target: Optional[TrackedPerson],
        total_count: int,
        per_person_seconds: Dict[int, float],
        total_seconds: float,
        first_seen_ts: Dict[int, float],
    ) -> np.ndarray:
        out = frame.copy()

        if target is not None:
            x1, y1, x2, y2 = target.detection.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return out
