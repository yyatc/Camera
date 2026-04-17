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
    ) -> np.ndarray:
        out = frame.copy()

        if target is not None:
            x1, y1, x2, y2 = target.detection.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"ID {target.track_id} / total {total_count}"
            cv2.putText(
                out,
                label,
                (x2 + 8, min(out.shape[0] - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Правая верхняя колонка с таймерами.
        x = out.shape[1] - 360
        y = 30
        cv2.putText(
            out,
            f"Total: {_fmt_hms(total_seconds)}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
        for person_id in sorted(per_person_seconds.keys()):
            sec = int(per_person_seconds[person_id])
            text = f"{person_id}: {sec} sec"
            cv2.putText(
                out,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        return out


def _fmt_hms(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
