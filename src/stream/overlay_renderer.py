from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.common.types import TrackedPerson

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE_MAIN = 0.45
_SCALE_BBOX = 0.5
_THICK = 1
_LINE_GAP = 16
_MARGIN_RIGHT = 10
_MAX_LIST = 10


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
        h, w = out.shape[:2]

        if target is not None:
            x1, y1, x2, y2 = target.detection.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"ID {target.track_id} / total {total_count}"
            _put_text_right_of_box(out, label, x2, y2, h, _SCALE_BBOX)

        lines: list[str] = [f"Total: {_fmt_hms(total_seconds)}"]
        ordered_ids = _last_n_discovered_ids(first_seen_ts, _MAX_LIST)
        for pid in ordered_ids:
            sec = int(per_person_seconds.get(pid, 0.0))
            lines.append(f"{pid}: {sec} sec")

        y = 18
        for line in lines:
            _put_text_right(out, line, w, y, _SCALE_MAIN)
            y += _LINE_GAP

        return out


def _last_n_discovered_ids(first_seen_ts: Dict[int, float], n: int) -> list[int]:
    if not first_seen_ts or n <= 0:
        return []
    return sorted(first_seen_ts.keys(), key=lambda i: first_seen_ts[i], reverse=True)[:n]


def _put_text_right(img: np.ndarray, text: str, frame_w: int, y: int, scale: float) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, _THICK)
    x = frame_w - _MARGIN_RIGHT - tw
    cv2.putText(
        img,
        text,
        (max(0, x), y),
        _FONT,
        scale,
        (255, 255, 255),
        _THICK,
        cv2.LINE_AA,
    )


def _put_text_right_of_box(
    img: np.ndarray,
    text: str,
    box_right: int,
    box_bottom: int,
    frame_h: int,
    scale: float,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, _THICK)
    x = min(img.shape[1] - tw - 4, box_right + 6)
    y = min(frame_h - 4, box_bottom + th + 4)
    cv2.putText(
        img,
        text,
        (max(0, x), y),
        _FONT,
        scale,
        (255, 255, 255),
        _THICK,
        cv2.LINE_AA,
    )


def _fmt_hms(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
