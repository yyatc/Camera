from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Tuple
import logging
import time

from src.common.types import Detection, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class _TrackState:
    person: TrackedPerson
    center: Tuple[float, float]


class PersonTracker:
    def __init__(self, max_distance_px: float = 120.0, timeout_sec: float = 2.0) -> None:
        self._max_distance = max_distance_px
        self._timeout_sec = timeout_sec
        self._next_track_id = 1
        self._tracks: Dict[int, _TrackState] = {}
        self._total_seen_unique = 0
        self._first_seen_ts: Dict[int, float] = {}

    @property
    def total_seen_unique(self) -> int:
        return self._total_seen_unique

    @property
    def first_seen_ts(self) -> Dict[int, float]:
        return dict(self._first_seen_ts)

    def update(self, detections: List[Detection], ts: float | None = None) -> List[TrackedPerson]:
        now = ts if ts is not None else time.time()
        assigned: Dict[int, Detection] = {}
        available_track_ids = list(self._tracks.keys())

        for det in detections:
            det_center = _center(det.bbox)
            best_id = None
            best_dist = 10**9
            for track_id in available_track_ids:
                tr = self._tracks[track_id]
                dist = hypot(det_center[0] - tr.center[0], det_center[1] - tr.center[1])
                if dist < best_dist and dist <= self._max_distance:
                    best_dist = dist
                    best_id = track_id
            if best_id is not None:
                assigned[best_id] = det
                available_track_ids.remove(best_id)
            else:
                new_id = self._next_track_id
                self._next_track_id += 1
                self._total_seen_unique += 1
                self._first_seen_ts[new_id] = now
                assigned[new_id] = det
                logger.info(
                    "Новый объект в кадре: id=%s (уникальных за сессию: %s)",
                    new_id,
                    self._total_seen_unique,
                )

        # Обновляем/создаем треки.
        updated_track_ids = set()
        for track_id, det in assigned.items():
            person = TrackedPerson(track_id=track_id, detection=det, last_seen_ts=now)
            self._tracks[track_id] = _TrackState(person=person, center=_center(det.bbox))
            updated_track_ids.add(track_id)

        # Чистим старые треки.
        expired = []
        for track_id, state in self._tracks.items():
            if now - state.person.last_seen_ts > self._timeout_sec:
                expired.append(track_id)
        for track_id in expired:
            logger.debug("Трек id=%s снят по таймауту (>%s с)", track_id, self._timeout_sec)
            self._tracks.pop(track_id, None)

        # Возвращаем все живые треки (включая недавно виденных, но не попавших в текущий кадр).
        # Это позволяет PTZ удерживать цель во время временных пропусков детекции
        # вместо мгновенного переключения в SEARCHING при каждом пропуске YOLO.
        return [state.person for state in self._tracks.values()]


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
