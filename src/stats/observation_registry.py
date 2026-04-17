from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional
import time


class ObservationRegistry:
    def __init__(self) -> None:
        self._person_seconds: Dict[int, float] = defaultdict(float)
        self._total_seconds: float = 0.0
        self._last_ts: Optional[float] = None
        self._last_target_id: Optional[int] = None

    def tick(self, current_target_id: Optional[int], ts: float | None = None) -> None:
        now = ts if ts is not None else time.time()
        if self._last_ts is None:
            self._last_ts = now
            self._last_target_id = current_target_id
            return

        delta = max(0.0, now - self._last_ts)
        self._total_seconds += delta
        if self._last_target_id is not None:
            self._person_seconds[self._last_target_id] += delta

        self._last_ts = now
        self._last_target_id = current_target_id

    @property
    def total_seconds(self) -> float:
        return self._total_seconds

    @property
    def per_person_seconds(self) -> Dict[int, float]:
        return dict(self._person_seconds)

    def known_ids_sorted(self) -> Iterable[int]:
        return sorted(self._person_seconds.keys())
