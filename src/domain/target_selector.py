from __future__ import annotations

from typing import List, Optional

from src.common.types import TrackedPerson


class TargetSelector:
    def choose_target(
        self,
        tracks: List[TrackedPerson],
        preferred_id: Optional[int] = None,
    ) -> Optional[TrackedPerson]:
        if not tracks:
            return None
        if preferred_id is not None:
            for track in tracks:
                if track.track_id == preferred_id:
                    return track
        # Берем самую уверенную цель.
        return max(tracks, key=lambda t: t.detection.confidence)
