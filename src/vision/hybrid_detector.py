from __future__ import annotations

from typing import List

import numpy as np

from src.common.types import Detection
from src.vision.local_person_detector import LocalPersonDetector


class CameraAnalyticsAdapter:
    """
    Заглушка для встроенной аналитики камеры.
    В реальном внедрении сюда подключается получение событий/боксов
    от вендор-API (или ONVIF events, если камера действительно отдает bbox).
    """

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


class HybridDetector:
    def __init__(
        self,
        local_detector: LocalPersonDetector,
        camera_adapter: CameraAnalyticsAdapter,
        min_confidence: float,
    ) -> None:
        self._local = local_detector
        self._camera = camera_adapter
        self._min_confidence = min_confidence

    def detect(self, frame: np.ndarray) -> List[Detection]:
        camera_dets = [d for d in self._camera.detect(frame) if d.confidence >= self._min_confidence]
        if camera_dets:
            return camera_dets
        return [d for d in self._local.detect(frame) if d.confidence >= self._min_confidence]
