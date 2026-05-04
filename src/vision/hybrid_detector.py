from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.common.types import Detection

logger = logging.getLogger(__name__)
from src.vision.local_person_detector import LocalPersonDetector


class CameraAnalyticsAdapter:
    """
    Заглушка для встроенной аналитики камеры.
    В реальном внедрении сюда подключается получение событий/боксов
    от вендор-API (или ONVIF events, если камера действительно отдает bbox).
    """

    def __init__(self, event_source: object | None = None) -> None:
        self._event_source = event_source

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._event_source is None:
            return []
        try:
            return self._event_source.detect((frame.shape[0], frame.shape[1]))
        except Exception as exc:
            logger.debug("CameraAnalyticsAdapter detect failed: %s", exc)
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
        self._camera_hits = 0
        self._local_hits = 0
        self._camera_dets_total = 0
        self._local_dets_total = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        camera_dets = [d for d in self._camera.detect(frame) if d.confidence >= self._min_confidence]
        if camera_dets:
            self._camera_hits += 1
            self._camera_dets_total += len(camera_dets)
            logger.debug("Детекция: камера, объектов=%s", len(camera_dets))
            return camera_dets
        out = [d for d in self._local.detect(frame) if d.confidence >= self._min_confidence]
        self._local_hits += 1
        self._local_dets_total += len(out)
        logger.debug("Детекция: локально, объектов=%s", len(out))
        return out

    def stats_snapshot(self) -> dict:
        total_cycles = self._camera_hits + self._local_hits
        camera_share = (self._camera_hits / total_cycles) if total_cycles > 0 else 0.0
        return {
            "camera_hits": self._camera_hits,
            "local_hits": self._local_hits,
            "camera_dets_total": self._camera_dets_total,
            "local_dets_total": self._local_dets_total,
            "camera_share": camera_share,
        }
