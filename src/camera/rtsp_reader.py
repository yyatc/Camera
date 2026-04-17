from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from src.common.logging_setup import sanitize_rtsp_url

logger = logging.getLogger(__name__)


class RtspReader:
    def __init__(self, url: str, width: int, height: int) -> None:
        self._url = url
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        # Стабильнее для IP-камер: принудительно TCP-транспорт + таймауты.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|reorder_queue_size;0"
        )
        safe = sanitize_rtsp_url(self._url)
        logger.info("Открытие входного потока %s (целевой размер %dx%d)", safe, self._width, self._height)
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            logger.error("Не удалось открыть входной поток: %s", safe)
            raise RuntimeError(f"Unable to open RTSP stream: {self._url}")
        logger.info("Входной поток открыт: %s", safe)

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._cap:
            raise RuntimeError("RTSP reader is not opened.")
        ok, frame = self._cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        return ok, frame

    def close(self) -> None:
        if self._cap is not None:
            logger.info("Закрытие входного потока: %s", sanitize_rtsp_url(self._url))
            self._cap.release()
