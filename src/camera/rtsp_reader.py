from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class RtspReader:
    def __init__(self, url: str, width: int, height: int) -> None:
        self._url = url
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open RTSP stream: {self._url}")

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
            self._cap.release()
