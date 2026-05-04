from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.common.logging_setup import sanitize_rtsp_url

logger = logging.getLogger(__name__)

# Если нет нового кадра дольше этого времени — сообщаем об ошибке чтения.
_FRAME_STALE_SEC = 2.0


class RtspReader:
    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        open_timeout_sec: float = 8.0,
        *,
        extra_ffmpeg_capture_options: str | None = None,
    ) -> None:
        self._url = url
        self._width = width
        self._height = height
        self._open_timeout_sec = max(0.5, float(open_timeout_sec))
        self._extra_ffmpeg_capture_options = (extra_ffmpeg_capture_options or "").strip()
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        # TCP + низкая задержка: уменьшаем reorder/max_delay, probesize для быстрого старта.
        base = (
            "rtsp_transport;tcp|stimeout;5000000|max_delay;250000|reorder_queue_size;0|"
            "fflags;nobuffer|flags;low_delay|probesize;500000|analyzeduration;500000"
        )
        if self._extra_ffmpeg_capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = base + "|" + self._extra_ffmpeg_capture_options
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = base
        safe = sanitize_rtsp_url(self._url)
        logger.info("Открытие входного потока %s (целевой размер %dx%d)", safe, self._width, self._height)
        result: dict[str, object] = {}

        def _open_capture() -> None:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            result["cap"] = cap

        opener = threading.Thread(target=_open_capture, daemon=True, name="rtsp-open")
        opener.start()
        opener.join(timeout=self._open_timeout_sec)
        if opener.is_alive():
            logger.error("Таймаут открытия входного потока: %s (%.1f c)", safe, self._open_timeout_sec)
            raise RuntimeError(f"RTSP open timeout after {self._open_timeout_sec:.1f}s: {self._url}")

        cap = result.get("cap")
        if not isinstance(cap, cv2.VideoCapture):
            logger.error("Не удалось инициализировать VideoCapture: %s", safe)
            raise RuntimeError(f"Unable to init RTSP capture: {self._url}")
        self._cap = cap
        if not self._cap.isOpened():
            logger.error("Не удалось открыть входной поток: %s", safe)
            raise RuntimeError(f"Unable to open RTSP stream: {self._url}")
        self._reset_frame_state()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="rtsp-grabber")
        self._thread.start()
        logger.info("Входной поток открыт: %s", safe)

    def _grab_loop(self) -> None:
        """Непрерывно вычитывает кадры из буфера камеры, сохраняя только последний.

        Это устраняет накопление буфера OpenCV/FFmpeg: даже когда основной цикл
        работает медленнее потока камеры, grab_loop опустошает буфер и гарантирует,
        что read() возвращает свежий (не устаревший) кадр.
        """
        while self._running:
            cap = self._cap  # локальная ссылка: защищает от гонки с close()
            if cap is None:
                time.sleep(0.01)
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                if frame.shape[1] != self._width or frame.shape[0] != self._height:
                    frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_ts = time.monotonic()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            frame = self._latest_frame
            ts = self._last_frame_ts
        if frame is None or (time.monotonic() - ts) > _FRAME_STALE_SEC:
            # Небольшой sleep предотвращает spin-loop в основном цикле пока камера
            # ещё подключается или временно недоступна. Без него главный цикл
            # набирает 30 «ошибок» за микросекунды и немедленно переподключается.
            time.sleep(0.05)
            return False, None
        return True, frame

    def close(self) -> None:
        self._running = False
        # Освобождаем cap ДО join: это прерывает заблокированный cap.read() в grab_loop,
        # иначе join(2s) истечёт раньше чем stimeout=5s для cap.read().
        cap = self._cap
        self._cap = None
        if cap is not None:
            logger.info("Закрытие входного потока: %s", sanitize_rtsp_url(self._url))
            cap.release()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._reset_frame_state()

    def _reset_frame_state(self) -> None:
        with self._lock:
            self._latest_frame = None
            self._last_frame_ts = 0.0
