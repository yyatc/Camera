from __future__ import annotations

import logging
import subprocess
import threading
from typing import Optional

import numpy as np

from src.common.logging_setup import sanitize_rtsp_url

logger = logging.getLogger(__name__)


class RtspPublisher:
    def __init__(
        self,
        ffmpeg_bin: str,
        output_rtsp_url: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._ffmpeg_bin = ffmpeg_bin
        self._output_rtsp_url = output_rtsp_url
        self._width = width
        self._height = height
        self._fps = fps
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def open(self) -> None:
        safe_out = sanitize_rtsp_url(self._output_rtsp_url)
        logger.info(
            "Запуск выходного кодировщика: %s %dx%d @ %d fps",
            safe_out,
            self._width,
            self._height,
            self._fps,
        )
        cmd = [
            self._ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-bf",
            "0",
            "-g",
            str(self._fps),
            "-keyint_min",
            str(self._fps),
            "-pix_fmt",
            "yuv420p",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self._output_rtsp_url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_ffmpeg_stderr, daemon=True)
        self._stderr_thread.start()
        logger.info("ffmpeg запущен (pid=%s), публикация в %s", self._proc.pid, safe_out)

    def _drain_ffmpeg_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for raw in iter(self._proc.stderr.readline, b""):
            if not raw:
                break
            line = raw.decode(errors="replace").strip()
            if line:
                logger.warning("ffmpeg: %s", line)

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("RTSP publisher is not opened.")
        if self._proc.poll() is not None:
            logger.error("Процесс ffmpeg завершился с кодом %s", self._proc.returncode)
            raise RuntimeError("ffmpeg process exited; cannot write frames")
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        safe_out = sanitize_rtsp_url(self._output_rtsp_url)
        logger.info("Остановка выходного потока: %s", safe_out)
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            logger.info("ffmpeg завершён (код %s)", self._proc.returncode)
