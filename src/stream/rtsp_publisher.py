from __future__ import annotations

import logging
import queue
import subprocess
import threading
from typing import Optional

import numpy as np

from src.common.logging_setup import sanitize_rtsp_url

logger = logging.getLogger(__name__)



def _detect_encoder(ffmpeg_bin: str) -> list:
    """
    Returns libx264 encoder args.
    NVENC disabled: libnvidia-encode.so.1 is not available in the container.
    To re-enable NVENC add 'video' to capabilities in docker-compose.yml
    and change this function to probe h264_nvenc first.
    """
    logger.info("Encoder: libx264 (CPU)")
    return [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-x264-params", "sync-lookahead=0:rc-lookahead=0",
    ]

class RtspPublisher:
    """Публикация raw BGR в ffmpeg → RTSP. Очередь на 1 кадр: при отставании кодера отбрасываем старые кадры."""

    def __init__(
        self,
        ffmpeg_bin: str,
        output_rtsp_url: str,
        width: int,
        height: int,
        fps: int,
        *,
        queue_size: int = 1,
    ) -> None:
        self._ffmpeg_bin = ffmpeg_bin
        self._output_rtsp_url = output_rtsp_url
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_size = max(1, int(queue_size))
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self._queue_size)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop = threading.Event()

    def open(self) -> None:
        safe_out = sanitize_rtsp_url(self._output_rtsp_url)
        logger.info(
            "Запуск выходного кодировщика: %s %dx%d @ %d fps (очередь кадров=%s)",
            safe_out,
            self._width,
            self._height,
            self._fps,
            self._queue_size,
        )
        encoder_args = _detect_encoder(self._ffmpeg_bin)
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
            *encoder_args,
            "-bf",
            "0",
            "-g",
            "1",  # keyframe every frame — minimizes decoder buffer delay
            "-keyint_min",
            "1",
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
        self._writer_stop.clear()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_ffmpeg_stderr, daemon=True)
        self._stderr_thread.start()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="ffmpeg-writer")
        self._writer_thread.start()
        logger.info("ffmpeg запущен (pid=%s), публикация в %s", self._proc.pid, safe_out)

    def _writer_loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        while not self._writer_stop.is_set():
            try:
                frame = self._frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._writer_stop.is_set():
                break
            if self._proc.poll() is not None:
                logger.error("Процесс ffmpeg завершился с кодом %s", self._proc.returncode)
                break
            try:
                stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                logger.warning("Запись в ffmpeg: %s", exc)
                break

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
        payload = frame.copy()
        try:
            self._frame_q.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        safe_out = sanitize_rtsp_url(self._output_rtsp_url)
        logger.info("Остановка выходного потока: %s", safe_out)
        self._writer_stop.set()
        # Сначала закрываем stdin ffmpeg — иначе writer может бесконечно ждать записи в полный буфер.
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=4.0)
            self._writer_thread = None
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            logger.info("ffmpeg завершён (код %s)", self._proc.returncode)
        self._proc = None
