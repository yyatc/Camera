from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


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

    def open(self) -> None:
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
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("RTSP publisher is not opened.")
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
