from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from random import uniform
from typing import Optional, Tuple

from src.common.types import PTZCommand, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class PtzPolicyConfig:
    pan_gain: float
    tilt_gain: float
    zoom_gain: float
    max_pan_speed: float
    max_tilt_speed: float
    max_zoom_speed: float
    center_tolerance_x: float
    center_tolerance_y: float
    target_area_ratio: float
    zoom_hysteresis: float
    search_pan_speed: float
    search_tilt_speed: float
    search_zoom_out_speed: float
    # Минимальная скорость при которой мотор физически начинает двигаться.
    # Команды ниже этого порога (но ненулевые) поднимаются до минимума.
    min_effective_pan_speed: float = 0.0
    min_effective_tilt_speed: float = 0.0


class PtzControlPolicy:
    def __init__(self, cfg: PtzPolicyConfig) -> None:
        self._cfg = cfg

    def tracking_command(
        self,
        target: TrackedPerson,
        frame_size: Tuple[int, int],
    ) -> PTZCommand:
        frame_w, frame_h = frame_size
        x1, y1, x2, y2 = target.detection.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        err_x = (cx - frame_w / 2.0) / frame_w
        err_y = (cy - frame_h / 2.0) / frame_h
        area_ratio = (w * h) / float(frame_w * frame_h)
        zoom_err = self._cfg.target_area_ratio - area_ratio

        pan_speed = 0.0 if abs(err_x) < self._cfg.center_tolerance_x else err_x * self._cfg.pan_gain
        tilt_speed = 0.0 if abs(err_y) < self._cfg.center_tolerance_y else -err_y * self._cfg.tilt_gain
        zoom_speed = 0.0 if abs(zoom_err) < self._cfg.zoom_hysteresis else zoom_err * self._cfg.zoom_gain

        pan_speed = _clip(_apply_min_speed(pan_speed, self._cfg.min_effective_pan_speed), self._cfg.max_pan_speed)
        tilt_speed = _clip(_apply_min_speed(tilt_speed, self._cfg.min_effective_tilt_speed), self._cfg.max_tilt_speed)
        zoom_speed = _clip(zoom_speed, self._cfg.max_zoom_speed)

        logger.debug(
            "PTZ tracking: err=(%.3f, %.3f) area=%.3f → pan=%.3f tilt=%.3f zoom=%.3f",
            err_x, err_y, area_ratio, pan_speed, tilt_speed, zoom_speed,
        )

        return PTZCommand(pan_speed=pan_speed, tilt_speed=tilt_speed, zoom_speed=zoom_speed)

    def search_command(self, reset_zoom: bool = False) -> Optional[PTZCommand]:
        pan = uniform(-self._cfg.search_pan_speed, self._cfg.search_pan_speed)
        tilt = uniform(-self._cfg.search_tilt_speed, self._cfg.search_tilt_speed)
        zoom = self._cfg.search_zoom_out_speed if reset_zoom else 0.0
        return PTZCommand(pan_speed=pan, tilt_speed=tilt, zoom_speed=zoom)

    def monitoring_command(self, ts: float) -> PTZCommand:
        """
        Мягкий мониторинг: плавные pan/tilt колебания + импульсный zoom-out.
        """
        jx = uniform(-0.20, 0.20)
        jy = uniform(-0.20, 0.20)
        pan = 0.7 * self._cfg.search_pan_speed * math.sin(ts * 0.24) + jx * self._cfg.search_pan_speed
        tilt = 0.7 * self._cfg.search_tilt_speed * math.cos(ts * 0.20) + jy * self._cfg.search_tilt_speed
        pan = _clip(pan, self._cfg.max_pan_speed)
        tilt = _clip(tilt, self._cfg.max_tilt_speed)
        zoom_speed = self._cfg.search_zoom_out_speed if int(ts * 2) % 4 == 0 else 0.0
        return PTZCommand(pan_speed=pan, tilt_speed=tilt, zoom_speed=zoom_speed)


def _apply_min_speed(speed: float, min_speed: float) -> float:
    """Если скорость ненулевая, но меньше минимума — поднять до минимума.
    Это компенсирует статическое трение мотора камеры."""
    if speed == 0.0 or min_speed <= 0.0:
        return speed
    return math.copysign(max(abs(speed), min_speed), speed)


def _clip(value: float, lim: float) -> float:
    return max(-lim, min(lim, value))
