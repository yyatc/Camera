from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from onvif import ONVIFCamera

from src.common.types import PTZCommand

logger = logging.getLogger(__name__)

# Минимальное изменение скорости для повторной отправки команды.
# Если новая команда отличается меньше — не тратим время на SOAP-вызов.
_CMD_CHANGE_THRESHOLD = 0.015

# Минимальный интервал между отправками (максимум ~15 команд/сек).
_MIN_SEND_INTERVAL_SEC = 0.067


@dataclass
class PTZCapabilities:
    has_ptz: bool
    can_zoom: bool


def _cmd_changed(a: PTZCommand, b: Optional[PTZCommand]) -> bool:
    """True если команда a существенно отличается от b."""
    if b is None:
        return True
    return (
        abs(a.pan_speed - b.pan_speed) > _CMD_CHANGE_THRESHOLD
        or abs(a.tilt_speed - b.tilt_speed) > _CMD_CHANGE_THRESHOLD
        or abs(a.zoom_speed - b.zoom_speed) > _CMD_CHANGE_THRESHOLD
    )


class OnvifPtzClient:
    def __init__(self, host: str, username: str, password: str, port: int = 80) -> None:
        self._cam = ONVIFCamera(host, port, username, password)
        self._media = self._cam.create_media_service()
        self._ptz = self._cam.create_ptz_service()

        self._profile = self._media.GetProfiles()[0]
        self._token = self._profile.token
        self._ptz_config = self._profile.PTZConfiguration
        self._velocity_space = self._ptz_config.DefaultContinuousPanTiltVelocitySpace
        self._zoom_space = self._ptz_config.DefaultContinuousZoomVelocitySpace

        # Асинхронный воркер: safe_move() не блокирует основной цикл.
        self._cmd_lock = threading.Lock()
        self._onvif_lock = threading.Lock()  # защищает SOAP-вызовы от гонок
        self._pending_cmd: Optional[PTZCommand] = None
        self._cmd_event = threading.Event()
        self._worker_active = True
        self._worker_thread = threading.Thread(
            target=self._cmd_worker, daemon=True, name="ptz-worker"
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def capabilities(self) -> PTZCapabilities:
        return PTZCapabilities(
            has_ptz=self._ptz_config is not None,
            can_zoom=bool(self._zoom_space),
        )

    def safe_move(self, cmd: Optional[PTZCommand]) -> None:
        """Неблокирующая постановка команды в очередь.

        Воркер отправит команду по ONVIF только если она изменилась.
        Вызов возвращается немедленно — основной цикл не блокируется.
        """
        if cmd is None:
            self.stop()
            return
        with self._cmd_lock:
            self._pending_cmd = cmd
        self._cmd_event.set()

    def continuous_move(self, cmd: PTZCommand) -> None:
        """Совместимость: маршрутизирует через воркер."""
        self.safe_move(cmd)

    def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        """Синхронная остановка камеры и завершение воркера (блокирующий вызов)."""
        with self._cmd_lock:
            self._pending_cmd = None
        self._worker_active = False
        self._cmd_event.set()  # разбудить воркер чтобы он увидел _worker_active=False
        with self._onvif_lock:
            self._do_stop(pan_tilt=pan_tilt, zoom=zoom)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _cmd_worker(self) -> None:
        """Воркер-поток: отправляет PTZ-команды без блокировки основного цикла."""
        last_sent: Optional[PTZCommand] = None
        last_sent_ts: float = 0.0

        while self._worker_active:
            self._cmd_event.wait(timeout=0.1)
            self._cmd_event.clear()

            if not self._worker_active:
                break

            with self._cmd_lock:
                cmd = self._pending_cmd

            if cmd is None:
                continue

            # Пропускаем если интервал ещё не истёк.
            now = time.monotonic()
            if (now - last_sent_ts) < _MIN_SEND_INTERVAL_SEC:
                continue

            # Пропускаем если команда не изменилась (кадр уже центрирован).
            if not _cmd_changed(cmd, last_sent):
                continue

            try:
                with self._onvif_lock:
                    self._do_continuous_move(cmd)
                last_sent = cmd
                last_sent_ts = now
            except Exception as exc:
                logger.warning("PTZ команда не выполнена: %s", exc)
                last_sent = None

    def _do_continuous_move(self, cmd: PTZCommand) -> None:
        request = self._ptz.create_type("ContinuousMove")
        request.ProfileToken = self._token
        request.Velocity = {
            "PanTilt": {"x": cmd.pan_speed, "y": cmd.tilt_speed},
            "Zoom": {"x": cmd.zoom_speed},
        }
        self._ptz.ContinuousMove(request)

    def _do_stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        request = self._ptz.create_type("Stop")
        request.ProfileToken = self._token
        request.PanTilt = pan_tilt
        request.Zoom = zoom
        self._ptz.Stop(request)
