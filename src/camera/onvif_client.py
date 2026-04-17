from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from onvif import ONVIFCamera

from src.common.types import PTZCommand


@dataclass
class PTZCapabilities:
    has_ptz: bool
    can_zoom: bool


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

    def capabilities(self) -> PTZCapabilities:
        return PTZCapabilities(
            has_ptz=self._ptz_config is not None,
            can_zoom=bool(self._zoom_space),
        )

    def continuous_move(self, cmd: PTZCommand) -> None:
        request = self._ptz.create_type("ContinuousMove")
        request.ProfileToken = self._token
        request.Velocity = {
            "PanTilt": {"x": cmd.pan_speed, "y": cmd.tilt_speed},
            "Zoom": {"x": cmd.zoom_speed},
        }
        self._ptz.ContinuousMove(request)

    def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        request = self._ptz.create_type("Stop")
        request.ProfileToken = self._token
        request.PanTilt = pan_tilt
        request.Zoom = zoom
        self._ptz.Stop(request)

    def safe_move(self, cmd: Optional[PTZCommand]) -> None:
        if cmd is None:
            self.stop()
            return
        self.continuous_move(cmd)
