from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from requests.auth import HTTPDigestAuth

from src.common.types import PTZCommand

logger = logging.getLogger(__name__)

# Не чаще (сек): ограничение нагрузки на камеру и сеть.
_MIN_SEND_INTERVAL_SEC = 0.05
# Повтор той же continuous-команды: многие Hikvision гасят движение без рефреша.
_CONTINUOUS_RESEND_SEC = 0.12


def _cmd_changed(a: PTZCommand, b: Optional[PTZCommand]) -> bool:
    if b is None:
        return True
    return (
        abs(a.pan_speed - b.pan_speed) > 0.004
        or abs(a.tilt_speed - b.tilt_speed) > 0.004
        or abs(a.zoom_speed - b.zoom_speed) > 0.004
    )


@dataclass
class IsapiPtzCapabilities:
    has_ptz: bool
    supports_continuous: bool
    supports_absolute: bool
    supports_presets: bool
    supports_status: bool


@dataclass
class IsapiPtzStatus:
    pan_deg: Optional[float]
    tilt_deg: Optional[float]
    zoom_ratio: Optional[float]
    raw_xml: str


class IsapiPtzClient:
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        channel_id: int,
        *,
        use_https: bool = False,
        timeout_sec: float = 2.0,
        verify_tls: bool = False,
    ) -> None:
        self._channel_id = int(channel_id)
        self._timeout_sec = max(0.5, float(timeout_sec))
        scheme = "https" if use_https else "http"
        self._base_url = f"{scheme}://{host}"
        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(username, password)
        self._session.verify = verify_tls

        self._http_lock = threading.RLock()
        self._cap = self._discover_capabilities()
        if not self._cap.has_ptz:
            raise RuntimeError(f"ISAPI PTZ is unavailable for channel {self._channel_id}")

        self._cmd_lock = threading.Lock()
        self._pending_cmd: Optional[PTZCommand] = None
        self._cmd_event = threading.Event()
        self._worker_active = True
        self._worker_thread = threading.Thread(
            target=self._cmd_worker,
            daemon=True,
            name="isapi-ptz-worker",
        )
        self._worker_thread.start()

    @property
    def capabilities(self) -> IsapiPtzCapabilities:
        return self._cap

    def safe_move(self, cmd: Optional[PTZCommand]) -> None:
        """Неблокирующая постановка continuous-команды (HTTP в фоне)."""
        if not self._cap.supports_continuous:
            raise RuntimeError("ISAPI PTZ continuous control is not supported")
        if cmd is None:
            with self._cmd_lock:
                self._pending_cmd = None
            self._cmd_event.set()
            return
        with self._cmd_lock:
            self._pending_cmd = cmd
        self._cmd_event.set()

    def stop(self) -> None:
        self._worker_active = False
        with self._cmd_lock:
            self._pending_cmd = None
        self._cmd_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=4.0)
        try:
            self._sync_put_stop()
        except Exception as exc:
            logger.debug("ISAPI PTZ final stop: %s", exc)

    def _cmd_worker(self) -> None:
        last_sent: Optional[PTZCommand] = None
        last_sent_ts = 0.0

        while self._worker_active:
            self._cmd_event.wait(timeout=0.05)
            self._cmd_event.clear()

            if not self._worker_active:
                break

            with self._cmd_lock:
                cmd = self._pending_cmd

            now = time.monotonic()

            if cmd is None:
                if last_sent is not None:
                    try:
                        self._sync_put_stop()
                    except Exception as exc:
                        logger.warning("ISAPI PTZ stop: %s", exc)
                    last_sent = None
                    last_sent_ts = now
                continue

            if (now - last_sent_ts) < _MIN_SEND_INTERVAL_SEC:
                self._cmd_event.set()
                continue

            need_send = _cmd_changed(cmd, last_sent) or (
                last_sent is not None
                and not _cmd_changed(cmd, last_sent)
                and (now - last_sent_ts) >= _CONTINUOUS_RESEND_SEC
            )
            if not need_send:
                continue

            try:
                self._sync_put_continuous(cmd)
                last_sent = cmd
                last_sent_ts = now
            except Exception as exc:
                logger.warning("ISAPI PTZ continuous: %s", exc)
                last_sent = None

        try:
            self._sync_put_stop()
        except Exception:
            pass

    def _sync_put_continuous(self, cmd: PTZCommand) -> None:
        pan = int(max(-1.0, min(1.0, cmd.pan_speed)) * 100)
        tilt = int(max(-1.0, min(1.0, cmd.tilt_speed)) * 100)
        zoom = int(max(-1.0, min(1.0, cmd.zoom_speed)) * 100)
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<PTZData version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
            f"<pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom>"
            "</PTZData>"
        )
        with self._http_lock:
            self._request(
                "PUT",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/continuous",
                data=payload,
                content_type="application/xml",
            )

    def _sync_put_stop(self) -> None:
        if not self._cap.supports_continuous:
            return
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<PTZData version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
            "<pan>0</pan><tilt>0</tilt><zoom>0</zoom>"
            "</PTZData>"
        )
        with self._http_lock:
            self._request(
                "PUT",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/continuous",
                data=payload,
                content_type="application/xml",
            )

    def move_absolute(
        self,
        *,
        pan_deg: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        zoom_ratio: Optional[float] = None,
    ) -> None:
        if not self._cap.supports_absolute:
            raise RuntimeError("ISAPI PTZ absolute control is not supported")
        with self._http_lock:
            with self._cmd_lock:
                self._pending_cmd = None
            self._sync_put_stop()
            fields = []
            if tilt_deg is not None:
                tilt = int(max(-90.0, min(270.0, tilt_deg)) * 10)
                fields.append(f"<elevation>{tilt}</elevation>")
            if pan_deg is not None:
                pan = int((pan_deg % 360.0) * 10)
                fields.append(f"<azimuth>{pan}</azimuth>")
            if zoom_ratio is not None:
                zoom = int(max(0.0, min(1.0, zoom_ratio)) * 1000)
                fields.append(f"<absoluteZoom>{zoom}</absoluteZoom>")
            if not fields:
                return
            payload = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<PTZData version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
                f"<AbsoluteHigh>{''.join(fields)}</AbsoluteHigh>"
                "</PTZData>"
            )
            self._request(
                "PUT",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/absolute",
                data=payload,
                content_type="application/xml",
            )

    def goto_preset(self, preset_id: int) -> None:
        if not self._cap.supports_presets:
            raise RuntimeError("ISAPI PTZ presets are not supported")
        with self._http_lock:
            with self._cmd_lock:
                self._pending_cmd = None
            self._sync_put_stop()
            self._request(
                "PUT",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/presets/{int(preset_id)}/goto",
            )

    def get_status(self) -> Optional[IsapiPtzStatus]:
        if not self._cap.supports_status:
            return None
        with self._http_lock:
            xml_text = self._request(
                "GET",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/status",
                allow_fail=True,
            )
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
            pan_raw = self._find_int(root, "azimuth")
            tilt_raw = self._find_int(root, "elevation")
            zoom_raw = self._find_int(root, "absoluteZoom")
            return IsapiPtzStatus(
                pan_deg=(pan_raw / 10.0) if pan_raw is not None else None,
                tilt_deg=(tilt_raw / 10.0) if tilt_raw is not None else None,
                zoom_ratio=(zoom_raw / 1000.0) if zoom_raw is not None else None,
                raw_xml=xml_text,
            )
        except Exception:
            logger.debug("ISAPI PTZ status parse failed")
            return None

    def _discover_capabilities(self) -> IsapiPtzCapabilities:
        ch_ok = self._probe(f"/ISAPI/PTZCtrl/channels/{self._channel_id}")
        cap_text = ""
        if ch_ok:
            text = self._request(
                "GET",
                f"/ISAPI/PTZCtrl/channels/{self._channel_id}/capabilities",
                allow_fail=True,
            )
            cap_text = text or ""

        content = cap_text.lower()
        supports_cont = ("continuous" in content) or ch_ok
        supports_abs = ("absolute" in content) or self._probe(
            f"/ISAPI/PTZCtrl/channels/{self._channel_id}/absoluteEx"
        )
        supports_presets = ("preset" in content) or self._probe(
            f"/ISAPI/PTZCtrl/channels/{self._channel_id}/presets"
        )
        supports_status = ("status" in content) or self._probe(
            f"/ISAPI/PTZCtrl/channels/{self._channel_id}/status"
        )
        return IsapiPtzCapabilities(
            has_ptz=ch_ok,
            supports_continuous=supports_cont,
            supports_absolute=supports_abs,
            supports_presets=supports_presets,
            supports_status=supports_status,
        )

    def _find_int(self, root: ET.Element, tag_name: str) -> Optional[int]:
        for elem in root.iter():
            if elem.tag.endswith(tag_name):
                try:
                    return int(str(elem.text).strip())
                except Exception:
                    return None
        return None

    def _probe(self, path: str, retries: int = 2) -> bool:
        """Probe endpoint availability with retries to avoid transient failures."""
        import time as _time
        for attempt in range(retries):
            result = self._request("GET", path, allow_fail=True)
            if result is not None:
                return True
            if attempt < retries - 1:
                _time.sleep(0.3)
        return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: str | None = None,
        content_type: str = "application/xml",
        allow_fail: bool = False,
    ) -> Optional[str]:
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.request(
                method=method,
                url=url,
                data=data,
                headers={"Content-Type": content_type},
                timeout=self._timeout_sec,
            )
            if 200 <= resp.status_code < 300:
                return resp.text
            if allow_fail:
                logger.warning("ISAPI %s %s returned HTTP %s", method, path, resp.status_code)
                return None
            raise RuntimeError(f"ISAPI {method} {path} failed: HTTP {resp.status_code}")
        except Exception as exc:
            if allow_fail:
                logger.debug("ISAPI %s %s failed: %s", method, path, exc)
                return None
            raise
