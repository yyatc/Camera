from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from requests.auth import HTTPDigestAuth

from src.common.types import Detection

logger = logging.getLogger(__name__)

_HUMAN_EVENT_KEYS = (
    "humandetection",
    "humanrecognition",
    "targetcapture",
    "mixedtargetdetection",
    "mtd_human",
    "person",
    # На ряде камер "человек" приходит через smart eventType без явного human в targetType.
    "fielddetection",
    "linedetection",
    "regionentrance",
    "regionexiting",
    "facedetection",
)


@dataclass
class _EventDetection:
    ts: float
    confidence: float
    # normalised xyxy in [0,1], if available
    bbox_norm: Optional[Tuple[float, float, float, float]]


class IsapiEventStreamClient:
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        use_https: bool = False,
        verify_tls: bool = False,
        connect_timeout_sec: float = 2.0,
        read_timeout_sec: float = 15.0,
        max_age_sec: float = 1.2,
        debug_samples_enabled: bool = False,
        debug_samples_limit: int = 10,
        capability_probe_enabled: bool = True,
        auto_configure_enabled: bool = False,
        auto_configure_dry_run: bool = False,
    ) -> None:
        self._base_url = f"{'https' if use_https else 'http'}://{host}"
        self._verify_tls = verify_tls
        self._connect_timeout = max(0.5, float(connect_timeout_sec))
        self._read_timeout = max(1.0, float(read_timeout_sec))
        self._max_age_sec = max(0.1, float(max_age_sec))
        self._debug_samples_enabled = bool(debug_samples_enabled)
        self._debug_samples_limit = max(1, int(debug_samples_limit))
        self._capability_probe_enabled = bool(capability_probe_enabled)
        self._auto_configure_enabled = bool(auto_configure_enabled)
        self._auto_configure_dry_run = bool(auto_configure_dry_run)

        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(username, password)
        self._session.verify = verify_tls

        self._run = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: List[_EventDetection] = []
        self._last_log_ts = 0.0
        self._events_total = 0
        self._events_human = 0
        self._events_human_with_bbox = 0
        self._events_parse_errors = 0
        self._last_human_event_ts = 0.0
        self._debug_samples_logged = 0
        self._event_type_counts: dict[str, int] = {}
        self._stats_log_last_ts = 0.0
        self._no_human_warning_logged = False
        self._probe_done = False
        self._auto_configure_done = False
        self._face_bbox_paths = [
            "/ISAPI/Smart/FaceDetection/1",
            "/ISAPI/Intelligent/FDLib/FaceDataRecord",
            "/ISAPI/Smart/FaceCapture/channels/1",
            "/ISAPI/Smart/FaceCapture/channels/1/targets",
            "/ISAPI/Smart/FaceCapture/channels/1/status",
        ]
        self._face_path_probe_done = False

    def start(self) -> None:
        if self._run:
            return
        self._run = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="isapi-alertstream")
        self._thread.start()
        logger.info("ISAPI EventStream: воркер запущен")

    def stop(self) -> None:
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def detect(self, frame_shape: Tuple[int, int]) -> List[Detection]:
        now = time.time()
        h, w = frame_shape
        with self._lock:
            fresh = [e for e in self._latest if (now - e.ts) <= self._max_age_sec]

        out: List[Detection] = []
        for e in fresh:
            if e.bbox_norm is None:
                continue
            x1n, y1n, x2n, y2n = e.bbox_norm
            x1 = int(max(0, min(w - 1, round(x1n * w))))
            y1 = int(max(0, min(h - 1, round(y1n * h))))
            x2 = int(max(0, min(w - 1, round(x2n * w))))
            y2 = int(max(0, min(h - 1, round(y2n * h))))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(bbox=(x1, y1, x2, y2), confidence=e.confidence, source="camera_isapi_event"))
        return out

    def stats_snapshot(self) -> dict:
        with self._lock:
            return {
                "events_total": self._events_total,
                "events_human": self._events_human,
                "events_human_with_bbox": self._events_human_with_bbox,
                "events_parse_errors": self._events_parse_errors,
                "top_event_types": sorted(self._event_type_counts.items(), key=lambda kv: kv[1], reverse=True)[:5],
                "last_human_event_age_sec": (time.time() - self._last_human_event_ts) if self._last_human_event_ts > 0 else None,
            }

    def has_recent_human_signal(self, window_sec: float = 2.5) -> bool:
        window = max(0.1, float(window_sec))
        with self._lock:
            if self._last_human_event_ts <= 0:
                return False
            return (time.time() - self._last_human_event_ts) <= window

    def _worker(self) -> None:
        # В цикле реконнекта: если поток отвалился, поднимаемся снова.
        while self._run:
            if self._capability_probe_enabled and not self._probe_done:
                self._probe_capabilities()
            if self._auto_configure_enabled and not self._auto_configure_done:
                self._auto_configure_smart_events()
            try:
                self._consume_stream()
            except Exception as exc:
                now = time.time()
                if now - self._last_log_ts > 3.0:
                    logger.warning("ISAPI EventStream reconnect: %s", exc)
                    self._last_log_ts = now
                time.sleep(1.0)

    def _auto_configure_smart_events(self) -> None:
        targets = [
            "/ISAPI/Smart/FieldDetection/1",
            "/ISAPI/Smart/LineDetection/1",
            "/ISAPI/Smart/RegionEntrance/1",
            "/ISAPI/Smart/RegionExiting/1",
            "/ISAPI/Smart/FaceDetection/1",
            "/ISAPI/Smart/VMD/1",
        ]
        logger.info(
            "ISAPI auto-config: запуск (dry_run=%s), цель — включить smart events.",
            self._auto_configure_dry_run,
        )
        changed = 0
        checked = 0
        for path in targets:
            url = f"{self._base_url}{path}"
            try:
                resp = self._session.get(url, timeout=(self._connect_timeout, self._connect_timeout))
            except Exception as exc:
                logger.info("ISAPI auto-config: %s -> GET error: %s", path, exc)
                continue

            checked += 1
            if resp.status_code != 200:
                logger.info("ISAPI auto-config: %s -> GET HTTP %s (skip)", path, resp.status_code)
                continue

            current_xml = resp.text or ""
            enabled_raw = _first_group(
                r"<enabled(?:\s+[^>]*)?>\s*(true|false)\s*</enabled>",
                current_xml,
            )
            if enabled_raw is not None and enabled_raw.strip().lower() == "true":
                logger.info("ISAPI auto-config: %s уже enabled=true", path)
                continue

            updated_xml = re.sub(
                r"<enabled(\s+[^>]*)?>\s*false\s*</enabled>",
                r"<enabled\1>true</enabled>",
                current_xml,
                count=1,
                flags=re.IGNORECASE,
            )
            if updated_xml == current_xml:
                logger.info("ISAPI auto-config: %s не найден тег <enabled>false</enabled>", path)
                continue

            if self._auto_configure_dry_run:
                logger.info("ISAPI auto-config: %s dry-run -> было бы включено", path)
                changed += 1
                continue

            try:
                put = self._session.put(
                    url,
                    data=updated_xml,
                    headers={"Content-Type": "application/xml"},
                    timeout=(self._connect_timeout, self._connect_timeout),
                )
                if 200 <= put.status_code < 300:
                    logger.info("ISAPI auto-config: %s успешно включено (HTTP %s)", path, put.status_code)
                    changed += 1
                else:
                    logger.info("ISAPI auto-config: %s PUT HTTP %s", path, put.status_code)
            except Exception as exc:
                logger.info("ISAPI auto-config: %s PUT error: %s", path, exc)

        logger.info(
            "ISAPI auto-config: завершено, checked=%s changed=%s dry_run=%s",
            checked,
            changed,
            self._auto_configure_dry_run,
        )
        self._auto_configure_done = True

    def _probe_capabilities(self) -> None:
        probes = [
            "/ISAPI/Event/triggers",
            "/ISAPI/Event/triggers?format=json",
            "/ISAPI/Smart/Capabilities",
            "/ISAPI/Smart/FieldDetection/1/capabilities",
            "/ISAPI/Smart/LineDetection/1/capabilities",
            "/ISAPI/Smart/RegionEntrance/1/capabilities",
            "/ISAPI/Smart/RegionExiting/1/capabilities",
            "/ISAPI/Smart/HumanDetection/1/capabilities",
            "/ISAPI/System/capabilities",
        ]
        logger.info("ISAPI probe: проверка доступности Smart/Event endpoints...")
        for suffix in probes:
            url = f"{self._base_url}{suffix}"
            try:
                resp = self._session.get(url, timeout=(self._connect_timeout, self._connect_timeout))
                body = (resp.text or "").strip()
                preview = re.sub(r"\s+", " ", body)[:180]
                logger.info("ISAPI probe: %s -> HTTP %s (%s)", suffix, resp.status_code, preview)
                if suffix == "/ISAPI/Event/triggers" and resp.status_code == 200:
                    self._log_trigger_summary(body)
                if suffix.startswith("/ISAPI/Smart/") and resp.status_code == 200:
                    enabled = _first_group(r"<enabled(?:\s+[^>]*)?>\s*(true|false)\s*</enabled>", body)
                    if enabled is not None:
                        logger.info("ISAPI probe: %s enabled=%s", suffix, enabled.strip().lower())
            except Exception as exc:
                logger.info("ISAPI probe: %s -> error: %s", suffix, exc)
        self._probe_done = True

    def _log_trigger_summary(self, xml_text: str) -> None:
        event_types = [v.strip().lower() for v in re.findall(r"<eventType>\s*([^<]+)\s*</eventType>", xml_text, re.IGNORECASE)]
        event_states = [v.strip().lower() for v in re.findall(r"<eventState>\s*([^<]+)\s*</eventState>", xml_text, re.IGNORECASE)]
        event_types_unique = sorted(set(event_types))
        event_states_unique = sorted(set(event_states))

        logger.info(
            "ISAPI triggers summary: types=%s states=%s",
            event_types_unique[:20],
            event_states_unique[:20],
        )

        smart_keywords = (
            "fielddetection",
            "linedetection",
            "regionentrance",
            "regionexiting",
            "vca",
            "humandetection",
            "person",
            "targetcapture",
            "mixedtargetdetection",
        )
        has_smart = any(any(k in et for k in smart_keywords) for et in event_types_unique)
        if not has_smart:
            logger.warning(
                "ISAPI triggers: smart/human event types не найдены. "
                "Сейчас камера отдаёт системные события (например, videoloss), "
                "поэтому camera_events не смогут заменить локальную YOLO."
            )

    def _consume_stream(self) -> None:
        url = f"{self._base_url}/ISAPI/Event/notification/alertStream"
        with self._session.get(
            url,
            stream=True,
            timeout=(self._connect_timeout, self._read_timeout),
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"alertStream HTTP {resp.status_code}")
            logger.info("ISAPI EventStream: подключено")
            buf = ""
            for chunk in resp.iter_content(chunk_size=2048):
                if not self._run:
                    break
                if not chunk:
                    continue
                buf += chunk.decode(errors="ignore")
                if len(buf) > 65536:
                    buf = buf[-32768:]
                self._parse_buffer(buf)

    def _parse_buffer(self, buf: str) -> None:
        # Ищем XML-блоки EventNotificationAlert. Некоторые камеры шлют multipart,
        # поэтому разбираем best-effort по шаблону.
        for m in re.finditer(r"<EventNotificationAlert[\s\S]*?</EventNotificationAlert>", buf, re.IGNORECASE):
            block = m.group(0)
            with self._lock:
                self._events_total += 1
            event_type = (_first_group(r"<eventType>\s*([^<]+)\s*</eventType>", block) or "").strip().lower() or "unknown"
            with self._lock:
                self._event_type_counts[event_type] = self._event_type_counts.get(event_type, 0) + 1
            evt = self._extract_event(block)
            if evt is not None:
                with self._lock:
                    self._events_human += 1
                    self._last_human_event_ts = time.time()
                    if evt.bbox_norm is not None:
                        self._events_human_with_bbox += 1
                with self._lock:
                    self._latest = [evt]
            elif self._debug_samples_enabled:
                self._maybe_log_sample(block, matched=False)
            self._maybe_log_stats()

    def _extract_event(self, block: str) -> Optional[_EventDetection]:
        event_type = _first_group(r"<eventType>\s*([^<]+)\s*</eventType>", block)
        event_state = _first_group(r"<eventState>\s*([^<]+)\s*</eventState>", block)
        target_type = _first_group(r"<targetType>\s*([^<]+)\s*</targetType>", block)

        et = (event_type or "").strip().lower()
        es = (event_state or "").strip().lower()
        tt = (target_type or "").strip().lower()

        is_human_evt = any(k in et for k in _HUMAN_EVENT_KEYS) or ("human" in tt)
        if not is_human_evt:
            if self._debug_samples_enabled:
                self._maybe_log_sample(block, matched=False, event_type=et, target_type=tt, event_state=es)
            return None
        if es and es != "active":
            if self._debug_samples_enabled:
                self._maybe_log_sample(block, matched=False, event_type=et, target_type=tt, event_state=es)
            return None

        try:
            bbox = _extract_norm_bbox(block)
            if bbox is None and "face" in et:
                bbox = self._enrich_face_bbox()
            confidence = _extract_confidence(block)
            if self._debug_samples_enabled:
                self._maybe_log_sample(
                    block,
                    matched=True,
                    event_type=et,
                    target_type=tt,
                    event_state=es,
                    bbox=bbox,
                    confidence=confidence,
                )
            return _EventDetection(ts=time.time(), confidence=confidence, bbox_norm=bbox)
        except Exception:
            with self._lock:
                self._events_parse_errors += 1
            return None

    def _enrich_face_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        # Некоторые камеры в alertStream face-событиях не отдают bbox,
        # но держат актуальную рамку в отдельном ISAPI endpoint.
        statuses: list[str] = []
        for path in list(self._face_bbox_paths):
            url = f"{self._base_url}{path}"
            try:
                resp = self._session.get(url, timeout=(self._connect_timeout, self._connect_timeout))
            except Exception:
                statuses.append(f"{path}=error")
                continue
            if resp.status_code == 404:
                statuses.append(f"{path}=404")
                continue
            if resp.status_code != 200:
                statuses.append(f"{path}={resp.status_code}")
                continue
            statuses.append(f"{path}=200")
            text = resp.text or ""
            bbox = _extract_norm_bbox(text)
            if bbox is not None:
                logger.info("ISAPI face enrichment: bbox from %s", path)
                return bbox
            if self._debug_samples_enabled and not self._face_path_probe_done:
                preview = re.sub(r"\s+", " ", text).strip()
                if len(preview) > 260:
                    preview = preview[:260] + "..."
                logger.info("ISAPI face enrichment: %s payload without bbox (%s)", path, preview)
        if self._debug_samples_enabled and not self._face_path_probe_done and statuses:
            logger.info("ISAPI face enrichment probe statuses: %s", statuses)
        self._face_path_probe_done = True
        return None

    def _maybe_log_sample(
        self,
        block: str,
        *,
        matched: bool,
        event_type: str = "",
        target_type: str = "",
        event_state: str = "",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        with self._lock:
            if self._debug_samples_logged >= self._debug_samples_limit:
                return
            self._debug_samples_logged += 1

        if not event_type:
            event_type = (_first_group(r"<eventType>\s*([^<]+)\s*</eventType>", block) or "").strip().lower()
        if not target_type:
            target_type = (_first_group(r"<targetType>\s*([^<]+)\s*</targetType>", block) or "").strip().lower()
        if not event_state:
            event_state = (_first_group(r"<eventState>\s*([^<]+)\s*</eventState>", block) or "").strip().lower()

        short_xml = re.sub(r"\s+", " ", block).strip()
        max_len = 300
        # Для событий, которые считаем "релевантными", но без bbox,
        # оставляем больше сырого payload для точной подстройки парсера.
        if matched and bbox is None:
            max_len = 1400
        if len(short_xml) > max_len:
            short_xml = short_xml[:max_len] + "..."

        logger.info(
            "Event sample #%s matched=%s type=%s target=%s state=%s conf=%s bbox=%s xml=%s",
            self._debug_samples_logged,
            matched,
            event_type or "-",
            target_type or "-",
            event_state or "-",
            f"{confidence:.3f}" if confidence is not None else "-",
            bbox,
            short_xml,
        )

    def _maybe_log_stats(self) -> None:
        now = time.time()
        with self._lock:
            if now - self._stats_log_last_ts < 30.0:
                return
            self._stats_log_last_ts = now
            total = self._events_total
            human = self._events_human
            with_bbox = self._events_human_with_bbox
            top_types = sorted(self._event_type_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            no_human = total >= 20 and human == 0
            should_warn = no_human and not self._no_human_warning_logged
            if should_warn:
                self._no_human_warning_logged = True

        logger.info(
            "ISAPI EventStream stats: total=%s human=%s human_bbox=%s top_types=%s",
            total,
            human,
            with_bbox,
            top_types,
        )
        if should_warn:
            logger.warning(
                "ISAPI EventStream: получаем события, но нет human-детекций. "
                "Проверьте, что в камере включены Smart/Human события для канала."
            )


def _first_group(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_confidence(block: str) -> float:
    for pat in (
        r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>",
        r"<targetConfidence>\s*([0-9]*\.?[0-9]+)\s*</targetConfidence>",
    ):
        m = re.search(pat, block, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if v > 1.0:
                    v /= 100.0
                return max(0.0, min(1.0, v))
            except Exception:
                pass
    return 0.7


def _extract_norm_bbox(block: str) -> Optional[Tuple[float, float, float, float]]:
    # 1) x/y/width/height
    x = _as_float(_first_group(r"<x>\s*([0-9]*\.?[0-9]+)\s*</x>", block))
    y = _as_float(_first_group(r"<y>\s*([0-9]*\.?[0-9]+)\s*</y>", block))
    w = _as_float(_first_group(r"<width>\s*([0-9]*\.?[0-9]+)\s*</width>", block))
    h = _as_float(_first_group(r"<height>\s*([0-9]*\.?[0-9]+)\s*</height>", block))
    if None not in (x, y, w, h):
        xn, yn, wn, hn = _norm4(x, y, w, h)
        return _clip_bbox((xn, yn, xn + wn, yn + hn))

    # 2) left/top/right/bottom
    left = _as_float(_first_group(r"<left>\s*([0-9]*\.?[0-9]+)\s*</left>", block))
    top = _as_float(_first_group(r"<top>\s*([0-9]*\.?[0-9]+)\s*</top>", block))
    right = _as_float(_first_group(r"<right>\s*([0-9]*\.?[0-9]+)\s*</right>", block))
    bottom = _as_float(_first_group(r"<bottom>\s*([0-9]*\.?[0-9]+)\s*</bottom>", block))
    if None not in (left, top, right, bottom):
        l, t, r, b = _norm4(left, top, right, bottom, is_xyxy=True)
        return _clip_bbox((l, t, r, b))

    # 3) normalizedPosition [x,y] + targetMaxY (best effort)
    nx = _as_float(_first_group(r"<NormalizationPosition>\s*\[\s*([0-9]*\.?[0-9]+)", block))
    ny = _as_float(_first_group(r"<NormalizationPosition>\s*\[\s*[0-9]*\.?[0-9]+\s*,\s*([0-9]*\.?[0-9]+)", block))
    maxy = _as_float(_first_group(r"<targetMaxY>\s*([0-9]*\.?[0-9]+)\s*</targetMaxY>", block))
    if None not in (nx, ny, maxy):
        x = max(0.0, min(1.0, nx / 1000.0))
        y = max(0.0, min(1.0, ny / 1000.0))
        y2 = max(y, min(1.0, maxy / 1500.0))
        # Без ширины рисуем узкий бокс вокруг центра.
        half_w = 0.05
        return _clip_bbox((x - half_w, y, x + half_w, y2))

    # 4) face/frame style (часто встречается в faceDetection payload)
    fx = _as_float(_first_group(r"<faceX>\s*([0-9]*\.?[0-9]+)\s*</faceX>", block))
    fy = _as_float(_first_group(r"<faceY>\s*([0-9]*\.?[0-9]+)\s*</faceY>", block))
    fw = _as_float(_first_group(r"<faceWidth>\s*([0-9]*\.?[0-9]+)\s*</faceWidth>", block))
    fh = _as_float(_first_group(r"<faceHeight>\s*([0-9]*\.?[0-9]+)\s*</faceHeight>", block))
    if None not in (fx, fy, fw, fh):
        xn, yn, wn, hn = _norm4(fx, fy, fw, fh)
        return _clip_bbox((xn, yn, xn + wn, yn + hn))

    # 5) rect in one tag: [x,y,w,h]
    rx = _as_float(_first_group(r"<rect>\s*\[\s*([0-9]*\.?[0-9]+)", block))
    ry = _as_float(_first_group(r"<rect>\s*\[\s*[0-9]*\.?[0-9]+\s*,\s*([0-9]*\.?[0-9]+)", block))
    rw = _as_float(
        _first_group(
            r"<rect>\s*\[\s*[0-9]*\.?[0-9]+\s*,\s*[0-9]*\.?[0-9]+\s*,\s*([0-9]*\.?[0-9]+)",
            block,
        )
    )
    rh = _as_float(
        _first_group(
            r"<rect>\s*\[\s*[0-9]*\.?[0-9]+\s*,\s*[0-9]*\.?[0-9]+\s*,\s*[0-9]*\.?[0-9]+\s*,\s*([0-9]*\.?[0-9]+)",
            block,
        )
    )
    if None not in (rx, ry, rw, rh):
        xn, yn, wn, hn = _norm4(rx, ry, rw, rh)
        return _clip_bbox((xn, yn, xn + wn, yn + hn))

    # 6) "X/Y/Width/Height" with suffix/prefix
    gx = _as_float(_first_group(r"<(?:face|target)?X(?:Coord)?>\s*([0-9]*\.?[0-9]+)\s*</(?:face|target)?X(?:Coord)?>", block))
    gy = _as_float(_first_group(r"<(?:face|target)?Y(?:Coord)?>\s*([0-9]*\.?[0-9]+)\s*</(?:face|target)?Y(?:Coord)?>", block))
    gw = _as_float(_first_group(r"<(?:face|target)?Width>\s*([0-9]*\.?[0-9]+)\s*</(?:face|target)?Width>", block))
    gh = _as_float(_first_group(r"<(?:face|target)?Height>\s*([0-9]*\.?[0-9]+)\s*</(?:face|target)?Height>", block))
    if None not in (gx, gy, gw, gh):
        xn, yn, wn, hn = _norm4(gx, gy, gw, gh)
        return _clip_bbox((xn, yn, xn + wn, yn + hn))

    return None


def _as_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _norm4(a: float, b: float, c: float, d: float, *, is_xyxy: bool = False) -> Tuple[float, float, float, float]:
    # У камер встречаются либо [0..1], либо [0..1000].
    scale = 1000.0 if max(a, b, c, d) > 1.5 else 1.0
    if is_xyxy:
        return a / scale, b / scale, c / scale, d / scale
    return a / scale, b / scale, c / scale, d / scale


def _clip_bbox(b: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
