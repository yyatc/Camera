from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2

from src.camera.isapi_client import IsapiPtzClient
from src.camera.isapi_event_stream import IsapiEventStreamClient
from src.camera.onvif_client import OnvifPtzClient
from src.camera.rtsp_reader import RtspReader
from src.common.logging_setup import configure_logging, sanitize_rtsp_url
from src.common.ml_device import resolve_inference_device
from src.common.types import PTZCommand
from src.common.settings import load_settings
from src.domain.ptz_control_policy import PtzPolicyConfig, PtzControlPolicy
from src.domain.target_selector import TargetSelector
from src.domain.tracking_state import TrackingMode
from src.stats.observation_registry import ObservationRegistry
from src.stream.overlay_renderer import OverlayRenderer
from src.stream.rtsp_publisher import RtspPublisher
from src.vision.hybrid_detector import CameraAnalyticsAdapter, HybridDetector
from src.vision.local_person_detector import LocalPersonDetector
from src.vision.person_tracker import PersonTracker

logger = logging.getLogger(__name__)



class AsyncDetector:
    """
    Wrapper over HybridDetector — runs inference in a separate thread.
    Main loop calls submit() and continues immediately without waiting.
    Always uses the last ready result via .latest.
    Eliminates main loop freezes caused by YOLO inference spikes.
    """

    def __init__(self, detector) -> None:
        self._detector = detector
        self._input: queue.Queue = queue.Queue(maxsize=1)
        self._latest = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="detector")
        self._thread.start()

    def _loop(self) -> None:
        while True:
            frame = self._input.get()
            if frame is None:
                break
            try:
                result = self._detector.detect(frame)
                with self._lock:
                    self._latest = result
            except Exception as exc:
                logger.error("Detector error: %s", exc)

    def submit(self, frame) -> None:
        try:
            self._input.put_nowait(frame)
        except queue.Full:
            pass  # detector is busy — use cached result

    @property
    def latest(self):
        with self._lock:
            return list(self._latest)

    def stats_snapshot(self):
        """Delegate to wrapped detector — preserves compatibility with heartbeat logging."""
        return self._detector.stats_snapshot()

    def __getattr__(self, name):
        """Fallback delegation: any unknown attribute is forwarded to the wrapped detector."""
        return getattr(self._detector, name)

    def stop(self) -> None:
        self._input.put(None)
        self._thread.join(timeout=5.0)


class PtzClient(Protocol):
    def safe_move(self, cmd: Optional[PTZCommand]) -> None:
        ...

    def stop(self) -> None:
        ...

    # Опционально для ISAPI-реализации
    def goto_preset(self, preset_id: int) -> None:
        ...

    def move_absolute(
        self,
        *,
        pan_deg: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        zoom_ratio: Optional[float] = None,
    ) -> None:
        ...

    def get_status(self) -> object:
        ...


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    settings = load_settings(root)

    app_cfg = settings.raw.get("app", {})
    log_level = os.getenv("LOG_LEVEL") or app_cfg.get("log_level", "INFO")
    configure_logging(str(log_level))

    heartbeat_sec = float(app_cfg.get("heartbeat_interval_sec", 5.0))

    width = settings.raw["app"]["frame_width"]
    height = settings.raw["app"]["frame_height"]
    fps = settings.raw["app"]["process_fps"]
    loop_pacing = bool(app_cfg.get("loop_pacing_enabled", False))
    step = 1.0 / max(1, fps) if loop_pacing else 0.0
    detect_every_n_frames = max(1, int(settings.raw["tracking"].get("detect_every_n_frames", 1)))
    detect_every_tracking = max(
        1,
        int(settings.raw["tracking"].get("detect_every_n_frames_tracking", detect_every_n_frames)),
    )
    detect_every_searching = max(
        1,
        int(settings.raw["tracking"].get("detect_every_n_frames_searching", detect_every_n_frames)),
    )
    presence_signal_enabled = bool(settings.raw["tracking"].get("camera_presence_signal_enabled", True))
    presence_signal_window_sec = float(settings.raw["tracking"].get("camera_presence_signal_window_sec", 2.5))
    detect_every_searching_boost = max(
        1,
        int(settings.raw["tracking"].get("detect_every_n_frames_searching_boost", 1)),
    )
    cv_threads = int(app_cfg.get("cv_threads", 0))
    cv2.setUseOptimized(True)
    if cv_threads > 0:
        cv2.setNumThreads(cv_threads)

    in_url = sanitize_rtsp_url(settings.input_rtsp)
    out_url = sanitize_rtsp_url(settings.output_rtsp)
    logger.info(
        "Старт трекера: разрешение %dx%d, целевой FPS=%s, детекция каждые %s кадр(ов)",
        width,
        height,
        fps,
        detect_every_n_frames,
    )
    logger.info("Входной RTSP: %s", in_url)
    logger.info("Выходной RTSP (публикация): %s", out_url)

    stream_cfg = settings.raw.get("stream", {})
    reader = RtspReader(
        settings.input_rtsp,
        width=width,
        height=height,
        open_timeout_sec=float(stream_cfg.get("open_timeout_sec", 8.0)),
        extra_ffmpeg_capture_options=stream_cfg.get("ffmpeg_capture_options_extra"),
    )
    ptz = _init_ptz_client(
        host=settings.camera_host,
        username=settings.camera_username,
        password=settings.camera_password,
        ptz_cfg=settings.raw.get("ptz", {}),
        preferred_port=settings.raw.get("ptz", {}).get("onvif_port"),
        connect_timeout_sec=float(settings.raw.get("ptz", {}).get("onvif_connect_timeout_sec", 1.5)),
    )
    if ptz is None:
        logger.warning("PTZ недоступен: трекинг только по видео, без поворота камеры.")
    else:
        logger.info("PTZ-клиент инициализирован.")

    tracking_cfg = settings.raw.get("tracking", {})
    event_client = _init_event_client(
        host=settings.camera_host,
        username=settings.camera_username,
        password=settings.camera_password,
        tracking_cfg=tracking_cfg,
        ptz_cfg=settings.raw.get("ptz", {}),
    )

    tr = settings.raw["tracking"]
    env_ml = os.getenv("ML_DEVICE")
    if env_ml is not None and str(env_ml).strip():
        ml_spec = str(env_ml).strip()
    else:
        ml_spec = str(tr.get("inference_device", "auto")).strip() or "auto"
    inference_device = resolve_inference_device(ml_spec)
    detector = HybridDetector(
        local_detector=LocalPersonDetector(
            tr["detection_confidence"],
            input_size=int(tr.get("detector_input_size", 640)),
            model_name=str(tr.get("detector_model", "yolo11n.pt")),
            min_bbox_area_ratio=float(tr.get("min_bbox_area_ratio", 0.004)),
            max_bbox_area_ratio=float(tr.get("max_bbox_area_ratio", 0.75)),
            min_aspect_h_w=float(tr.get("min_aspect_h_w", 0.85)),
            max_aspect_h_w=float(tr.get("max_aspect_h_w", 4.2)),
            yolo_iou=float(tr.get("yolo_iou", 0.5)),
            max_detections=int(tr.get("detector_max_detections", 20)),
            mediapipe_enabled=bool(tr.get("mediapipe_enabled", True)),
            mediapipe_min_visibility=float(tr.get("mediapipe_min_visibility", 0.45)),
            inference_device=inference_device,
        ),
        camera_adapter=CameraAnalyticsAdapter(event_source=event_client),
        min_confidence=tr["detection_confidence"],
    )
    detector = AsyncDetector(detector)
    logger.info(
        "Детектор: модель=%s, conf>=%s, input_size=%s, max_det=%s, stride(track/search/search_boost)=%s/%s/%s, "
        "camera_events=%s, ml_device=%s (запрос=%s)",
        tr.get("detector_model", "yolo11n.pt"),
        tr["detection_confidence"],
        tr.get("detector_input_size", 640),
        tr.get("detector_max_detections", 20),
        detect_every_tracking,
        detect_every_searching,
        detect_every_searching_boost,
        bool(event_client is not None),
        inference_device,
        ml_spec,
    )

    tr_timeout = float(settings.raw["tracking"]["tracking_timeout_sec"])
    tr_match_px = float(settings.raw["tracking"].get("max_track_match_distance_px", 120.0))
    tracker = PersonTracker(
        max_distance_px=tr_match_px,
        timeout_sec=tr_timeout,
    )
    selector = TargetSelector()
    registry = ObservationRegistry()
    overlay = OverlayRenderer()
    publisher = RtspPublisher(
        ffmpeg_bin=stream_cfg["ffmpeg_bin"],
        output_rtsp_url=settings.output_rtsp,
        width=width,
        height=height,
        fps=fps,
        queue_size=int(stream_cfg.get("publisher_queue_size", 1)),
    )
    ptz_cfg = settings.raw["ptz"]
    policy = PtzControlPolicy(
        PtzPolicyConfig(
            pan_gain=ptz_cfg["pan_gain"],
            tilt_gain=ptz_cfg["tilt_gain"],
            zoom_gain=ptz_cfg["zoom_gain"],
            max_pan_speed=ptz_cfg["max_pan_speed"],
            max_tilt_speed=ptz_cfg["max_tilt_speed"],
            max_zoom_speed=ptz_cfg["max_zoom_speed"],
            center_tolerance_x=settings.raw["tracking"]["center_tolerance_x"],
            center_tolerance_y=settings.raw["tracking"]["center_tolerance_y"],
            target_area_ratio=settings.raw["tracking"]["target_area_ratio"],
            zoom_hysteresis=settings.raw["tracking"]["zoom_hysteresis"],
            search_pan_speed=ptz_cfg["search_pan_speed"],
            search_tilt_speed=ptz_cfg["search_tilt_speed"],
            search_zoom_out_speed=ptz_cfg["search_zoom_out_speed"],
            min_effective_pan_speed=float(ptz_cfg.get("min_effective_pan_speed", 0.0)),
            min_effective_tilt_speed=float(ptz_cfg.get("min_effective_tilt_speed", 0.0)),
        )
    )

    mode = TrackingMode.SEARCHING
    current_target_id = None
    prev_mode: Optional[TrackingMode] = None
    prev_target_id = None
    monitor_ptz_interval = float(settings.raw["ptz"].get("monitoring_ptz_interval_sec", 0.20))
    ptz_status_log_interval = float(settings.raw["ptz"].get("status_log_interval_sec", 3.0))
    last_monitor_ptz_ts = 0.0
    last_ptz_status_log_ts = 0.0
    last_ptz_error_log_ts = 0.0
    reconnect_threshold = int(stream_cfg.get("reconnect_threshold_frames", 30))
    read_fail_count = 0
    frame_index = 0
    cached_detections = []
    stream_ready_logged = False
    last_heartbeat_ts = time.time()
    frames_in_heartbeat = 0
    reconnect_events = 0

    # --- Profiler state ---
    _prof_read_ms: list = []
    _prof_track_ms: list = []
    _prof_render_ms: list = []
    _prof_publish_ms: list = []
    _PROF_SPIKE_MS = 50.0   # warn if any stage exceeds this threshold

    reader.open()
    publisher.open()
    logger.info("Главный цикл запущен (Ctrl+C для остановки).")
    try:
        while True:
            loop_started = time.time()
            ts = time.time()
            _t0 = time.perf_counter()
            ok, frame = reader.read()
            _prof_read_ms.append((time.perf_counter() - _t0) * 1000)
            if not ok:
                read_fail_count += 1
                if read_fail_count == 1 or read_fail_count % reconnect_threshold == 0:
                    logger.warning(
                        "Нет кадра с входа (подряд: %s), ожидание...",
                        read_fail_count,
                    )
                # При длительном отсутствии кадров переподключаемся к RTSP камеры,
                # иначе MediaMTX закрывает исходящий publisher как неактивный.
                if read_fail_count >= reconnect_threshold:
                    logger.warning(
                        "Переподключение к входному RTSP после %s неудачных чтений...",
                        read_fail_count,
                    )
                    try:
                        reader.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    try:
                        reader.open()
                        read_fail_count = 0
                        reconnect_events += 1
                        stream_ready_logged = False
                        logger.info(
                            "Входной поток снова открыт (переподключений за сессию: %s)",
                            reconnect_events,
                        )
                    except Exception as exc:
                        logger.error("Не удалось переподключиться к входу: %s", exc)
                        time.sleep(1.0)
                continue
            read_fail_count = 0
            frame_index += 1
            frames_in_heartbeat += 1

            if not stream_ready_logged:
                logger.info("Приём кадров с камеры стабилен.")
                stream_ready_logged = True

            _t1 = time.perf_counter()
            camera_presence_signal = bool(
                presence_signal_enabled
                and event_client is not None
                and event_client.has_recent_human_signal(presence_signal_window_sec)
            )
            detect_stride = detect_every_tracking if mode == TrackingMode.TRACKING else detect_every_searching
            if mode == TrackingMode.SEARCHING and camera_presence_signal:
                detect_stride = detect_every_searching_boost
            if frame_index % detect_stride == 0:
                detector.submit(frame)
            detections = detector.latest or cached_detections
            cached_detections = detections
            tracks = tracker.update(detections, ts=ts)
            target = selector.choose_target(tracks, preferred_id=current_target_id)
            if target is not None:
                mode = TrackingMode.TRACKING
                current_target_id = target.track_id
                cmd = policy.tracking_command(target, (frame.shape[1], frame.shape[0]))
                if ptz is not None:
                    last_ptz_error_log_ts = _safe_ptz_move(ptz, cmd, ts, last_ptz_error_log_ts)
            else:
                mode = TrackingMode.SEARCHING
                current_target_id = None
                if prev_mode != TrackingMode.SEARCHING:
                    _on_enter_search_mode(ptz, ptz_cfg)
                if ptz is not None and (ts - last_monitor_ptz_ts) >= monitor_ptz_interval:
                    last_ptz_error_log_ts = _safe_ptz_move(
                        ptz, policy.monitoring_command(ts), ts, last_ptz_error_log_ts
                    )
                    last_monitor_ptz_ts = ts

            if ptz is not None and (ts - last_ptz_status_log_ts) >= ptz_status_log_interval:
                _log_ptz_status(ptz)
                last_ptz_status_log_ts = ts

            if mode != prev_mode or current_target_id != prev_target_id:
                logger.info(
                    "Режим: %s, цель id=%s, треков в кадре=%s, уникальных=%s",
                    mode.name,
                    current_target_id,
                    len(tracks),
                    tracker.total_seen_unique,
                )
                prev_mode = mode
                prev_target_id = current_target_id

            _prof_track_ms.append((time.perf_counter() - _t1) * 1000)
            registry.tick(current_target_id, ts=ts)

            _t2 = time.perf_counter()
            rendered = overlay.render(
                frame=frame,
                target=target,
                total_count=tracker.total_seen_unique,
                per_person_seconds=registry.per_person_seconds,
                total_seconds=registry.total_seconds,
                first_seen_ts=tracker.first_seen_ts,
            )
            _prof_render_ms.append((time.perf_counter() - _t2) * 1000)
            _t3 = time.perf_counter()
            publisher.write(rendered)
            _prof_publish_ms.append((time.perf_counter() - _t3) * 1000)

            # Spike alert — log immediately if any stage is unusually slow
            _last_read = _prof_read_ms[-1] if _prof_read_ms else 0
            _last_track = _prof_track_ms[-1] if _prof_track_ms else 0
            _last_render = _prof_render_ms[-1] if _prof_render_ms else 0
            _last_pub = _prof_publish_ms[-1] if _prof_publish_ms else 0
            if max(_last_read, _last_track, _last_render, _last_pub) > _PROF_SPIKE_MS:
                logger.warning(
                    "SPIKE frame=%s: read=%.1fms track=%.1fms render=%.1fms publish=%.1fms",
                    frame_index, _last_read, _last_track, _last_render, _last_pub,
                )

            if heartbeat_sec > 0 and (ts - last_heartbeat_ts) >= heartbeat_sec:
                elapsed = ts - last_heartbeat_ts
                eff_fps = frames_in_heartbeat / elapsed if elapsed > 0 else 0.0
                def _pstat(lst):
                    if not lst: return "n/a"
                    avg = sum(lst)/len(lst)
                    mx = max(lst)
                    p95 = sorted(lst)[int(len(lst)*0.95)] if len(lst) >= 20 else mx
                    lst.clear()
                    return f"avg={avg:.1f} p95={p95:.1f} max={mx:.1f}ms"
                logger.info(
                    "PROFILER read=%s | track+ptz=%s | render=%s | publish=%s",
                    _pstat(_prof_read_ms), _pstat(_prof_track_ms),
                    _pstat(_prof_render_ms), _pstat(_prof_publish_ms),
                )
                det_stats = detector.stats_snapshot()
                evt_stats = event_client.stats_snapshot() if event_client is not None else None
                logger.info(
                    "Пульс: режим=%s, цель=%s, кадров=%s, ~%.1f FPS, переподключений=%s, "
                    "наблюдение всего %.1f с, уник. объектов=%s, src(camera/local)=%s/%s "
                    "camera_share=%.2f, camera_signal=%s",
                    mode.name,
                    current_target_id,
                    frame_index,
                    eff_fps,
                    reconnect_events,
                    registry.total_seconds,
                    tracker.total_seen_unique,
                    det_stats.get("camera_hits", 0),
                    det_stats.get("local_hits", 0),
                    float(det_stats.get("camera_share", 0.0)),
                    camera_presence_signal,
                )
                if evt_stats is not None:
                    logger.debug(
                        "EventStream stats: total=%s human=%s with_bbox=%s parse_errors=%s",
                        evt_stats.get("events_total", 0),
                        evt_stats.get("events_human", 0),
                        evt_stats.get("events_human_with_bbox", 0),
                        evt_stats.get("events_parse_errors", 0),
                    )
                last_heartbeat_ts = ts
                frames_in_heartbeat = 0

            elapsed = time.time() - loop_started
            if step > 0:
                remaining = step - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        detector.stop()
        logger.info("Остановка: закрытие PTZ и потоков...")
        if event_client is not None:
            try:
                event_client.stop()
            except Exception:
                pass
        if ptz is not None:
            try:
                ptz.stop()
                logger.info("PTZ остановлен.")
            except Exception as exc:
                logger.warning("Ошибка при остановке PTZ: %s", exc)
        publisher.close()
        reader.close()
        logger.info("Сервис остановлен.")


def _safe_ptz_move(
    ptz: PtzClient,
    cmd: PTZCommand,
    ts: float,
    last_error_log_ts: float,
) -> float:
    # safe_move() неблокирующий — ставит команду в очередь PTZ-воркера.
    # ONVIF Fault-ы обрабатываются внутри воркера; здесь исключений не ожидается.
    try:
        ptz.safe_move(cmd)
    except Exception as exc:
        if ts - last_error_log_ts >= 2.0:
            logger.warning("Команда PTZ не выполнена: %s", exc)
            last_error_log_ts = ts
    return last_error_log_ts


def _on_enter_search_mode(ptz: Optional[PtzClient], ptz_cfg: dict) -> None:
    if ptz is None:
        return
    # 1) Возврат в "home" preset, если есть.
    preset_id = ptz_cfg.get("home_preset_id")
    if preset_id is not None and hasattr(ptz, "goto_preset"):
        try:
            ptz.goto_preset(int(preset_id))
            logger.info("PTZ: переход в home preset id=%s", preset_id)
        except Exception as exc:
            logger.debug("PTZ: home preset недоступен: %s", exc)

    # 2) Дополнительно сбрасываем zoom (для камер с absolute control).
    if bool(ptz_cfg.get("search_reset_zoom", True)) and hasattr(ptz, "move_absolute"):
        try:
            ptz.move_absolute(zoom_ratio=float(ptz_cfg.get("search_home_zoom_ratio", 0.0)))
            logger.info("PTZ: сброс zoom для режима SEARCHING")
        except Exception as exc:
            logger.debug("PTZ: absolute zoom reset недоступен: %s", exc)


def _log_ptz_status(ptz: PtzClient) -> None:
    if not hasattr(ptz, "get_status"):
        return
    try:
        status = ptz.get_status()
    except Exception:
        return
    if status is None:
        return
    pan = getattr(status, "pan_deg", None)
    tilt = getattr(status, "tilt_deg", None)
    zoom = getattr(status, "zoom_ratio", None)
    logger.debug("PTZ status: pan=%s tilt=%s zoom=%s", pan, tilt, zoom)


def _init_ptz_client(
    host: str,
    username: str,
    password: str,
    ptz_cfg: dict,
    preferred_port: Optional[int],
    connect_timeout_sec: float = 1.5,
) -> Optional[PtzClient]:
    # 1) ISAPI first: быстрее и обычно стабильнее на нативных камерах.
    if bool(ptz_cfg.get("isapi_enabled", True)):
        isapi_timeout = float(ptz_cfg.get("isapi_timeout_sec", 2.0))
        channel_candidates = ptz_cfg.get("isapi_channel_candidates", [1, 101])
        for channel in channel_candidates:
            try:
                channel_id = int(channel)
            except Exception:
                continue
            try:
                logger.info("Попытка ISAPI PTZ: channel=%s ...", channel_id)
                isapi = IsapiPtzClient(
                    host=host,
                    username=username,
                    password=password,
                    channel_id=channel_id,
                    timeout_sec=isapi_timeout,
                    use_https=bool(ptz_cfg.get("isapi_use_https", False)),
                    verify_tls=bool(ptz_cfg.get("isapi_verify_tls", False)),
                )
                cap = isapi.capabilities
                logger.info(
                    "ISAPI PTZ подключен: channel=%s continuous=%s absolute=%s presets=%s status=%s",
                    channel_id,
                    cap.supports_continuous,
                    cap.supports_absolute,
                    cap.supports_presets,
                    cap.supports_status,
                )
                return isapi
            except Exception as exc:
                logger.debug("ISAPI channel=%s недоступен: %s", channel_id, exc)

    # 2) Fallback to ONVIF
    ports = []
    if isinstance(preferred_port, int):
        ports.append(preferred_port)
    for p in (80, 8899, 8000):
        if p not in ports:
            ports.append(p)

    last_error = None
    for port in ports:
        try:
            logger.info("Попытка ONVIF на %s:%s...", host, port)
            result: dict[str, object] = {}

            def _build_client() -> None:
                try:
                    result["client"] = OnvifPtzClient(host, username, password, port=port)
                except Exception as exc:
                    result["error"] = exc

            t = threading.Thread(target=_build_client, daemon=True, name=f"onvif-init-{port}")
            t.start()
            t.join(timeout=max(0.1, connect_timeout_sec))

            if t.is_alive():
                last_error = RuntimeError(
                    f"ONVIF connect timeout after {connect_timeout_sec:.1f}s on port {port}"
                )
                logger.warning("ONVIF порт %s: таймаут подключения (%.1f c)", port, connect_timeout_sec)
                continue

            if "client" in result:
                return result["client"]  # type: ignore[return-value]

            if "error" in result:
                raise result["error"]  # type: ignore[misc]
        except Exception as exc:
            last_error = exc
            logger.debug("ONVIF порт %s: %s", port, exc)

    logger.warning("ONVIF недоступен, работа без PTZ: %s", last_error)
    return None


def _init_event_client(
    host: str,
    username: str,
    password: str,
    tracking_cfg: dict,
    ptz_cfg: dict,
) -> Optional[IsapiEventStreamClient]:
    if not bool(tracking_cfg.get("camera_events_enabled", True)):
        logger.info("Camera events disabled by config")
        return None

    try:
        client = IsapiEventStreamClient(
            host=host,
            username=username,
            password=password,
            use_https=bool(ptz_cfg.get("isapi_use_https", False)),
            verify_tls=bool(ptz_cfg.get("isapi_verify_tls", False)),
            connect_timeout_sec=float(tracking_cfg.get("camera_events_connect_timeout_sec", 2.0)),
            read_timeout_sec=float(tracking_cfg.get("camera_events_read_timeout_sec", 15.0)),
            max_age_sec=float(tracking_cfg.get("camera_events_max_age_sec", 1.2)),
            debug_samples_enabled=bool(tracking_cfg.get("camera_events_debug_samples_enabled", False)),
            debug_samples_limit=int(tracking_cfg.get("camera_events_debug_samples_limit", 10)),
            capability_probe_enabled=bool(tracking_cfg.get("camera_events_capability_probe_enabled", True)),
            auto_configure_enabled=bool(tracking_cfg.get("camera_events_auto_configure_enabled", False)),
            auto_configure_dry_run=bool(tracking_cfg.get("camera_events_auto_configure_dry_run", False)),
        )
        client.start()
        return client
    except Exception as exc:
        logger.warning("Camera event stream unavailable, fallback to local detector: %s", exc)
        return None


if __name__ == "__main__":
    run()
