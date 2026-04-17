from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2

from src.camera.onvif_client import OnvifPtzClient
from src.camera.rtsp_reader import RtspReader
from src.common.logging_setup import configure_logging, sanitize_rtsp_url
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
    step = 1.0 / max(1, fps)
    detect_every_n_frames = max(1, int(settings.raw["tracking"].get("detect_every_n_frames", 1)))
    detect_every_tracking = max(
        1,
        int(settings.raw["tracking"].get("detect_every_n_frames_tracking", detect_every_n_frames)),
    )
    detect_every_searching = max(
        1,
        int(settings.raw["tracking"].get("detect_every_n_frames_searching", detect_every_n_frames)),
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

    reader = RtspReader(settings.input_rtsp, width=width, height=height)
    ptz = _init_ptz_client(
        host=settings.camera_host,
        username=settings.camera_username,
        password=settings.camera_password,
        preferred_port=settings.raw.get("ptz", {}).get("onvif_port"),
    )
    if ptz is None:
        logger.warning("PTZ недоступен: трекинг только по видео, без поворота камеры.")
    else:
        logger.info("PTZ-клиент ONVIF инициализирован.")

    tr = settings.raw["tracking"]
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
        ),
        camera_adapter=CameraAnalyticsAdapter(),
        min_confidence=tr["detection_confidence"],
    )
    logger.info(
        "Детектор: модель=%s, conf>=%s, input_size=%s, max_det=%s, stride(track/search)=%s/%s",
        tr.get("detector_model", "yolo11n.pt"),
        tr["detection_confidence"],
        tr.get("detector_input_size", 640),
        tr.get("detector_max_detections", 20),
        detect_every_tracking,
        detect_every_searching,
    )

    tracker = PersonTracker(timeout_sec=settings.raw["tracking"]["tracking_timeout_sec"])
    selector = TargetSelector()
    registry = ObservationRegistry()
    overlay = OverlayRenderer()
    publisher = RtspPublisher(
        ffmpeg_bin=settings.raw["stream"]["ffmpeg_bin"],
        output_rtsp_url=settings.output_rtsp,
        width=width,
        height=height,
        fps=fps,
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
    last_monitor_ptz_ts = 0.0
    last_ptz_error_log_ts = 0.0
    read_fail_count = 0
    frame_index = 0
    cached_detections = []
    stream_ready_logged = False
    last_heartbeat_ts = time.time()
    frames_in_heartbeat = 0
    reconnect_events = 0

    reader.open()
    publisher.open()
    logger.info("Главный цикл запущен (Ctrl+C для остановки).")
    try:
        while True:
            loop_started = time.time()
            ts = time.time()
            ok, frame = reader.read()
            if not ok:
                read_fail_count += 1
                if read_fail_count == 1 or read_fail_count % 30 == 0:
                    logger.warning(
                        "Нет кадра с входа (подряд: %s), ожидание...",
                        read_fail_count,
                    )
                # При длительном отсутствии кадров переподключаемся к RTSP камеры,
                # иначе MediaMTX закрывает исходящий publisher как неактивный.
                if read_fail_count >= 30:
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

            detect_stride = detect_every_tracking if mode == TrackingMode.TRACKING else detect_every_searching
            if frame_index % detect_stride == 0 or not cached_detections:
                cached_detections = detector.detect(frame)
            detections = cached_detections
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
                if ptz is not None and (ts - last_monitor_ptz_ts) >= monitor_ptz_interval:
                    last_ptz_error_log_ts = _safe_ptz_move(
                        ptz, policy.monitoring_command(ts), ts, last_ptz_error_log_ts
                    )
                    last_monitor_ptz_ts = ts

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

            registry.tick(current_target_id, ts=ts)

            rendered = overlay.render(
                frame=frame,
                target=target,
                total_count=tracker.total_seen_unique,
                per_person_seconds=registry.per_person_seconds,
                total_seconds=registry.total_seconds,
                first_seen_ts=tracker.first_seen_ts,
            )
            publisher.write(rendered)

            if heartbeat_sec > 0 and (ts - last_heartbeat_ts) >= heartbeat_sec:
                elapsed = ts - last_heartbeat_ts
                eff_fps = frames_in_heartbeat / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Пульс: режим=%s, цель=%s, кадров=%s, ~%.1f FPS, переподключений=%s, "
                    "наблюдение всего %.1f с, уник. объектов=%s",
                    mode.name,
                    current_target_id,
                    frame_index,
                    eff_fps,
                    reconnect_events,
                    registry.total_seconds,
                    tracker.total_seen_unique,
                )
                last_heartbeat_ts = ts
                frames_in_heartbeat = 0

            elapsed = time.time() - loop_started
            remaining = step - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        logger.info("Остановка: закрытие PTZ и потоков...")
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
    ptz: OnvifPtzClient,
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


def _init_ptz_client(
    host: str,
    username: str,
    password: str,
    preferred_port: Optional[int],
) -> Optional[OnvifPtzClient]:
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
            return OnvifPtzClient(host, username, password, port=port)
        except Exception as exc:
            last_error = exc
            logger.debug("ONVIF порт %s: %s", port, exc)

    logger.warning("ONVIF недоступен, работа без PTZ: %s", last_error)
    return None


if __name__ == "__main__":
    run()
