from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from src.camera.onvif_client import OnvifPtzClient
from src.camera.rtsp_reader import RtspReader
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


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    settings = load_settings(root)

    width = settings.raw["app"]["frame_width"]
    height = settings.raw["app"]["frame_height"]
    fps = settings.raw["app"]["process_fps"]
    step = 1.0 / max(1, fps)
    detect_every_n_frames = max(1, int(settings.raw["tracking"].get("detect_every_n_frames", 1)))

    reader = RtspReader(settings.input_rtsp, width=width, height=height)
    ptz = _init_ptz_client(
        host=settings.camera_host,
        username=settings.camera_username,
        password=settings.camera_password,
        preferred_port=settings.raw.get("ptz", {}).get("onvif_port"),
    )
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
        ),
        camera_adapter=CameraAnalyticsAdapter(),
        min_confidence=tr["detection_confidence"],
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
    policy = PtzControlPolicy(
        PtzPolicyConfig(
            pan_gain=settings.raw["ptz"]["pan_gain"],
            tilt_gain=settings.raw["ptz"]["tilt_gain"],
            zoom_gain=settings.raw["ptz"]["zoom_gain"],
            max_pan_speed=settings.raw["ptz"]["max_pan_speed"],
            max_tilt_speed=settings.raw["ptz"]["max_tilt_speed"],
            max_zoom_speed=settings.raw["ptz"]["max_zoom_speed"],
            center_tolerance_x=settings.raw["tracking"]["center_tolerance_x"],
            center_tolerance_y=settings.raw["tracking"]["center_tolerance_y"],
            target_area_ratio=settings.raw["tracking"]["target_area_ratio"],
            zoom_hysteresis=settings.raw["tracking"]["zoom_hysteresis"],
            search_pan_speed=settings.raw["ptz"]["search_pan_speed"],
            search_tilt_speed=settings.raw["ptz"]["search_tilt_speed"],
            search_zoom_out_speed=settings.raw["ptz"]["search_zoom_out_speed"],
        )
    )

    mode = TrackingMode.SEARCHING
    current_target_id = None
    monitor_ptz_interval = float(settings.raw["ptz"].get("monitoring_ptz_interval_sec", 0.20))
    last_monitor_ptz_ts = 0.0
    last_ptz_error_log_ts = 0.0
    read_fail_count = 0
    frame_index = 0
    cached_detections = []

    reader.open()
    publisher.open()
    try:
        while True:
            ts = time.time()
            ok, frame = reader.read()
            if not ok:
                read_fail_count += 1
                # При длительном отсутствии кадров переподключаемся к RTSP камеры,
                # иначе MediaMTX закрывает исходящий publisher как неактивный.
                if read_fail_count >= 30:
                    try:
                        reader.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    try:
                        reader.open()
                        read_fail_count = 0
                    except Exception:
                        time.sleep(1.0)
                continue
            read_fail_count = 0
            frame_index += 1

            if frame_index % detect_every_n_frames == 0 or not cached_detections:
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
            time.sleep(step)
    finally:
        if ptz is not None:
            try:
                ptz.stop()
            except Exception:
                pass
        publisher.close()
        reader.close()


def _safe_ptz_move(
    ptz: OnvifPtzClient,
    cmd: PTZCommand,
    ts: float,
    last_error_log_ts: float,
) -> float:
    try:
        ptz.safe_move(cmd)
        return last_error_log_ts
    except Exception as exc:
        # Не допускаем падения сервиса из-за ONVIF Fault (например "out of bounds").
        if ts - last_error_log_ts >= 2.0:
            print(f"[PTZ] move failed: {exc}")
            last_error_log_ts = ts
        try:
            ptz.stop()
        except Exception:
            pass
        time.sleep(0.2)
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
            print(f"[PTZ] Trying ONVIF on port {port}...")
            return OnvifPtzClient(host, username, password, port=port)
        except Exception as exc:
            last_error = exc

    print(f"[PTZ] ONVIF unavailable, continue without PTZ control: {last_error}")
    return None


if __name__ == "__main__":
    run()
