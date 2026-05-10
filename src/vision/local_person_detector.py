from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from src.common.types import Detection

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


class LocalPersonDetector:
    def __init__(
        self,
        confidence: float = 0.45,
        input_size: int = 640,
        model_name: str = "yolo11n.pt",
        min_bbox_area_ratio: float = 0.004,
        max_bbox_area_ratio: float = 0.75,
        min_aspect_h_w: float = 0.85,
        max_aspect_h_w: float = 4.2,
        yolo_iou: float = 0.5,
        max_detections: int = 20,
        mediapipe_enabled: bool = True,
        mediapipe_min_visibility: float = 0.45,
        inference_device: str = "cpu",
    ) -> None:
        self._confidence = confidence
        self._input_size = input_size
        self._min_bbox_area_ratio = min_bbox_area_ratio
        self._max_bbox_area_ratio = max_bbox_area_ratio
        self._min_aspect_h_w = min_aspect_h_w
        self._max_aspect_h_w = max_aspect_h_w
        self._yolo_iou = yolo_iou
        self._max_detections = max(1, int(max_detections))
        self._mediapipe_enabled = bool(mediapipe_enabled)
        self._mediapipe_min_visibility = float(mediapipe_min_visibility)
        self._inference_device = (inference_device or "cpu").strip() or "cpu"
        self._model = YOLO(model_name) if YOLO else None
        if self._model is not None and self._inference_device != "cpu":
            try:
                self._model.to(self._inference_device)
                logger.info("YOLO веса перенесены на устройство: %s", self._inference_device)
            except Exception as exc:
                logger.warning(
                    "YOLO .to(%s) не удалось (%s); predict(..., device=%s)",
                    self._inference_device,
                    exc,
                    self._inference_device,
                )
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._pose = None
        if self._mediapipe_enabled and mp is not None:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=max(0.2, min(0.9, self._confidence)),
                    min_tracking_confidence=max(0.2, min(0.9, self._confidence * 0.9)),
                )
            except Exception:
                self._pose = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if self._model is not None:
            detections.extend(self._detect_yolo(frame))
        else:
            detections.extend(self._detect_hog(frame))
        detections.extend(self._detect_mediapipe(frame))
        return self._deduplicate(detections)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        results = self._model.predict(
            frame,
            classes=[0],
            conf=self._confidence,
            iou=self._yolo_iou,
            imgsz=self._input_size,
            max_det=self._max_detections,
            device=self._inference_device,
            verbose=False,
        )
        fh, fw = frame.shape[0], frame.shape[1]
        frame_area = float(max(1, fw * fh))
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf < self._confidence:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if not self._is_plausible_person_bbox(x1, y1, x2, y2, frame_area):
                    continue
                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=conf, source="local_yolo")
                )
        return detections

    def _is_plausible_person_bbox(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_area: float,
    ) -> bool:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area_ratio = (w * h) / frame_area
        if area_ratio < self._min_bbox_area_ratio or area_ratio > self._max_bbox_area_ratio:
            return False
        aspect = h / float(w)
        if aspect < self._min_aspect_h_w or aspect > self._max_aspect_h_w:
            return False
        return True

    def _detect_hog(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        rects, weights = self._hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, source="local_hog"))
        return detections

    def _detect_mediapipe(self, frame: np.ndarray) -> List[Detection]:
        if self._pose is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return []

        h, w = frame.shape[:2]
        xs: List[float] = []
        ys: List[float] = []
        visibility_sum = 0.0
        visibility_count = 0
        for lm in result.pose_landmarks.landmark:
            vis = float(getattr(lm, "visibility", 1.0))
            if vis < self._mediapipe_min_visibility:
                continue
            visibility_sum += vis
            visibility_count += 1
            xs.append(float(lm.x))
            ys.append(float(lm.y))

        if visibility_count < 6:
            return []

        x1n, x2n = min(xs), max(xs)
        y1n, y2n = min(ys), max(ys)
        # Немного расширяем bbox, чтобы накрыть человека полностью.
        pad_x = (x2n - x1n) * 0.20
        pad_y = (y2n - y1n) * 0.25
        x1n = max(0.0, x1n - pad_x)
        y1n = max(0.0, y1n - pad_y)
        x2n = min(1.0, x2n + pad_x)
        y2n = min(1.0, y2n + pad_y)

        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
        frame_area = float(max(1, w * h))
        if not self._is_plausible_person_bbox(x1, y1, x2, y2, frame_area):
            return []
        conf = min(0.99, max(0.35, visibility_sum / visibility_count))
        return [Detection(bbox=(x1, y1, x2, y2), confidence=conf, source="local_mediapipe")]

    def _deduplicate(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        out: List[Detection] = []
        # Сначала более уверенные боксы.
        for d in sorted(detections, key=lambda x: x.confidence, reverse=True):
            keep = True
            for e in out:
                if _iou(d.bbox, e.bbox) >= 0.55:
                    keep = False
                    break
            if keep:
                out.append(d)
        return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)
