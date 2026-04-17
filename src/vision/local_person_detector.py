from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.common.types import Detection

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


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
    ) -> None:
        self._confidence = confidence
        self._input_size = input_size
        self._min_bbox_area_ratio = min_bbox_area_ratio
        self._max_bbox_area_ratio = max_bbox_area_ratio
        self._min_aspect_h_w = min_aspect_h_w
        self._max_aspect_h_w = max_aspect_h_w
        self._yolo_iou = yolo_iou
        self._model = YOLO(model_name) if YOLO else None
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is not None:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        results = self._model.predict(
            frame,
            classes=[0],
            conf=self._confidence,
            iou=self._yolo_iou,
            imgsz=self._input_size,
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
