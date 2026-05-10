from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_inference_device(spec: Optional[str]) -> str:
    """
    Строка для Ultralytics YOLO predict(..., device=...): cpu, 0, cuda:0, mps.

    spec (обычно из конфига или после приоритета ML_DEVICE в main):
      - None / "" / "auto" — CUDA при наличии, иначе Apple MPS, иначе CPU.
      - "cpu" — принудительно CPU.
      - "0", "1", ... — индекс GPU CUDA.
      - "cuda", "cuda:0" — явный CUDA.
      - "mps" — Apple Silicon (если недоступно — fallback на CPU с предупреждением).
    """
    raw = (spec or "").strip()
    key = raw.lower() if raw else "auto"

    if key in ("", "auto"):
        return _auto_device()

    if key == "cpu":
        logger.info("ML inference device: CPU (явно)")
        return "cpu"

    if key == "mps":
        if _mps_available():
            logger.info("ML inference device: MPS (явно)")
            return "mps"
        logger.warning("ML device=mps запрошен, но MPS недоступен — используется CPU")
        return "cpu"

    if key.startswith("cuda"):
        if not _cuda_available():
            logger.warning("ML device=%s запрошен, но CUDA недоступна — используется CPU", raw)
            return "cpu"
        out = raw if ":" in raw else "cuda:0"
        logger.info("ML inference device: %s", out)
        return out

    if raw.isdigit():
        if not _cuda_available():
            logger.warning("ML device GPU %s запрошен, но CUDA недоступна — используется CPU", raw)
            return "cpu"
        logger.info("ML inference device: CUDA GPU index %s", raw)
        return raw

    logger.warning("Неизвестное ML device=%r — используется CPU", raw)
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            dev = "0"
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
            logger.info("ML inference device (auto): CUDA %s (%s)", dev, name)
            return dev
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("ML inference device (auto): MPS (Apple GPU)")
            return "mps"
    except Exception as exc:
        logger.debug("torch недоступен при выборе device: %s", exc)
    logger.info("ML inference device (auto): CPU")
    return "cpu"
