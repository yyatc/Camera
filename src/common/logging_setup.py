from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse, urlunparse


def sanitize_rtsp_url(url: str) -> str:
    """Маскирует учётные данные в RTSP/HTTP URL для безопасного лога."""
    try:
        p = urlparse(url)
        if p.username is not None or p.password is not None:
            host = p.hostname or ""
            port = f":{p.port}" if p.port else ""
            netloc = f"***:***@{host}{port}"
            return urlunparse((p.scheme, netloc, p.path, "", p.query, p.fragment))
    except Exception:
        pass
    # Fallback: грубо скрыть user:pass@
    return re.sub(r"//[^/@\s]+@", "//***:***@", url, count=1)


def configure_logging(level_name: str | None = None) -> None:
    """
    Настраивает корневой логгер приложения.
    Уровень: LOG_LEVEL из окружения, иначе level_name, иначе INFO.
    """
    name = (level_name or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    root.setLevel(level)

    # Шум сторонних библиотек по умолчанию приглушаем
    for noisy in (
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "urllib3",
        "ultralytics",
        "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
