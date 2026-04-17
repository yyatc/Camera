from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os
from urllib.parse import quote

import yaml
from dotenv import dotenv_values


@dataclass
class Settings:
    raw: Dict[str, Any]
    camera_host: str
    camera_username: str
    camera_password: str

    @property
    def input_rtsp(self) -> str:
        template = self.raw["stream"]["input_rtsp"]
        return template.format(
            CAMERA_HOST=self.camera_host,
            CAMERA_USERNAME=quote(self.camera_username, safe=""),
            CAMERA_PASSWORD=quote(self.camera_password, safe=""),
        )

    @property
    def output_rtsp(self) -> str:
        return self.raw["stream"]["output_rtsp"]


def load_settings(base_path: str | Path) -> Settings:
    base = Path(base_path)
    config_rel = os.getenv("CONFIG_PATH", "config/config.yaml")
    secrets_rel = os.getenv("SECRETS_PATH", "config/secrets.env")
    config_path = base / config_rel
    secrets_path = base / secrets_rel

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    secret_env = dotenv_values(secrets_path)
    camera_host = secret_env.get("CAMERA_HOST") or os.getenv("CAMERA_HOST")
    camera_username = secret_env.get("CAMERA_USERNAME") or os.getenv("CAMERA_USERNAME")
    camera_password = secret_env.get("CAMERA_PASSWORD") or os.getenv("CAMERA_PASSWORD")

    if not camera_host or not camera_username or not camera_password:
        raise ValueError("CAMERA_HOST/CAMERA_USERNAME/CAMERA_PASSWORD must be provided.")

    _validate_required_paths(raw)
    return Settings(
        raw=raw,
        camera_host=camera_host,
        camera_username=camera_username,
        camera_password=camera_password,
    )


def _validate_required_paths(raw: Dict[str, Any]) -> None:
    for section in ("app", "tracking", "ptz", "stream"):
        if section not in raw:
            raise ValueError(f"Missing section in config.yaml: {section}")
