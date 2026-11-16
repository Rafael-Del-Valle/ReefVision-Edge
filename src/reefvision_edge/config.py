from __future__ import annotations

import ipaddress
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REEFVISION_EDGE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Field(default=Path("data"))
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8765, ge=1, le=65535)
    auth_username: str | None = None
    auth_password: str | None = None

    @property
    def database_path(self) -> Path:
        return self.data_dir / "reefvision-edge.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def is_loopback(self) -> bool:
        if self.bind_host.casefold() == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.bind_host).is_loopback
        except ValueError:
            return False


def require_safe_bind(settings: Settings) -> None:
    if settings.is_loopback:
        return
    if not settings.auth_username or not settings.auth_password:
        raise ValueError(
            "Non-loopback ReefVision-Edge binding requires authentication credentials."
        )
