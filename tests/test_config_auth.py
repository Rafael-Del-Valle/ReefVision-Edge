from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reefvision_edge.app import create_app
from reefvision_edge.config import Settings, require_safe_bind


def test_loopback_is_the_default_and_needs_no_credentials(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.bind_host == "127.0.0.1"
    assert settings.bind_port == 8765
    require_safe_bind(settings)


def test_non_loopback_bind_refuses_missing_credentials(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, bind_host="0.0.0.0")

    with pytest.raises(ValueError, match="authentication"):
        require_safe_bind(settings)


def test_lan_auth_covers_ui_api_and_preview_routes(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        bind_host="0.0.0.0",
        auth_username="field",
        auth_password="secret-value",
    )
    client = TestClient(create_app(settings))

    for path in ("/", "/api/health", "/api/previews/missing"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == 'Basic realm="ReefVision-Edge"'

    token = base64.b64encode(b"field:secret-value").decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    assert client.get("/", headers=headers).status_code != 401
    health = client.get("/api/health", headers=headers)
    assert health.status_code == 200
    assert health.json()["product"] == "ReefVision-Edge"
    assert client.get("/api/previews/missing", headers=headers).status_code == 404


def test_bad_basic_credentials_are_rejected(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        bind_host="192.168.1.20",
        auth_username="field",
        auth_password="secret-value",
    )
    token = base64.b64encode(b"field:wrong").decode("ascii")

    response = TestClient(create_app(settings)).get(
        "/api/health", headers={"Authorization": f"Basic {token}"}
    )

    assert response.status_code == 401
