from __future__ import annotations

import importlib.resources
import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video
from tests.test_model_registry import valid_metadata

from reefvision_edge.app import create_app
from reefvision_edge.cli import settings_from_argv
from reefvision_edge.config import Settings


def test_packaged_dashboard_assets_are_served_without_runtime_network(tmp_path: Path) -> None:
    package_root = importlib.resources.files("reefvision_edge")
    index = package_root.joinpath("static", "index.html")
    assert index.is_file()

    client = TestClient(
        create_app(Settings(data_dir=tmp_path / "data")), base_url="http://127.0.0.1"
    )
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ReefVision-Edge" in response.text
    asset_path = re.search(r'(?:src|href)="(/assets/[^"]+)"', response.text)
    assert asset_path is not None
    asset = client.get(asset_path.group(1))
    assert asset.status_code == 200
    assert len(asset.content) > 100


def test_cli_refuses_lan_without_credentials_and_accepts_explicit_auth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="authentication"):
        settings_from_argv(
            ["--host", "0.0.0.0", "--data-dir", str(tmp_path / "unsafe")]
        )

    settings = settings_from_argv(
        [
            "--host",
            "0.0.0.0",
            "--data-dir",
            str(tmp_path / "safe"),
            "--username",
            "field",
            "--password",
            "secret-value",
        ]
    )

    assert settings.bind_host == "0.0.0.0"
    assert settings.auth_username == "field"
    assert settings.data_dir == tmp_path / "safe"


def test_completed_session_reopens_and_exports_after_app_restart(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    video_path = build_mjpeg_video(tmp_path / "source.avi")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as first:
        model = first.post(
            "/api/models",
            files={"model": ("tiny.onnx", model_path.read_bytes())},
            data={"metadata_json": json.dumps(valid_metadata())},
        ).json()
        source = first.post(
            "/api/sources/video",
            files={"video": ("source.avi", video_path.read_bytes())},
        ).json()
        session = first.post(
            "/api/sessions",
            json={"model_id": model["id"], "source_id": source["id"], "sample_every": 2},
        ).json()
        first.post(f"/api/sessions/{session['id']}/start")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            completed = first.get(f"/api/sessions/{session['id']}").json()
            if completed["status"] == "completed":
                break
            time.sleep(0.02)
        assert completed["status"] == "completed"

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as restarted:
        reopened = restarted.get(f"/api/sessions/{session['id']}")
        preview = restarted.get(f"/api/previews/{session['id']}")
        exported = restarted.get(f"/api/sessions/{session['id']}/export")

    assert reopened.status_code == 200
    assert reopened.json()["detection_count"] == 6
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.headers["x-frame-index"] == "4"
    assert exported.status_code == 200
    assert exported.content.startswith(b"PK")
