from __future__ import annotations

import json
import time
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video
from tests.test_model_registry import valid_metadata

from reefvision_edge.app import create_app
from reefvision_edge.config import Settings


def test_api_runs_recorded_field_workflow_end_to_end(tmp_path: Path) -> None:
    client = TestClient(
        create_app(Settings(data_dir=tmp_path / "data")), base_url="http://127.0.0.1"
    )
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    video_path = build_mjpeg_video(tmp_path / "source.avi")

    model_response = client.post(
        "/api/models",
        files={"model": ("tiny.onnx", model_path.read_bytes(), "application/octet-stream")},
        data={"metadata_json": json.dumps(valid_metadata())},
    )
    assert model_response.status_code == 201
    model = model_response.json()
    assert model["name"] == "Tiny Coral Detector"
    assert model["provider"] == "CPUExecutionProvider"
    assert len(model["model_sha256"]) == 64

    source_response = client.post(
        "/api/sources/video",
        files={"video": ("source.avi", video_path.read_bytes(), "video/x-msvideo")},
    )
    assert source_response.status_code == 201
    source = source_response.json()
    assert source["codec"].strip() == "MJPG"
    assert source["width"] == 160

    assert client.get("/api/capture-devices").status_code == 200
    selected = client.post("/api/monitor/select", json={"source_id": source["id"]})
    assert selected.status_code == 202
    preview_deadline = time.monotonic() + 2
    while time.monotonic() < preview_deadline:
        monitor_status = client.get("/api/monitor/status").json()
        if monitor_status["frame_index"] >= 0:
            break
        time.sleep(0.01)
    assert monitor_status["state"] == "replaying"
    monitor_frame = client.get("/api/monitor/frame")
    assert monitor_frame.headers["content-type"] == "image/jpeg"
    assert int(monitor_frame.headers["x-frame-index"]) >= 0
    assert float(monitor_frame.headers["x-frame-published-at"]) > 0

    created_response = client.post(
        "/api/sessions",
        json={
            "model_id": model["id"],
            "source_id": source["id"],
            "sample_every": 2,
            "compute_device": "cpu",
            "record_data": True,
            "record_monitor": False,
            "preview_overlays": {
                "timer": False,
                "trails": False,
                "boxes": True,
                "masks": False,
                "labels": True,
            },
            "recording_overlays": {
                "timer": True,
                "trails": False,
                "boxes": False,
                "masks": False,
                "labels": True,
            },
        },
    )
    assert created_response.status_code == 201
    created = created_response.json()
    session_id = created["id"]
    assert created["compute_device"] == "cpu"
    assert created["record_data"] is True
    assert created["record_monitor"] is False
    assert created["preview_overlays"]["timer"] is False
    assert created["recording_overlays"]["boxes"] is False
    assert client.post(f"/api/sessions/{session_id}/start").status_code == 202
    assert client.get("/api/monitor/status").json()["state"] == "idle"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        session = client.get(f"/api/sessions/{session_id}").json()
        if session["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert session["status"] == "completed"
    assert session["frame_count"] == 3
    assert session["detection_count"] == 6
    assert session["provider"] == "CPUExecutionProvider"
    assert session["actual_device"] == "CPU"
    assert session["provider_reason"] == "CPU was selected explicitly."
    assert session["median_latency_ms"] >= 0
    assert session["p95_latency_ms"] >= session["median_latency_ms"]

    preview = client.get(f"/api/previews/{session_id}")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert int(preview.headers["x-frame-index"]) == 4
    assert int(preview.headers["x-detection-count"]) == 2
    assert float(preview.headers["x-frame-published-at"]) > 0

    export = client.get(f"/api/sessions/{session_id}/export")
    assert export.status_code == 200
    assert export.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(export.content)) as bundle:
        assert bundle.namelist() == [
            "manifest.json",
            "frames.csv",
            "detections.csv",
        ]

    sessions = client.get("/api/sessions").json()
    assert [item["id"] for item in sessions] == [session_id]


def test_api_returns_structured_model_video_and_camera_errors(tmp_path: Path) -> None:
    client = TestClient(
        create_app(Settings(data_dir=tmp_path / "data")), base_url="http://127.0.0.1"
    )

    invalid_model = client.post(
        "/api/models",
        files={"model": ("checkpoint.pt", b"weights", "application/octet-stream")},
        data={"metadata_json": "{}"},
    )
    assert invalid_model.status_code == 422
    assert invalid_model.json()["error"]["code"] == "invalid_model"
    assert "ONNX" in invalid_model.json()["error"]["message"]

    unreadable = client.post(
        "/api/sources/video",
        files={"video": ("broken.mp4", b"not video", "video/mp4")},
    )
    assert unreadable.status_code == 422
    assert unreadable.json()["error"]["code"] == "invalid_source"
    assert "decode" in unreadable.json()["error"]["message"].casefold()

    camera = client.post("/api/sources/camera", json={"device": "/dev/video999"})
    assert camera.status_code == 422
    assert camera.json()["error"]["code"] == "camera_unavailable"


def test_capabilities_and_settings_report_runtime_without_secrets(tmp_path: Path) -> None:
    client = TestClient(
        create_app(Settings(data_dir=tmp_path / "data")), base_url="http://127.0.0.1"
    )

    capabilities = client.get("/api/capabilities").json()
    settings = client.get("/api/settings").json()

    assert "CPUExecutionProvider" in capabilities["available_providers"]
    assert capabilities["camera_backend"] in {
        "V4L2",
        "Media Foundation / DirectShow",
        "unavailable",
    }
    assert capabilities["offline_runtime"] is True
    assert settings["bind_host"] == "127.0.0.1"
    assert settings["auth_required"] is False
    assert settings["data_dir"] == str(tmp_path / "data")
    assert "auth_password" not in settings
