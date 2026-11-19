from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from reefvision_edge.app import create_app
from reefvision_edge.config import Settings


def test_field_switch_config_is_remembered_after_application_restart(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    client = TestClient(create_app(settings))
    assert client.get("/api/field-switch/config").json() == {
        "mode": "inference_and_logging"
    }

    response = client.put(
        "/api/field-switch/config", json={"mode": "logging_only"}
    )
    assert response.status_code == 200
    assert response.json() == {"mode": "logging_only"}
    assert client.get("/api/field-switch/config").json() == {
        "mode": "logging_only"
    }

    restarted = TestClient(create_app(settings))
    assert restarted.get("/api/field-switch/config").json() == {
        "mode": "logging_only"
    }


def test_field_switch_config_rejects_invalid_and_unknown_fields(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(Settings(data_dir=tmp_path / "data")))
    for payload in (
        {"mode": "unrecognized"},
        {"mode": "logging_only", "unexpected": True},
        {},
    ):
        assert client.put("/api/field-switch/config", json=payload).status_code == 422
    assert client.get("/api/field-switch/config").json() == {
        "mode": "inference_and_logging"
    }
