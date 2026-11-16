from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from reefvision_edge.exports import ExportError, ExportService, validate_export
from tests.test_sessions import prepared_services


def completed_export(tmp_path: Path):
    database, model, source, _, runner, service = prepared_services(tmp_path)
    session = service.create(model.id, source.id, sample_every=2)
    runner.run(session.id)
    exporter = ExportService(database, tmp_path / "exports")
    return database, model, source, service.get(session.id), exporter


def test_export_contains_versioned_provenance_and_original_coordinate_csv(
    tmp_path: Path,
) -> None:
    _, model, source, session, exporter = completed_export(tmp_path)

    archive = exporter.build_zip(session.id)

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["manifest.json", "detections.csv"]
        manifest = json.loads(bundle.read("manifest.json"))
        rows = list(csv.DictReader(io.StringIO(bundle.read("detections.csv").decode())))
    assert manifest["schema"] == "reefvision-edge-session/v1"
    assert manifest["session"]["id"] == session.id
    assert manifest["session"]["status"] == "completed"
    assert manifest["sampling"] == {"sample_every": 2, "first_frame_index": 0}
    assert manifest["timestamp_basis"] == "frame_index / reported_fps"
    assert manifest["model"]["sha256"] == model.model_sha256
    assert manifest["model"]["metadata_sha256"] == model.metadata_sha256
    assert manifest["model"]["provider"] == model.provider
    assert manifest["model"]["class_map"] == {"0": "coral", "1": "fish"}
    assert manifest["source"]["sha256"] == source.sha256
    assert manifest["source"]["dimensions"] == {"width": 160, "height": 90}
    assert manifest["thresholds"] == {"confidence": 0.5, "iou": 0.45}
    assert len(rows) == 6
    assert rows[0]["schema"] == "reefvision-edge-detection/v1"
    assert rows[0]["frame_index"] == "0"
    assert rows[0]["timestamp_ms"] == "0.000"
    assert rows[0]["class_name"] == "coral"
    assert [rows[0][key] for key in ("x_min", "y_min", "x_max", "y_max")] == [
        "60.000",
        "35.000",
        "100.000",
        "55.000",
    ]


def test_export_is_deterministic_for_same_persisted_session(tmp_path: Path) -> None:
    _, _, _, session, exporter = completed_export(tmp_path)

    first = exporter.build_zip(session.id)
    with zipfile.ZipFile(first) as bundle:
        first_manifest = bundle.read("manifest.json")
        first_csv = bundle.read("detections.csv")
    second = exporter.build_zip(session.id)
    with zipfile.ZipFile(second) as bundle:
        second_manifest = bundle.read("manifest.json")
        second_csv = bundle.read("detections.csv")

    assert second_manifest == first_manifest
    assert second_csv == first_csv


def test_round_trip_validator_checks_counts_coordinates_and_hashes(tmp_path: Path) -> None:
    _, _, _, session, exporter = completed_export(tmp_path)
    archive = exporter.build_zip(session.id)

    summary = validate_export(archive)

    assert summary.session_id == session.id
    assert summary.frame_count == 3
    assert summary.detection_count == 6
    assert summary.class_names == ("coral", "fish")


def test_validator_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../manifest.json", "{}")

    with pytest.raises(ExportError, match="unsafe"):
        validate_export(archive)


def test_checked_in_sample_round_trips() -> None:
    summary = validate_export(Path("examples/session-export"))

    assert summary.session_id == "00000000-0000-4000-8000-000000000001"
    assert summary.frame_count == 1
    assert summary.detection_count == 1
