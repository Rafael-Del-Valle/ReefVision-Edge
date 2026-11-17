from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from tests.test_model_registry import conversion_provenance
from tests.test_sessions import prepared_services

from reefvision_edge.domain import ModelMetadata
from reefvision_edge.exports import ExportError, ExportService, validate_export
from reefvision_edge.model_registry import canonical_metadata_bytes


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
        assert bundle.namelist() == [
            "manifest.json",
            "frames.csv",
            "detections.csv",
        ]
        manifest = json.loads(bundle.read("manifest.json"))
        frame_rows = list(csv.DictReader(io.StringIO(bundle.read("frames.csv").decode())))
        rows = list(csv.DictReader(io.StringIO(bundle.read("detections.csv").decode())))
    assert manifest["schema"] == "reefvision-edge-session/v1"
    assert manifest["session"]["id"] == session.id
    assert manifest["session"]["status"] == "completed"
    assert manifest["sampling"] == {"sample_every": 2, "first_frame_index": 0}
    assert manifest["inference_profile"] == {
        "name": "full_frame",
        "source_tile_width": None,
        "source_tile_height": None,
        "overlap": None,
        "merge": "class_aware_nms",
    }
    assert manifest["timestamp_basis"] == "frame_index / reported_fps"
    assert manifest["model"]["sha256"] == model.model_sha256
    assert manifest["model"]["metadata_sha256"] == model.metadata_sha256
    assert manifest["model"]["provider"] == model.provider
    assert manifest["model"]["class_map"] == {"0": "coral", "1": "fish"}
    assert manifest["source"]["sha256"] == source.sha256
    assert manifest["source"]["dimensions"] == {"width": 160, "height": 90}
    assert manifest["thresholds"] == {"confidence": 0.5, "iou": 0.45}
    assert manifest["compute"]["requested"] == "auto"
    assert manifest["compute"]["actual_device"] == "CPU"
    assert manifest["recording"]["record_data"] is True
    assert manifest["recording"]["record_monitor"] is False
    assert manifest["latency_statistics"] == {
        "kind": "rolling_window",
        "window_frames": 4096,
    }
    assert [frame["frame_index"] for frame in manifest["frames"]] == [0, 2, 4]
    assert [frame["detection_count"] for frame in manifest["frames"]] == [2, 2, 2]
    assert len(rows) == 6
    assert [row["inference_status"] for row in frame_rows] == [
        "inferred",
        "inferred",
        "inferred",
    ]
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
        first_frames = bundle.read("frames.csv")
        first_csv = bundle.read("detections.csv")
    second = exporter.build_zip(session.id)
    with zipfile.ZipFile(second) as bundle:
        second_manifest = bundle.read("manifest.json")
        second_frames = bundle.read("frames.csv")
        second_csv = bundle.read("detections.csv")

    assert second_manifest == first_manifest
    assert second_frames == first_frames
    assert second_csv == first_csv


def test_export_includes_monitor_video_and_independent_overlay_settings(
    tmp_path: Path,
) -> None:
    database, model, source, _, runner, service = prepared_services(tmp_path)
    from reefvision_edge.domain import OverlaySettings

    session = service.create(
        model.id,
        source.id,
        record_monitor=True,
        preview_overlays=OverlaySettings(timer=False, boxes=True, labels=False),
        recording_overlays=OverlaySettings(timer=True, boxes=False, labels=True),
    )
    runner.run(session.id)
    archive = ExportService(database, tmp_path / "exports").build_zip(session.id)

    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        video = bundle.read("monitor.mp4")
        names = bundle.namelist()
    assert names == ["manifest.json", "frames.csv", "detections.csv", "monitor.mp4"]
    assert len(video) > 0
    assert manifest["recording"]["preview_overlays"] == {
        "timer": False,
        "trails": False,
        "boxes": True,
        "masks": False,
        "labels": False,
    }
    assert manifest["recording"]["recording_overlays"]["timer"] is True
    assert manifest["recording"]["recording_overlays"]["boxes"] is False


def test_round_trip_validator_checks_counts_coordinates_and_hashes(tmp_path: Path) -> None:
    _, _, _, session, exporter = completed_export(tmp_path)
    archive = exporter.build_zip(session.id)

    summary = validate_export(archive)

    assert summary.session_id == session.id
    assert summary.frame_count == 3
    assert summary.detection_count == 6
    assert summary.class_names == ("coral", "fish")


def test_round_trip_preserves_frames_with_no_detections(tmp_path: Path) -> None:
    database, _, source, session, exporter = completed_export(tmp_path)
    with database.connect() as connection:
        connection.execute("DELETE FROM detections")
        connection.execute(
            "UPDATE sessions SET detection_count = 0 WHERE id = ?", (session.id,)
        )
        connection.execute(
            "UPDATE sources SET metadata_json = ? WHERE id = ?",
            (json.dumps({"timestamp_basis": "capture order"}), source.id),
        )

    archive = exporter.build_zip(session.id)
    summary = validate_export(archive)

    assert summary.frame_count == 3
    assert summary.detection_count == 0
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        rows = list(csv.DictReader(io.StringIO(bundle.read("detections.csv").decode())))
    assert manifest["source"]["dimensions"] == {"width": 160, "height": 90}
    assert [frame["detection_count"] for frame in manifest["frames"]] == [0, 0, 0]
    assert rows == []


def test_validator_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../manifest.json", "{}")

    with pytest.raises(ExportError, match="unsafe"):
        validate_export(archive)


def test_validator_rejects_inconsistent_row_provenance_and_malformed_numbers(
    tmp_path: Path,
) -> None:
    _, _, _, session, exporter = completed_export(tmp_path)
    original = exporter.build_zip(session.id)
    with zipfile.ZipFile(original) as bundle:
        manifest = bundle.read("manifest.json")
        rows = list(csv.DictReader(io.StringIO(bundle.read("detections.csv").decode())))

    rows[0]["model_id"] = "different-model"
    rows[0]["score"] = "not-a-number"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as bundle:
        bundle.writestr("manifest.json", manifest)
        bundle.writestr("detections.csv", output.getvalue())

    with pytest.raises(ExportError, match="different model"):
        validate_export(tampered)

    rows[0]["model_id"] = json.loads(manifest)["model"]["id"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(tampered, "w") as bundle:
        bundle.writestr("manifest.json", manifest)
        bundle.writestr("detections.csv", output.getvalue())
    with pytest.raises(ExportError, match="score must be numeric"):
        validate_export(tampered)


def test_checked_in_sample_round_trips() -> None:
    summary = validate_export(Path("examples/session-export"))

    assert summary.session_id == "00000000-0000-4000-8000-000000000001"
    assert summary.frame_count == 1
    assert summary.detection_count == 1


def test_export_includes_converted_checkpoint_provenance(tmp_path: Path) -> None:
    database, model, _, session, exporter = completed_export(tmp_path)
    metadata_value = model.metadata.model_dump(mode="json")
    provenance = conversion_provenance(
        b"trusted checkpoint", model.path.read_bytes(), ("coral", "fish")
    )
    metadata_value["conversion"] = provenance.model_dump(mode="json")
    metadata_value["supported_profiles"] = ["full_frame", "thesis_tiling"]
    metadata = ModelMetadata.model_validate(metadata_value)
    metadata_bytes = canonical_metadata_bytes(metadata)
    with database.connect() as connection:
        connection.execute(
            "UPDATE models SET metadata_json = ?, metadata_sha256 = ? WHERE id = ?",
            (
                metadata_bytes.decode("utf-8"),
                __import__("hashlib").sha256(metadata_bytes).hexdigest(),
                model.id,
            ),
        )

    archive = exporter.build_zip(session.id)

    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["model"]["source_format"] == "ultralytics_pt"
    assert manifest["model"]["source_sha256"] == provenance.source_sha256
    assert manifest["model"]["conversion"]["exporter_version"] == "8.4.215"
    assert manifest["model"]["supported_profiles"] == [
        "full_frame",
        "thesis_tiling",
    ]
