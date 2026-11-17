from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reefvision_edge.db import Database
from reefvision_edge.domain import LATENCY_STATISTICS_WINDOW_FRAMES, ModelMetadata
from reefvision_edge.tiling import (
    THESIS_TILE_HEIGHT,
    THESIS_TILE_OVERLAP,
    THESIS_TILE_WIDTH,
)


class ExportError(ValueError):
    pass


@dataclass(frozen=True)
class ExportSummary:
    session_id: str
    frame_count: int
    detection_count: int
    class_names: tuple[str, ...]


CSV_FIELDS = (
    "schema",
    "session_id",
    "source_id",
    "model_id",
    "frame_index",
    "timestamp_ms",
    "source_width",
    "source_height",
    "class_id",
    "class_name",
    "score",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
)

FRAME_CSV_FIELDS = (
    "schema",
    "session_id",
    "frame_index",
    "timestamp_ms",
    "source_width",
    "source_height",
    "latency_ms",
    "inference_status",
    "detection_count",
)


class ExportService:
    def __init__(self, database: Database, output_dir: Path) -> None:
        self.database = database
        self.output_dir = output_dir

    def build_zip(self, session_id: str) -> Path:
        manifest, frames_csv, detections_csv, attachments = self._serialize(session_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"{session_id}.zip"
        temporary = self.output_dir / f".{session_id}.tmp"
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            bundle.writestr("manifest.json", manifest)
            bundle.writestr("frames.csv", frames_csv)
            bundle.writestr("detections.csv", detections_csv)
            for name, path in attachments:
                bundle.write(path, name)
        os.replace(temporary, target)
        return target

    def _serialize(
        self, session_id: str
    ) -> tuple[bytes, bytes, bytes, list[tuple[str, Path]]]:
        with self.database.connect() as connection:
            session = connection.execute(
                """
                SELECT id, model_id, source_id, status, settings_json, provider,
                    requested_compute, actual_device, provider_reason, frame_count,
                    detection_count, input_fps, inference_fps, average_latency_ms,
                    median_latency_ms, p95_latency_ms, peak_vram_mb, dropped_frames,
                    operator_profile_json, operator_profile_path, monitor_video_path,
                    error_message, created_at, updated_at, completed_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                raise ExportError(f"Session {session_id} was not found.")
            if session[3] in {"queued", "running"}:
                raise ExportError("A running or queued session cannot be exported.")
            model = connection.execute(
                """
                SELECT id, name, model_sha256, metadata_sha256, metadata_json, provider
                FROM models WHERE id = ?
                """,
                (session[1],),
            ).fetchone()
            source = connection.execute(
                """
                SELECT id, kind, name, sha256, metadata_json
                FROM sources WHERE id = ?
                """,
                (session[2],),
            ).fetchone()
            frame_rows = connection.execute(
                """
                SELECT f.frame_index, f.timestamp_ms, f.width, f.height,
                    f.latency_ms, f.inference_status, COUNT(d.id)
                FROM frames f LEFT JOIN detections d ON d.frame_id = f.id
                WHERE f.session_id = ?
                GROUP BY f.id
                ORDER BY f.frame_index
                """,
                (session_id,),
            ).fetchall()
            rows = connection.execute(
                """
                SELECT f.frame_index, f.timestamp_ms, f.width, f.height,
                    d.class_id, d.class_name, d.score,
                    d.x_min, d.y_min, d.x_max, d.y_max, d.id
                FROM frames f JOIN detections d ON d.frame_id = f.id
                WHERE f.session_id = ?
                ORDER BY f.frame_index, d.class_id, d.id
                """,
                (session_id,),
            ).fetchall()
        if model is None or source is None:
            raise ExportError("Session provenance is incomplete.")
        settings = json.loads(session[4])
        metadata = ModelMetadata.model_validate_json(model[4])
        source_metadata = json.loads(source[4])
        source_width = source_metadata.get("width")
        source_height = source_metadata.get("height")
        if frame_rows:
            source_width = source_width or frame_rows[0][2]
            source_height = source_height or frame_rows[0][3]
        manifest_value = {
            "schema": "reefvision-edge-session/v1",
            "exported_at": session[24] or session[23],
            "session": {
                "id": session[0],
                "status": session[3],
                "created_at": session[22],
                "updated_at": session[23],
                "completed_at": session[24],
                "frame_count": session[9],
                "detection_count": session[10],
                "input_fps": session[11],
                "inference_fps": session[12],
                "average_latency_ms": session[13],
                "median_latency_ms": session[14],
                "p95_latency_ms": session[15],
                "peak_vram_mb": session[16],
                "dropped_frames": session[17],
                "error_message": session[21],
            },
            "compute": {
                "requested": session[6],
                "actual_device": session[7],
                "active_provider": session[5] or model[5],
                "explanation": session[8],
                "operator_profile": json.loads(session[18]) if session[18] else None,
                "profile_artifact": (
                    "operator-profile.json" if session[19] else None
                ),
            },
            "latency_statistics": {
                "kind": "rolling_window",
                "window_frames": LATENCY_STATISTICS_WINDOW_FRAMES,
            },
            "inference_profile": {
                "name": settings.get("inference_profile", "full_frame"),
                "source_tile_width": (
                    THESIS_TILE_WIDTH
                    if settings.get("inference_profile") == "thesis_tiling"
                    else None
                ),
                "source_tile_height": (
                    THESIS_TILE_HEIGHT
                    if settings.get("inference_profile") == "thesis_tiling"
                    else None
                ),
                "overlap": (
                    THESIS_TILE_OVERLAP
                    if settings.get("inference_profile") == "thesis_tiling"
                    else None
                ),
                "merge": "class_aware_nms",
            },
            "recording": {
                "record_data": bool(settings.get("record_data", True)),
                "record_monitor": bool(settings.get("record_monitor", False)),
                "preview_overlays": settings.get("preview_overlays", {}),
                "recording_overlays": settings.get("recording_overlays", {}),
                "monitor_video_artifact": "monitor.mp4" if session[20] else None,
            },
            "model": {
                "id": model[0],
                "name": model[1],
                "sha256": model[2],
                "metadata_sha256": model[3],
                "format": metadata.format,
                "input": {
                    "width": metadata.input_width,
                    "height": metadata.input_height,
                },
                "class_map": {str(key): value for key, value in metadata.classes.items()},
                "provider": session[5] or model[5],
                "source_format": (
                    metadata.conversion.source_format if metadata.conversion else "onnx"
                ),
                "source_sha256": (
                    metadata.conversion.source_sha256
                    if metadata.conversion
                    else model[2]
                ),
                "conversion": (
                    metadata.conversion.model_dump(mode="json")
                    if metadata.conversion
                    else None
                ),
                "supported_profiles": [
                    profile.value for profile in metadata.supported_profiles
                ],
            },
            "source": {
                "id": source[0],
                "kind": source[1],
                "name": source[2],
                "sha256": source[3],
                "container": source_metadata.get("container"),
                "codec": source_metadata.get("codec"),
                "dimensions": {
                    "width": source_width,
                    "height": source_height,
                },
                "reported_fps": source_metadata.get("fps"),
            },
            "frames": [
                {
                    "frame_index": row[0],
                    "timestamp_ms": row[1],
                    "width": row[2],
                    "height": row[3],
                    "latency_ms": row[4],
                    "inference_status": row[5],
                    "detection_count": row[6],
                }
                for row in frame_rows
            ],
            "sampling": {
                "sample_every": int(settings["sample_every"]),
                "first_frame_index": 0,
            },
            "timestamp_basis": source_metadata.get(
                "timestamp_basis", "frame_index / reported_fps"
            ),
            "coordinate_space": "original_source_pixels_xyxy",
            "thresholds": {
                "confidence": metadata.confidence_threshold,
                "iou": metadata.iou_threshold,
            },
        }
        manifest_bytes = (
            json.dumps(manifest_value, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        frame_output = io.StringIO(newline="")
        frame_writer = csv.DictWriter(
            frame_output, fieldnames=FRAME_CSV_FIELDS, lineterminator="\n"
        )
        frame_writer.writeheader()
        for row in frame_rows:
            frame_writer.writerow(
                {
                    "schema": "reefvision-edge-frame/v1",
                    "session_id": session[0],
                    "frame_index": row[0],
                    "timestamp_ms": "" if row[1] is None else f"{row[1]:.3f}",
                    "source_width": row[2],
                    "source_height": row[3],
                    "latency_ms": f"{row[4]:.6f}",
                    "inference_status": row[5],
                    "detection_count": row[6],
                }
            )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "schema": "reefvision-edge-detection/v1",
                    "session_id": session[0],
                    "source_id": source[0],
                    "model_id": model[0],
                    "frame_index": row[0],
                    "timestamp_ms": "" if row[1] is None else f"{row[1]:.3f}",
                    "source_width": row[2],
                    "source_height": row[3],
                    "class_id": row[4],
                    "class_name": row[5],
                    "score": f"{row[6]:.8f}",
                    "x_min": f"{row[7]:.3f}",
                    "y_min": f"{row[8]:.3f}",
                    "x_max": f"{row[9]:.3f}",
                    "y_max": f"{row[10]:.3f}",
                }
            )
        attachments: list[tuple[str, Path]] = []
        if session[20] and Path(session[20]).is_file():
            attachments.append(("monitor.mp4", Path(session[20])))
        if session[19] and Path(session[19]).is_file():
            attachments.append(("operator-profile.json", Path(session[19])))
        return (
            manifest_bytes,
            frame_output.getvalue().encode("utf-8"),
            output.getvalue().encode("utf-8"),
            attachments,
        )


def _read_export(path: Path) -> tuple[bytes, bytes]:
    if path.is_dir():
        return (path / "manifest.json").read_bytes(), (path / "detections.csv").read_bytes()
    try:
        with zipfile.ZipFile(path) as bundle:
            for name in bundle.namelist():
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts:
                    raise ExportError(f"Export contains unsafe ZIP member {name}.")
            return bundle.read("manifest.json"), bundle.read("detections.csv")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ExportError(f"Export bundle is incomplete or invalid: {exc}") from exc


def _integer(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{field} must be an integer.") from exc


def _number(value: object, field: str) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{field} must be numeric.") from exc


def validate_export(path: Path) -> ExportSummary:
    try:
        manifest_bytes, csv_bytes = _read_export(path)
        manifest = json.loads(manifest_bytes)
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ExportError(f"Export could not be read: {exc}") from exc
    if manifest.get("schema") != "reefvision-edge-session/v1":
        raise ExportError("Unsupported export manifest schema.")
    session = manifest.get("session", {})
    model = manifest.get("model", {})
    source = manifest.get("source", {})
    frames = manifest.get("frames", [])
    if not isinstance(session, dict) or not isinstance(model, dict) or not isinstance(source, dict):
        raise ExportError("Manifest provenance records are invalid.")
    class_map = model.get("class_map", {})
    dimensions = source.get("dimensions", {})
    if not isinstance(class_map, dict) or not isinstance(dimensions, dict):
        raise ExportError("Manifest class map or source dimensions are invalid.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model.get("sha256", ""))):
        raise ExportError("Model SHA-256 is missing or invalid.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model.get("metadata_sha256", ""))):
        raise ExportError("Metadata SHA-256 is missing or invalid.")
    if not isinstance(frames, list):
        raise ExportError("Frame records are missing or invalid.")
    frame_map: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ExportError("Frame record is invalid.")
        frame_index = _integer(frame.get("frame_index", -1), "frame_index")
        if frame_index < 0 or frame_index in frame_map:
            raise ExportError("Frame indices must be unique non-negative integers.")
        if _integer(frame.get("width", 0), "frame width") <= 0 or _integer(
            frame.get("height", 0), "frame height"
        ) <= 0:
            raise ExportError("Frame dimensions must be positive.")
        frame_map[frame_index] = frame
    source_width = _integer(dimensions.get("width") or 0, "source width")
    source_height = _integer(
        dimensions.get("height") or 0, "source height"
    )
    if frames and (source_width <= 0 or source_height <= 0):
        raise ExportError("Source dimensions are missing or invalid.")
    session_id = str(session.get("id", ""))
    detection_counts: dict[int, int] = {}
    for row in rows:
        if row.get("schema") != "reefvision-edge-detection/v1":
            raise ExportError("Unsupported detection row schema.")
        if row.get("session_id") != session_id:
            raise ExportError("Detection row belongs to a different session.")
        if row.get("model_id") != str(model.get("id", "")):
            raise ExportError("Detection row references a different model.")
        if row.get("source_id") != str(source.get("id", "")):
            raise ExportError("Detection row references a different source.")
        frame_index = _integer(row.get("frame_index"), "detection frame_index")
        frame = frame_map.get(frame_index)
        if frame is None:
            raise ExportError("Detection row references an unknown frame.")
        detection_counts[frame_index] = detection_counts.get(frame_index, 0) + 1
        class_id = _integer(row.get("class_id"), "class_id")
        if class_map.get(str(class_id)) != row["class_name"]:
            raise ExportError("Detection class does not match the manifest class map.")
        score = _number(row.get("score"), "score")
        x_min, y_min, x_max, y_max = (
            _number(row.get(key), key) for key in ("x_min", "y_min", "x_max", "y_max")
        )
        if not 0 <= score <= 1:
            raise ExportError("Detection score is outside [0, 1].")
        width = _integer(frame.get("width"), "frame width")
        height = _integer(frame.get("height"), "frame height")
        if _integer(row.get("source_width"), "source_width") != width or _integer(
            row.get("source_height"), "source_height"
        ) != height:
            raise ExportError("Detection dimensions do not match the frame record.")
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ExportError("Detection coordinates are outside source dimensions.")
    recording = manifest.get("recording", {})
    record_data = not isinstance(recording, dict) or bool(recording.get("record_data", True))
    expected_detections = (
        _integer(session.get("detection_count", -1), "detection_count")
        if record_data
        else 0
    )
    if len(rows) != expected_detections:
        raise ExportError("Detection count does not match the manifest.")
    expected_frames = (
        _integer(session.get("frame_count", -1), "frame_count") if record_data else 0
    )
    if len(frame_map) != expected_frames:
        raise ExportError("Frame count does not match the manifest.")
    for frame_index, frame in frame_map.items():
        if _integer(
            frame.get("detection_count", -1), "frame detection_count"
        ) != detection_counts.get(frame_index, 0):
            raise ExportError("Per-frame detection count does not match the manifest.")
    try:
        class_names = tuple(class_map[str(index)] for index in range(len(class_map)))
    except KeyError as exc:
        raise ExportError("Manifest class IDs must be contiguous from zero.") from exc
    return ExportSummary(
        session_id=session_id,
        frame_count=len(frame_map),
        detection_count=len(rows),
        class_names=class_names,
    )
