from __future__ import annotations

import json
import platform
from dataclasses import asdict

import onnxruntime as ort
from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from reefvision_edge.capture_devices import discover_capture_devices
from reefvision_edge.compute import ComputeChoice
from reefvision_edge.config import Settings
from reefvision_edge.domain import InferenceProfile, OverlaySettings, RegisteredModel
from reefvision_edge.exports import ExportError, ExportService
from reefvision_edge.field_controls import (
    FieldSwitchPreferences,
    load_field_switch_preferences,
    save_field_switch_preferences,
)
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_imports import ModelImportError, ModelImportService
from reefvision_edge.model_registry import ModelRegistrationError, ModelRegistry
from reefvision_edge.monitor import MonitorPreviewService
from reefvision_edge.monitor_video import overlay_capabilities
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sessions import SessionService
from reefvision_edge.sources import SourceError


class CameraRequest(BaseModel):
    device: str = Field(min_length=1)


class SessionRequest(BaseModel):
    model_id: str
    source_id: str
    sample_every: int = Field(default=1, ge=1, le=10000)
    inference_profile: InferenceProfile = InferenceProfile.FULL_FRAME
    compute_device: ComputeChoice = ComputeChoice.AUTO
    record_data: bool = True
    record_monitor: bool = False
    preview_overlays: OverlaySettings = OverlaySettings()
    recording_overlays: OverlaySettings = OverlaySettings()
    enable_profiling: bool = False


class MonitorSelectRequest(BaseModel):
    source_id: str = Field(min_length=1)


class ModelConvertRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=160)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def model_response(registered: RegisteredModel) -> dict[str, object]:
    conversion = registered.metadata.conversion
    return {
        "id": registered.id,
        "name": registered.metadata.name,
        "format": registered.metadata.format,
        "input_width": registered.metadata.input_width,
        "input_height": registered.metadata.input_height,
        "classes": registered.metadata.classes,
        "confidence_threshold": registered.metadata.confidence_threshold,
        "iou_threshold": registered.metadata.iou_threshold,
        "model_sha256": registered.model_sha256,
        "metadata_sha256": registered.metadata_sha256,
        "provider": registered.provider,
        "available_providers": registered.available_providers,
        "source_format": conversion.source_format if conversion else "onnx",
        "source_sha256": conversion.source_sha256 if conversion else registered.model_sha256,
        "conversion": conversion.model_dump(mode="json") if conversion else None,
        "supported_profiles": [
            profile.value for profile in registered.metadata.supported_profiles
        ],
    }


def source_response(source: dict[str, object]) -> dict[str, object]:
    metadata = source["metadata"]
    assert isinstance(metadata, dict)
    return {
        "id": source["id"],
        "kind": source["kind"],
        "name": source["name"],
        "sha256": source["sha256"],
        "created_at": source["created_at"],
        **metadata,
    }


def create_api_router(
    settings: Settings,
    model_registry: ModelRegistry,
    model_imports: ModelImportService,
    media_store: MediaStore,
    sessions: SessionService,
    previews: PreviewStore,
    exports: ExportService,
    monitor: MonitorPreviewService,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/capabilities")
    def capabilities() -> dict[str, object]:
        return {
            "available_providers": ort.get_available_providers(),
            "camera_backend": (
                "V4L2"
                if platform.system() == "Linux"
                else "Media Foundation / DirectShow"
                if platform.system() == "Windows"
                else "unavailable"
            ),
            "offline_runtime": True,
            "overlay_capabilities": overlay_capabilities(),
        }

    @router.get("/settings")
    def public_settings() -> dict[str, object]:
        return {
            "bind_host": settings.bind_host,
            "bind_port": settings.bind_port,
            "data_dir": str(settings.data_dir),
            "auth_required": not settings.is_loopback,
        }

    @router.get("/field-switch/config")
    def get_field_switch_config() -> FieldSwitchPreferences:
        return load_field_switch_preferences(sessions.database)

    @router.put("/field-switch/config")
    def put_field_switch_config(request: FieldSwitchPreferences) -> FieldSwitchPreferences:
        return save_field_switch_preferences(sessions.database, request)

    @router.get("/models")
    def list_models() -> list[dict[str, object]]:
        return [model_response(model) for model in model_registry.list()]

    @router.post("/models", status_code=201, response_model=None)
    def register_model(
        model: UploadFile,
        metadata_json: str = Form(),
    ) -> dict[str, object] | JSONResponse:
        try:
            metadata = json.loads(metadata_json)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            registered = model_registry.register(model.file, model.filename or "model", metadata)
            return model_response(registered)
        except (json.JSONDecodeError, ValueError, ModelRegistrationError) as exc:
            return error_response(422, "invalid_model", str(exc))

    @router.get("/model-imports")
    def list_model_imports() -> list[dict[str, object]]:
        return [asdict(record) for record in model_imports.list()]

    @router.get("/model-imports/{import_id}", response_model=None)
    def get_model_import(import_id: str) -> dict[str, object] | JSONResponse:
        try:
            return asdict(model_imports.get(import_id))
        except ModelImportError as exc:
            return error_response(404, "model_import_not_found", str(exc))

    @router.post("/model-imports", status_code=202, response_model=None)
    def inspect_model_import(
        checkpoint: UploadFile,
        trusted: bool = Form(default=False),
        configuration: UploadFile | None = None,
    ) -> dict[str, object] | JSONResponse:
        try:
            record = model_imports.inspect(
                checkpoint.file,
                checkpoint.filename or "checkpoint.pt",
                trusted=trusted,
                config_stream=configuration.file if configuration else None,
                config_filename=configuration.filename if configuration else None,
            )
            return asdict(record)
        except ModelImportError as exc:
            return error_response(422, "invalid_checkpoint_import", str(exc))

    @router.post(
        "/model-imports/{import_id}/convert", status_code=202, response_model=None
    )
    def convert_model_import(
        import_id: str, request: ModelConvertRequest
    ) -> dict[str, object] | JSONResponse:
        try:
            return asdict(
                model_imports.convert(
                    import_id,
                    model_name=request.model_name,
                    confidence_threshold=request.confidence_threshold,
                    iou_threshold=request.iou_threshold,
                )
            )
        except ModelImportError as exc:
            return error_response(409, "invalid_checkpoint_state", str(exc))

    @router.get("/sources")
    def list_sources() -> list[dict[str, object]]:
        return [source_response(source) for source in media_store.list()]

    @router.get("/capture-devices")
    def capture_devices() -> list[dict[str, object]]:
        return [asdict(device) for device in discover_capture_devices()]

    @router.get("/monitor/status")
    def monitor_status() -> dict[str, object]:
        return asdict(monitor.status())

    @router.post("/monitor/select", status_code=202, response_model=None)
    def monitor_select(request: MonitorSelectRequest) -> dict[str, object] | JSONResponse:
        try:
            return asdict(monitor.select(request.source_id))
        except SourceError as exc:
            return error_response(422, "source_unavailable", str(exc))

    @router.post("/monitor/stop", status_code=202)
    def monitor_stop() -> dict[str, object]:
        monitor.stop()
        return asdict(monitor.status())

    @router.get("/monitor/frame", response_model=None)
    def monitor_frame() -> Response | JSONResponse:
        try:
            value, status = monitor.jpeg_snapshot()
        except KeyError:
            return error_response(404, "preview_not_found", "No monitor frame is available yet.")
        except SourceError as exc:
            return error_response(409, "preview_failed", str(exc))
        return Response(
            value,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Index": str(status.frame_index),
                "X-Frame-Published-At": str(status.updated_at),
            },
        )

    @router.post("/sources/video", status_code=201, response_model=None)
    def import_video(video: UploadFile) -> dict[str, object] | JSONResponse:
        try:
            stored = media_store.import_video(video.file, video.filename or "video")
            return source_response(media_store.get_record(stored.id))
        except SourceError as exc:
            return error_response(422, "invalid_source", str(exc))

    @router.post("/sources/camera", status_code=201, response_model=None)
    def register_camera(request: CameraRequest) -> dict[str, object] | JSONResponse:
        try:
            return source_response(media_store.register_camera(request.device))
        except SourceError as exc:
            return error_response(422, "camera_unavailable", str(exc))

    @router.get("/sessions")
    def list_sessions() -> list[dict[str, object]]:
        return [asdict(session) for session in sessions.list()]

    @router.post("/sessions", status_code=201, response_model=None)
    def create_session(request: SessionRequest) -> dict[str, object] | JSONResponse:
        try:
            return asdict(
                sessions.create(
                    request.model_id,
                    request.source_id,
                    sample_every=request.sample_every,
                    inference_profile=request.inference_profile,
                    compute_device=request.compute_device,
                    record_data=request.record_data,
                    record_monitor=request.record_monitor,
                    preview_overlays=request.preview_overlays,
                    recording_overlays=request.recording_overlays,
                    enable_profiling=request.enable_profiling,
                )
            )
        except (KeyError, ValueError, ModelRegistrationError, SourceError) as exc:
            return error_response(422, "invalid_session", str(exc))

    @router.get("/sessions/{session_id}", response_model=None)
    def get_session(session_id: str) -> dict[str, object] | JSONResponse:
        try:
            return asdict(sessions.get(session_id))
        except KeyError:
            return error_response(404, "session_not_found", "Session was not found.")

    @router.post(
        "/sessions/{session_id}/start", status_code=202, response_model=None
    )
    def start_session(session_id: str) -> dict[str, object] | JSONResponse:
        try:
            monitor.release_for_inference(sessions.get(session_id).source_id)
            return asdict(sessions.start(session_id))
        except KeyError:
            return error_response(404, "session_not_found", "Session was not found.")
        except (ValueError, RuntimeError, SourceError) as exc:
            return error_response(409, "invalid_state", str(exc))

    @router.post(
        "/sessions/{session_id}/stop", status_code=202, response_model=None
    )
    def stop_session(session_id: str) -> dict[str, object] | JSONResponse:
        try:
            return asdict(sessions.stop(session_id))
        except KeyError:
            return error_response(404, "session_not_found", "Session was not found.")

    @router.get("/previews/{session_id}", response_model=None)
    def preview(session_id: str) -> Response | JSONResponse:
        try:
            current = previews.get(session_id)
        except KeyError:
            return error_response(404, "preview_not_found", "No preview is available yet.")
        return Response(
            current.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Index": str(current.frame_index),
                "X-Detection-Count": str(current.detection_count),
                "X-Frame-Published-At": str(current.published_at),
            },
        )

    @router.get("/sessions/{session_id}/export", response_model=None)
    def export_session(session_id: str) -> FileResponse | JSONResponse:
        try:
            archive = exports.build_zip(session_id)
            return FileResponse(
                archive,
                media_type="application/zip",
                filename=f"reefvision-edge-{session_id}.zip",
            )
        except ExportError as exc:
            return error_response(409, "export_unavailable", str(exc))

    return router
