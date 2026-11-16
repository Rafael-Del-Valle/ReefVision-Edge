from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import onnxruntime as ort
from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from reefvision_edge.config import Settings
from reefvision_edge.domain import RegisteredModel
from reefvision_edge.exports import ExportError, ExportService
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistrationError, ModelRegistry
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sessions import SessionService
from reefvision_edge.sources import SourceError


class CameraRequest(BaseModel):
    device: str = Field(min_length=1)


class SessionRequest(BaseModel):
    model_id: str
    source_id: str
    sample_every: int = Field(default=1, ge=1, le=10000)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def model_response(registered: RegisteredModel) -> dict[str, object]:
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
    media_store: MediaStore,
    sessions: SessionService,
    previews: PreviewStore,
    exports: ExportService,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/capabilities")
    def capabilities() -> dict[str, object]:
        return {
            "available_providers": ort.get_available_providers(),
            "camera_backend": "V4L2" if platform.system() == "Linux" else "unavailable",
            "offline_runtime": True,
        }

    @router.get("/settings")
    def public_settings() -> dict[str, object]:
        return {
            "bind_host": settings.bind_host,
            "bind_port": settings.bind_port,
            "data_dir": str(settings.data_dir),
            "auth_required": not settings.is_loopback,
        }

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

    @router.get("/sources")
    def list_sources() -> list[dict[str, object]]:
        return [source_response(source) for source in media_store.list()]

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
            return source_response(media_store.register_camera(Path(request.device)))
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
            return asdict(sessions.start(session_id))
        except KeyError:
            return error_response(404, "session_not_found", "Session was not found.")
        except (ValueError, RuntimeError) as exc:
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
