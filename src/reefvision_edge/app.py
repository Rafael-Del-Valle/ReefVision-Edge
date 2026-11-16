from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from reefvision_edge import __version__
from reefvision_edge.api import create_api_router
from reefvision_edge.auth import authentication_middleware
from reefvision_edge.config import Settings, require_safe_bind
from reefvision_edge.db import Database
from reefvision_edge.exports import ExportService
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistry
from reefvision_edge.monitor import MonitorPreviewService
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sessions import SessionRunner, SessionService


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    require_safe_bind(resolved)
    database = Database(resolved.database_path)
    database.initialize()
    database.recover_interrupted_sessions()
    model_registry = ModelRegistry(
        database,
        resolved.models_dir,
        max_model_bytes=resolved.max_model_upload_bytes,
    )
    media_store = MediaStore(
        database,
        resolved.media_dir,
        max_video_bytes=resolved.max_video_upload_bytes,
    )
    previews = PreviewStore(resolved.previews_dir)
    runner = SessionRunner(database, model_registry, media_store, previews)
    sessions = SessionService(database, runner)
    exports = ExportService(database, resolved.exports_dir)
    monitor = MonitorPreviewService(media_store)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            monitor.shutdown()
            sessions.shutdown(timeout=5)

    app = FastAPI(title="ReefVision-Edge", version=__version__, lifespan=lifespan)
    app.state.settings = resolved
    app.state.database = database
    app.state.model_registry = model_registry
    app.state.media_store = media_store
    app.state.previews = previews
    app.state.runner = runner
    app.state.sessions = sessions
    app.state.exports = exports
    app.state.monitor = monitor

    @app.middleware("http")
    async def authenticate(request: Request, call_next: object):  # type: ignore[no-untyped-def]
        return await authentication_middleware(request, call_next, resolved)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"product": "ReefVision-Edge", "version": __version__, "status": "ok"}

    app.include_router(
        create_api_router(
            resolved,
            model_registry,
            media_store,
            sessions,
            previews,
            exports,
            monitor,
        )
    )

    packaged_dashboard = Path(__file__).resolve().parent / "static"
    development_dashboard = Path(__file__).resolve().parents[2] / "web" / "dist"
    dashboard = (
        packaged_dashboard if packaged_dashboard.is_dir() else development_dashboard
    )
    if dashboard.is_dir():
        app.mount("/", StaticFiles(directory=dashboard, html=True), name="dashboard")
    else:
        @app.get("/")
        def root() -> dict[str, str]:
            return {"product": "ReefVision-Edge", "status": "dashboard not built"}

    return app
