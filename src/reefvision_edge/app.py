from __future__ import annotations

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
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sessions import SessionRunner, SessionService


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    require_safe_bind(resolved)
    database = Database(resolved.database_path)
    database.initialize()
    database.recover_interrupted_sessions()
    model_registry = ModelRegistry(database, resolved.models_dir)
    media_store = MediaStore(database, resolved.media_dir)
    previews = PreviewStore()
    runner = SessionRunner(database, model_registry, media_store, previews)
    sessions = SessionService(database, runner)
    exports = ExportService(database, resolved.exports_dir)

    app = FastAPI(title="ReefVision-Edge", version=__version__)
    app.state.settings = resolved
    app.state.database = database
    app.state.model_registry = model_registry
    app.state.media_store = media_store
    app.state.previews = previews
    app.state.runner = runner
    app.state.sessions = sessions
    app.state.exports = exports

    @app.middleware("http")
    async def authenticate(request: Request, call_next: object):  # type: ignore[no-untyped-def]
        return await authentication_middleware(request, call_next, resolved)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"product": "ReefVision-Edge", "version": __version__, "status": "ok"}

    app.include_router(
        create_api_router(
            resolved, model_registry, media_store, sessions, previews, exports
        )
    )

    dashboard = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dashboard.is_dir():
        app.mount("/", StaticFiles(directory=dashboard, html=True), name="dashboard")
    else:
        @app.get("/")
        def root() -> dict[str, str]:
            return {"product": "ReefVision-Edge", "status": "dashboard not built"}

    return app
