from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from reefvision_edge import __version__
from reefvision_edge.auth import authentication_middleware
from reefvision_edge.config import Settings, require_safe_bind
from reefvision_edge.db import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    require_safe_bind(resolved)
    database = Database(resolved.database_path)
    database.initialize()
    database.recover_interrupted_sessions()

    app = FastAPI(title="ReefVision-Edge", version=__version__)
    app.state.settings = resolved
    app.state.database = database

    @app.middleware("http")
    async def authenticate(request: Request, call_next: object):  # type: ignore[no-untyped-def]
        return await authentication_middleware(request, call_next, resolved)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"product": "ReefVision-Edge", "version": __version__, "status": "ok"}

    @app.get("/api/previews/{session_id}")
    def missing_preview(session_id: str) -> None:
        raise HTTPException(status_code=404, detail=f"No preview for session {session_id}")

    @app.get("/")
    def root() -> dict[str, str]:
        return {"product": "ReefVision-Edge", "status": "dashboard not built"}

    return app
