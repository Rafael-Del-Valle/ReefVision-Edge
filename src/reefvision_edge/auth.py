from __future__ import annotations

import base64
import binascii
import secrets

from starlette.requests import Request
from starlette.responses import Response

from reefvision_edge.config import Settings


def _unauthorized() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="ReefVision-Edge"'},
    )


def credentials_match(authorization: str | None, settings: Settings) -> bool:
    if not authorization or not authorization.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
        username, password = raw.split(":", 1)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    expected_username = settings.auth_username or ""
    expected_password = settings.auth_password or ""
    return secrets.compare_digest(username, expected_username) and secrets.compare_digest(
        password, expected_password
    )


async def authentication_middleware(
    request: Request,
    call_next: object,
    settings: Settings,
) -> Response:
    if not settings.is_loopback and not credentials_match(
        request.headers.get("Authorization"), settings
    ):
        return _unauthorized()
    return await call_next(request)  # type: ignore[operator,no-any-return]
