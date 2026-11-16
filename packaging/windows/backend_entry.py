from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from reefvision_edge.app import create_app
from reefvision_edge.cli import settings_from_argv


def main() -> None:
    settings = settings_from_argv()
    if settings.bind_host != "127.0.0.1":
        raise SystemExit("The packaged desktop backend only accepts --host 127.0.0.1.")
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.bind_port,
        log_level="info",
    )

if __name__ == "__main__":
    log_file = os.environ.get("REEFVISION_EDGE_LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        stream = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = stream
        sys.stderr = stream
    main()
