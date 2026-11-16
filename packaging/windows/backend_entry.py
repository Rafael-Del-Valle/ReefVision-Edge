from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import onnxruntime as ort
import uvicorn

from reefvision_edge.app import create_app
from reefvision_edge.cli import settings_from_argv
from reefvision_edge.compute import preload_nvidia_runtime


def runtime_diagnostics() -> dict[str, object]:
    preload_status = "unavailable"
    preload_error: str | None = None
    try:
        preload_nvidia_runtime()
        if "CUDAExecutionProvider" in ort.get_available_providers():
            preload_status = "loaded"
    except Exception as exc:
        preload_status = "failed"
        preload_error = str(exc)
    return {
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "nvidia_runtime_preload": preload_status,
        "preload_error": preload_error,
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def main() -> None:
    if sys.argv[1:] == ["--runtime-diagnostics"]:
        print(json.dumps(runtime_diagnostics(), sort_keys=True))
        return
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
