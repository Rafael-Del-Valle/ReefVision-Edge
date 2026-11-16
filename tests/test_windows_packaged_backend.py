from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGED_EXE = (
    ROOT
    / ".artifacts"
    / "windows-msi"
    / "backend"
    / "reefvision-edge-backend"
    / "reefvision-edge-backend.exe"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("REEFVISION_EDGE_RUN_PACKAGED_TESTS") != "1",
    reason="set REEFVISION_EDGE_RUN_PACKAGED_TESTS=1 after building the Windows payload",
)


def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_packaged_backend_runs_without_source_checkout(tmp_path: Path) -> None:
    assert PACKAGED_EXE.is_file()
    version = subprocess.run(
        [str(PACKAGED_EXE), "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == "reefvision-edge 0.1.0"

    source = ROOT / "src"
    disabled_source = ROOT / ".artifacts" / "windows-msi" / "source-disabled"
    assert source.is_dir()
    assert not disabled_source.exists()
    source.rename(disabled_source)

    port = _free_loopback_port()
    data_dir = tmp_path / "data"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    process = subprocess.Popen(
        [
            str(PACKAGED_EXE),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 30
        health_payload: dict[str, str] | None = None
        dashboard = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"packaged backend exited early with code {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout=1
                ) as response:
                    health_payload = json.load(response)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                    dashboard = response.read().decode("utf-8")
                break
            except OSError:
                time.sleep(0.1)

        assert health_payload is not None
        assert health_payload["product"] == "ReefVision-Edge"
        assert "ReefVision-Edge" in dashboard
        assert data_dir.joinpath("reefvision-edge.sqlite3").is_file()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        disabled_source.rename(source)
