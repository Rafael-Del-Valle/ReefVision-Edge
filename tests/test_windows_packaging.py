from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"
VERIFIED_BUNDLE = ROOT / ".artifacts" / "reefvision-edge-offline-v0.1-bdc9d87"


LAUNCHER = WINDOWS / "launcher"


def test_launcher_has_distinct_release_identity_and_no_dev_server() -> None:
    config = json.loads((LAUNCHER / "tauri.conf.json").read_text(encoding="utf-8"))

    assert config["productName"] == "ReefVision-Edge"
    assert config["version"] == "0.1.0"
    assert config["identifier"] == "org.reefvision.edge.desktop"
    assert config["identifier"] != "org.reefvision.desktop"
    assert "devUrl" not in config["build"]
    assert config["build"]["frontendDist"] == "../launcher-dist"


def test_backend_entry_executes_the_verified_application_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(WINDOWS / "backend_entry.py"), "--version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "reefvision-edge 0.1.0"


def test_verified_application_wheel_contains_dashboard_assets() -> None:
    wheels = list((VERIFIED_BUNDLE / "app").glob("reefvision_edge-0.1.0-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert "reefvision_edge/static/index.html" in names
    assert any(
        name.startswith("reefvision_edge/static/assets/")
        and name.endswith(".js")
        for name in names
    )
    assert any(
        name.startswith("reefvision_edge/static/assets/") and name.endswith(".css")
        for name in names
    )
