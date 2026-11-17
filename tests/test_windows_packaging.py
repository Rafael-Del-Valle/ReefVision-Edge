from __future__ import annotations

import json
import re
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

    rejected = subprocess.run(
        [
            sys.executable,
            str(WINDOWS / "backend_entry.py"),
            "--host",
            "0.0.0.0",
            "--port",
            "8766",
            "--username",
            "field",
            "--password",
            "secret",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert rejected.returncode != 0
    assert "only accepts --host 127.0.0.1" in rejected.stderr

    diagnostics = subprocess.run(
        [sys.executable, str(WINDOWS / "backend_entry.py"), "--runtime-diagnostics"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert diagnostics.returncode == 0, diagnostics.stderr
    payload = json.loads(diagnostics.stdout)
    assert "CPUExecutionProvider" in payload["available_providers"]
    assert payload["onnxruntime_version"]
    assert payload["nvidia_runtime_preload"] in {"loaded", "unavailable", "failed"}


def test_runtime_bundle_profiles_resolve_exact_hashed_dependencies() -> None:
    script = ROOT / "scripts" / "build_windows_runtime_bundle.ps1"
    for profile, expected_runtime in (
        ("cpu", "onnxruntime==1.32.0"),
        ("cuda", "onnxruntime-gpu==1.32.0"),
    ):
        completed = subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(script),
                "-Profile",
                profile,
                "-ValidateOnly",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["profile"] == profile
        assert expected_runtime in payload["requirements"]
        assert all("==" in requirement for requirement in payload["requirements"])
        assert payload["all_requirements_hashed"] is True
        assert len(payload["lock_sha256"]) == 64

    cuda = json.loads(
        subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(script),
                "-Profile",
                "cuda",
                "-ValidateOnly",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert "nvidia-cudnn-cu13==9.28.1.4" in cuda["requirements"]
    assert "nvidia-cuda-runtime==13.4.92" in cuda["requirements"]


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


def test_msi_build_requires_a_pinned_webview2_input() -> None:
    script = (ROOT / "scripts" / "build_windows_msi.ps1").read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in script
    assert "$WebView2Installer" in script
    assert "D3CBE8FEDB104282D745FFF106993EFEAD51AE94F5595A4E75804446A974F9C3" in script
    assert ".codex\\worktrees" not in script


def test_msi_build_reinstalls_the_application_wheel_from_the_selected_bundle() -> None:
    script = (ROOT / "scripts" / "build_windows_msi.ps1").read_text(encoding="utf-8")

    assert "--force-reinstall" in script
    assert '"reefvision-edge==0.1.0"' in script


def test_frozen_backend_collects_and_activates_nested_nvidia_dll_directories() -> None:
    spec = (WINDOWS / "reefvision-edge-backend.spec").read_text(encoding="utf-8")
    runtime_hook = (WINDOWS / "runtime_hook_nvidia.py").read_text(encoding="utf-8")

    assert '"nvidia-cublas"' in spec
    assert '"nvidia-cuda-runtime"' in spec
    assert '"nvidia-cudnn-cu13"' in spec
    assert "distribution(package)" in spec
    assert 'suffix.lower() == ".dll"' in spec
    assert 'rglob("bin")' in runtime_hook


def test_frozen_backend_embeds_and_reports_bundle_commit_identity() -> None:
    build_script = (ROOT / "scripts" / "build_windows_msi.ps1").read_text(
        encoding="utf-8"
    )
    backend_entry = (WINDOWS / "backend_entry.py").read_text(encoding="utf-8")

    assert "REEFVISION_EDGE_BUILD_COMMIT" in build_script
    assert '"build_commit"' in backend_entry
    assert "BUNDLE-METADATA.json" in build_script


def test_installed_acceptance_requires_new_uploads_and_uninstall_isolation() -> None:
    acceptance = (ROOT / "scripts" / "acceptance_windows_installed.ps1").read_text(
        encoding="utf-8"
    )
    uninstall = (
        ROOT / "scripts" / "acceptance_windows_uninstall_isolation.ps1"
    ).read_text(encoding="utf-8")

    assert "$existingModelIds" in acceptance
    assert "$existingSourceIds" in acceptance
    assert "model_sha256" in acceptance
    assert "catch {" not in acceptance
    assert "msiexec.exe" in uninstall
    assert "CodexMsiIsolationSentinel" in uninstall
    assert "ReefVision-Edge.exe" in uninstall


def test_model_converter_lock_is_exact_hashed_windows_python_312_closure() -> None:
    lock = WINDOWS / "model-converter.lock"
    lines = [
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert "ultralytics==8.4.215" in "\n".join(lines)
    assert "torch==2.15.0" in "\n".join(lines)
    assert "torchvision==0.29.2" in "\n".join(lines)
    assert "onnx==1.23.2" in "\n".join(lines)
    assert len(lines) >= 40
    assert all(
        re.fullmatch(r"[a-zA-Z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}", line)
        for line in lines
    )


def test_converter_license_review_covers_every_locked_distribution() -> None:
    lock_names = {
        line.split("==", 1)[0].casefold().replace("_", "-")
        for line in (WINDOWS / "model-converter.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }
    review = json.loads(
        (WINDOWS / "model-converter-licenses.json").read_text(encoding="utf-8")
    )
    reviewed = {
        item["name"].casefold().replace("_", "-") for item in review["packages"]
    }

    assert reviewed == lock_names
    assert all(item["redistribution_reviewed"] is True for item in review["packages"])
    assert all(item["license"] for item in review["packages"])
    assert review["project_license"] == "AGPL-3.0-only"
    assert review["ultralytics_license"] == "AGPL-3.0-only"
    fallback = WINDOWS / review["packages_by_name"]["nvidia-ml-py"]["license_file"]
    assert fallback.is_file()
    assert "Redistribution and use" in fallback.read_text(encoding="utf-8")


def test_converter_bundle_and_msi_fail_closed_and_exclude_research_files() -> None:
    converter = (ROOT / "scripts" / "build_windows_converter_bundle.ps1").read_text(
        encoding="utf-8"
    )
    offline = (ROOT / "scripts" / "build_offline_bundle.ps1").read_text(
        encoding="utf-8"
    )
    msi = (ROOT / "scripts" / "build_windows_msi.ps1").read_text(encoding="utf-8")
    spec = (WINDOWS / "reefvision-edge-model-converter.spec").read_text(
        encoding="utf-8"
    )

    assert "--require-hashes" in converter
    assert "model-converter-licenses.json" in converter
    assert "converter-wheels" in offline
    assert "reefvision-edge-model-converter.exe" in msi
    assert "converter-licenses" in msi
    assert "converter_worker.py" in spec
    assert "collect_delvewheel_libs_directory" in spec
    assert '"onnx", datas=datas, binaries=binaries' in spec
    assert "ULTRALYTICS_OFFLINE" in msi
    assert all(extension in offline for extension in ("*.pt", "*.pth", "*.onnx"))


def test_offline_bundle_cleans_only_generated_python_build_outputs() -> None:
    offline = (ROOT / "scripts" / "build_offline_bundle.ps1").read_text(
        encoding="utf-8"
    )

    assert 'Join-Path $repoRoot "build"' in offline
    assert 'Join-Path $repoRoot "src\\reefvision_edge.egg-info"' in offline
    assert "Remove-Item -LiteralPath $generatedPath -Recurse -Force" in offline
    assert offline.index("Remove-Item -LiteralPath $generatedPath -Recurse -Force") < (
        offline.index("-m build --wheel")
    )
