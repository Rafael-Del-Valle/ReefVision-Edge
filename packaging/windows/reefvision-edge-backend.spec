from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


SPEC_DIR = Path(SPECPATH)
ROOT = SPEC_DIR.parents[1]

datas = collect_data_files("reefvision_edge")
binaries: list[tuple[str, str]] = []
hiddenimports = collect_submodules("uvicorn")

for package in ("onnxruntime", "cv2"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

datas.extend(
    [
        (str(ROOT / "LICENSE"), "licenses"),
        (str(ROOT / "THIRD_PARTY_NOTICES.md"), "licenses"),
    ]
)

analysis = Analysis(
    [str(SPEC_DIR / "backend_entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="reefvision-edge-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="reefvision-edge-backend",
)
