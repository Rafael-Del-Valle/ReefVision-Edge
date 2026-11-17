from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


SPEC_DIR = Path(SPECPATH)
ROOT = SPEC_DIR.parents[1]
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports = collect_submodules("ultralytics")

for package in ("ultralytics", "torch", "torchvision", "onnx", "cv2", "yaml"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

analysis = Analysis(
    [str(ROOT / "src" / "reefvision_edge" / "converter_worker.py")],
    pathex=[str(ROOT / "src")],
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
    name="reefvision-edge-model-converter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="reefvision-edge-model-converter",
)
