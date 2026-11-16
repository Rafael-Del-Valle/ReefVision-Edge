from pathlib import Path

from importlib.metadata import PackageNotFoundError, distribution

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

for package in (
    "nvidia-cublas",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-nvjitlink",
):
    try:
        package_distribution = distribution(package)
    except PackageNotFoundError:
        continue
    for package_file in package_distribution.files or ():
        if package_file.suffix.lower() == ".dll":
            binaries.append(
                (
                    str(package_distribution.locate_file(package_file)),
                    package_file.parent.as_posix(),
                )
            )

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
    runtime_hooks=[str(SPEC_DIR / "runtime_hook_nvidia.py")],
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
