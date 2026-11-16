from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []


def _activate_bundled_nvidia_dlls() -> None:
    if os.name != "nt":
        return
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    directories = sorted(path for path in root.rglob("bin") if "nvidia" in path.parts)
    for directory in directories:
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


_activate_bundled_nvidia_dlls()
