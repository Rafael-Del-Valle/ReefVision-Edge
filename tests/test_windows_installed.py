from __future__ import annotations

import os
import winreg
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("REEFVISION_EDGE_RUN_INSTALLED_TESTS") != "1",
    reason="set REEFVISION_EDGE_RUN_INSTALLED_TESTS=1 after installing the MSI",
)


def _installed_app_values() -> dict[str, object]:
    uninstall = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        with winreg.OpenKey(hive, uninstall) as root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                name = winreg.EnumKey(root, index)
                with winreg.OpenKey(root, name) as product:
                    try:
                        display_name = winreg.QueryValueEx(product, "DisplayName")[0]
                    except FileNotFoundError:
                        continue
                    if display_name != "ReefVision-Edge":
                        continue
                    values: dict[str, object] = {"ProductCode": name}
                    for value_name in (
                        "DisplayName",
                        "DisplayVersion",
                        "Publisher",
                        "InstallLocation",
                    ):
                        try:
                            values[value_name] = winreg.QueryValueEx(product, value_name)[0]
                        except FileNotFoundError:
                            values[value_name] = None
                    return values
    raise AssertionError("ReefVision-Edge is not registered in Windows Installed Apps")


def test_installed_app_is_independent_and_complete() -> None:
    values = _installed_app_values()
    local_app_data = Path(os.environ["LOCALAPPDATA"])
    install_dir = local_app_data / "Programs" / "ReefVision-Edge"
    data_dir = local_app_data / "ReefVision-Edge"
    start_menu = (
        Path(os.environ["APPDATA"])
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "ReefVision-Edge"
        / "ReefVision-Edge.lnk"
    )

    assert values["DisplayVersion"] == "0.1.0"
    assert values["Publisher"] == "ReefVision"
    assert Path(str(values["InstallLocation"])).resolve() == install_dir.resolve()
    assert install_dir.joinpath("ReefVision-Edge.exe").is_file()
    assert install_dir.joinpath(
        "backend", "reefvision-edge-backend", "reefvision-edge-backend.exe"
    ).is_file()
    assert install_dir.joinpath(
        "backend",
        "reefvision-edge-backend",
        "_internal",
        "reefvision_edge",
        "static",
        "index.html",
    ).is_file()
    assert start_menu.is_file()
    assert data_dir != local_app_data / "ReefVision"
    assert data_dir != install_dir
