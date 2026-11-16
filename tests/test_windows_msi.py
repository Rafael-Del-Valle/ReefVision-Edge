from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


MSI = Path.home() / "Downloads" / "ReefVision-Edge-0.1.0-windows-x64.msi"
EDGE_UPGRADE_CODE = "{C2F68AF2-90BF-4CD5-982C-42987B44F5E0}"
REEFVISION_UPGRADE_CODE = "{5EDA8EE0-2A5D-5B08-A8DF-9BECFC3A4351}"

pytestmark = pytest.mark.skipif(
    os.environ.get("REEFVISION_EDGE_RUN_MSI_TESTS") != "1",
    reason="set REEFVISION_EDGE_RUN_MSI_TESTS=1 after building the Windows installer",
)


def _msi_tables() -> dict[str, object]:
    script = r"""
$ErrorActionPreference = 'Stop'
$installer = New-Object -ComObject WindowsInstaller.Installer
$database = $installer.OpenDatabase($env:REEFVISION_EDGE_TEST_MSI, 0)
function Scalar([string]$Sql) {
    $view = $database.OpenView($Sql)
    [void]$view.Execute()
    $record = $view.Fetch()
    $value = if ($null -eq $record) { $null } else { $record.StringData(1) }
    [void]$view.Close()
    return $value
}
function Rows([string]$Sql, [int]$Columns) {
    $view = $database.OpenView($Sql)
    [void]$view.Execute()
    $rows = @()
    while ($record = $view.Fetch()) {
        $row = @()
        for ($index = 1; $index -le $Columns; $index += 1) {
            $row += $record.StringData($index)
        }
        $rows += ,$row
    }
    [void]$view.Close()
    return $rows
}
$result = [ordered]@{
    productName = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='ProductName'"
    productVersion = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='ProductVersion'"
    manufacturer = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='Manufacturer'"
    upgradeCode = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='UpgradeCode'"
    allUsers = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='ALLUSERS'"
    arpIcon = Scalar "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='ARPPRODUCTICON'"
    installDirectory = Rows "SELECT ``Directory_Parent``, ``DefaultDir`` FROM ``Directory`` WHERE ``Directory``='INSTALLFOLDER'" 2
    programsDirectory = Rows "SELECT ``Directory_Parent``, ``DefaultDir`` FROM ``Directory`` WHERE ``Directory``='EdgeProgramsFolder'" 2
    shortcuts = Rows "SELECT ``Directory_``, ``Name``, ``Target`` FROM ``Shortcut``" 3
}
$result | ConvertTo-Json -Compress -Depth 5
"""
    environment = os.environ.copy()
    environment["REEFVISION_EDGE_TEST_MSI"] = str(MSI)
    completed = subprocess.run(
        ["pwsh.exe", "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_msi_has_distinct_per_user_product_and_start_shortcut() -> None:
    assert MSI.is_file()
    tables = _msi_tables()

    assert tables["productName"] == "ReefVision-Edge"
    assert tables["productVersion"] == "0.1.0"
    assert tables["manufacturer"] == "ReefVision"
    assert tables["upgradeCode"] == EDGE_UPGRADE_CODE
    assert tables["upgradeCode"] != REEFVISION_UPGRADE_CODE
    assert tables["allUsers"] is None
    assert tables["arpIcon"] == "ReefVisionEdgeIcon"
    assert tables["installDirectory"][0] == "EdgeProgramsFolder"
    assert tables["installDirectory"][1].split("|")[-1] == "ReefVision-Edge"
    assert tables["programsDirectory"] == ["LocalAppDataFolder", "Programs"]
    assert tables["shortcuts"][0] == "ReefVisionEdgeProgramMenu"
    assert tables["shortcuts"][1].split("|")[-1] == "ReefVision-Edge"
    assert tables["shortcuts"][2] == "[#ReefVisionEdgeLauncherExe]"
