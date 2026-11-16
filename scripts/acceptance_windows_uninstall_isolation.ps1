[CmdletBinding()]
param(
    [string]$MsiPath = (Join-Path $env:USERPROFILE "Downloads\ReefVision-Edge-0.1.0-windows-x64.msi")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-That([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-EdgeRegistration {
    $roots = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        foreach ($key in Get-ChildItem -LiteralPath $root) {
            $values = Get-ItemProperty -LiteralPath $key.PSPath
            if ($values.DisplayName -eq "ReefVision-Edge") {
                return [pscustomobject]@{
                    ProductCode = $key.PSChildName
                    RegistryPath = $key.PSPath
                }
            }
        }
    }
    return $null
}

function Invoke-Msi([string[]]$Arguments, [string]$Action) {
    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList $Arguments `
        -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) {
        throw "$Action failed with Windows Installer exit code $($process.ExitCode)."
    }
}

$resolvedMsi = (Resolve-Path -LiteralPath $MsiPath).Path
$runToken = [Guid]::NewGuid().ToString("N")
$referenceDirectory = Join-Path $env:LOCALAPPDATA "Programs\ReefVision"
$referenceDirectoryExisted = Test-Path -LiteralPath $referenceDirectory
$referenceSentinel = Join-Path $referenceDirectory "CodexMsiIsolationSentinel-$runToken.txt"
$referenceRegistry = "HKCU:\Software\ReefVision\CodexMsiIsolationSentinel-$runToken"
$edgeDataDirectory = Join-Path $env:LOCALAPPDATA "ReefVision-Edge"
$edgeDataSentinel = Join-Path $edgeDataDirectory "uninstall-isolation-$runToken.txt"
$edgeInstall = Join-Path $env:LOCALAPPDATA "Programs\ReefVision-Edge\ReefVision-Edge.exe"
$edgeShortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\ReefVision-Edge\ReefVision-Edge.lnk"

$initialRegistration = Get-EdgeRegistration
Assert-That ($null -ne $initialRegistration) "ReefVision-Edge must be installed before the isolation cycle."

New-Item -ItemType Directory -Path $referenceDirectory -Force | Out-Null
Set-Content -LiteralPath $referenceSentinel -Value $runToken -Encoding utf8
New-Item -ItemType Directory -Path $referenceRegistry -Force | Out-Null
New-ItemProperty -LiteralPath $referenceRegistry -Name "Value" -Value $runToken `
    -PropertyType String -Force | Out-Null
New-Item -ItemType Directory -Path $edgeDataDirectory -Force | Out-Null
Set-Content -LiteralPath $edgeDataSentinel -Value $runToken -Encoding utf8

$uninstallVerified = $false
try {
    $launcher = Get-Process | Where-Object { $_.Path -eq $edgeInstall }
    if ($launcher) {
        [void]$launcher.CloseMainWindow()
        $launcher | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue
    }
    $backendPath = Join-Path $env:LOCALAPPDATA "Programs\ReefVision-Edge\backend\reefvision-edge-backend\reefvision-edge-backend.exe"
    Get-Process | Where-Object { $_.Path -eq $backendPath } | Stop-Process -Force

    Invoke-Msi @("/x", $initialRegistration.ProductCode, "/qn", "/norestart") "Uninstall"

    Assert-That (-not (Test-Path -LiteralPath $edgeInstall)) "Edge launcher remained after uninstall."
    Assert-That (-not (Test-Path -LiteralPath $edgeShortcut)) "Edge shortcut remained after uninstall."
    Assert-That ($null -eq (Get-EdgeRegistration)) "Edge remained in Installed Apps after uninstall."
    Assert-That (Test-Path -LiteralPath $referenceSentinel) "ReefVision file sentinel was removed."
    Assert-That (Test-Path -LiteralPath $referenceRegistry) "ReefVision registry sentinel was removed."
    Assert-That (Test-Path -LiteralPath $edgeDataSentinel) "Per-user Edge data was removed."
    $uninstallVerified = $true
}
finally {
    if ($null -eq (Get-EdgeRegistration)) {
        Invoke-Msi @("/i", $resolvedMsi, "/qn", "/norestart") "Reinstall"
    }
    Remove-Item -LiteralPath $referenceSentinel -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $referenceRegistry -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $edgeDataSentinel -Force -ErrorAction SilentlyContinue
    if (-not $referenceDirectoryExisted) {
        $remaining = @(Get-ChildItem -LiteralPath $referenceDirectory -Force -ErrorAction SilentlyContinue)
        if ($remaining.Count -eq 0) {
            Remove-Item -LiteralPath $referenceDirectory -Force
        }
    }
}

$finalRegistration = Get-EdgeRegistration
Assert-That $uninstallVerified "The uninstall isolation assertions did not complete."
Assert-That ($null -ne $finalRegistration) "ReefVision-Edge was not reinstalled."
Assert-That (Test-Path -LiteralPath $edgeInstall) "Reinstalled Edge launcher is missing."
Assert-That (Test-Path -LiteralPath $edgeShortcut) "Reinstalled Edge shortcut is missing."

[ordered]@{
    product = "ReefVision-Edge"
    uninstall_isolation = "passed"
    reefvision_file_sentinel_preserved = $true
    reefvision_registry_sentinel_preserved = $true
    edge_data_preserved = $true
    reinstalled_product_code = $finalRegistration.ProductCode
} | ConvertTo-Json -Compress
