[CmdletBinding()]
param(
    [switch]$BackendOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$verifiedCommit = "bdc9d8795961c71cf55af244addd2c18281f2ec8"
$bundleRoot = Join-Path $repoRoot ".artifacts\reefvision-edge-offline-v0.1-bdc9d87"
$workRoot = Join-Path $repoRoot ".artifacts\windows-msi"
$buildVenv = Join-Path $workRoot "build-venv"
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$backendDist = Join-Path $workRoot "backend"
$backendWork = Join-Path $workRoot "pyinstaller-build"
$spec = Join-Path $repoRoot "packaging\windows\reefvision-edge-backend.spec"

Push-Location $repoRoot
try {
    & git merge-base --is-ancestor $verifiedCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "HEAD does not contain verified application commit $verifiedCommit."
    }

    $checksums = Join-Path $bundleRoot "SHA256SUMS.txt"
    if (-not (Test-Path -LiteralPath $checksums)) {
        throw "Verified offline bundle is missing: $bundleRoot"
    }
    foreach ($line in Get-Content -LiteralPath $checksums) {
        if (-not $line.Trim()) { continue }
        $match = [regex]::Match($line, "^([0-9a-fA-F]{64})\s{2}(.+)$")
        if (-not $match.Success) {
            throw "Malformed checksum line: $line"
        }
        $relative = $match.Groups[2].Value.Replace("/", "\")
        $path = Join-Path $bundleRoot $relative
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $match.Groups[1].Value.ToLowerInvariant()) {
            throw "Checksum mismatch for $relative"
        }
    }

    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $buildPython)) {
        & (Join-Path $repoRoot ".venv\Scripts\python.exe") -m venv $buildVenv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows packaging environment." }
    }

    & $buildPython -m pip install --disable-pip-version-check "pyinstaller==6.16.0"
    if ($LASTEXITCODE -ne 0) { throw "Could not install pinned PyInstaller." }
    & $buildPython -m pip install --disable-pip-version-check --no-index `
        --find-links (Join-Path $bundleRoot "app") `
        --find-links (Join-Path $bundleRoot "wheels") `
        "reefvision-edge==0.1.0"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the verified offline application bundle." }

    & $buildPython -m PyInstaller --noconfirm --clean `
        --distpath $backendDist --workpath $backendWork $spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller backend build failed." }

    $backendExe = Join-Path $backendDist "reefvision-edge-backend\reefvision-edge-backend.exe"
    if (-not (Test-Path -LiteralPath $backendExe)) {
        throw "Expected packaged backend was not produced: $backendExe"
    }
    Write-Host "Packaged backend: $backendExe"

    if ($BackendOnly) {
        return
    }

    throw "MSI assembly is added by the next packaging task. Use -BackendOnly for now."
}
finally {
    Pop-Location
}
