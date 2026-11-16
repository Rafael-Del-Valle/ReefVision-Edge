[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [Parameter(Mandatory = $true)]
    [string]$WebView2Installer,
    [string]$BundleRoot = ".artifacts\reefvision-edge-offline-windows-cuda-v0.1"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$verifiedCommit = "bdc9d8795961c71cf55af244addd2c18281f2ec8"
$bundleRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BundleRoot))
$workRoot = Join-Path $repoRoot ".artifacts\windows-msi"
$buildVenv = Join-Path $workRoot "build-venv"
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$backendDist = Join-Path $workRoot "backend"
$backendWork = Join-Path $workRoot "pyinstaller-build"
$spec = Join-Path $repoRoot "packaging\windows\reefvision-edge-backend.spec"
$stageRoot = Join-Path $workRoot "stage"
$wixObjectRoot = Join-Path $workRoot "wix-obj"
$wixTools = Join-Path $env:LOCALAPPDATA "tauri\WixTools314"
$wixSource = Join-Path $repoRoot "packaging\windows\wix\main.wxs"
$wixBackendSource = Join-Path $workRoot "backend-payload.wxs"
$launcherTarget = Join-Path $repoRoot "packaging\windows\launcher\target\release\reefvision-edge-launcher.exe"
$launcherStage = Join-Path $stageRoot "ReefVision-Edge.exe"
$expectedWebView2Sha256 = "D3CBE8FEDB104282D745FFF106993EFEAD51AE94F5595A4E75804446A974F9C3"
$downloadsMsi = Join-Path $env:USERPROFILE "Downloads\ReefVision-Edge-0.1.0-windows-x64.msi"

function Clear-BuildDirectory([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullWorkRoot = [System.IO.Path]::GetFullPath($workRoot)
    if (-not $fullPath.StartsWith($fullWorkRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clear a directory outside the Windows build root: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
}

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
    $bundleMetadataPath = Join-Path $bundleRoot "BUNDLE-METADATA.json"
    $runtimeLock = Join-Path $bundleRoot "runtime.lock"
    if (-not (Test-Path -LiteralPath $bundleMetadataPath)) {
        throw "Bundle metadata is missing: $bundleMetadataPath"
    }
    if (-not (Test-Path -LiteralPath $runtimeLock)) {
        throw "Runtime lock is missing: $runtimeLock"
    }
    $bundleMetadata = Get-Content -Raw -LiteralPath $bundleMetadataPath | ConvertFrom-Json
    $headCommit = (& git rev-parse HEAD).Trim()
    if ($bundleMetadata.commit -ne $headCommit) {
        throw "Offline bundle commit $($bundleMetadata.commit) does not match HEAD $headCommit."
    }

    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $buildPython)) {
        & (Join-Path $repoRoot ".venv\Scripts\python.exe") -m venv $buildVenv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows packaging environment." }
    }

    & $buildPython -m pip install --disable-pip-version-check --no-index `
        --find-links (Join-Path $bundleRoot "wheels") "pyinstaller==6.16.0"
    if ($LASTEXITCODE -ne 0) { throw "Could not install pinned PyInstaller." }
    & $buildPython -m pip install --disable-pip-version-check --no-index --force-reinstall `
        --find-links (Join-Path $bundleRoot "app") `
        --find-links (Join-Path $bundleRoot "wheels") `
        "reefvision-edge==0.1.0"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the verified offline application bundle." }
    & $buildPython -m pip install --disable-pip-version-check --no-index `
        --find-links (Join-Path $bundleRoot "wheels") `
        --require-hashes --requirement $runtimeLock
    if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned offline inference runtime." }

    $previousBuildCommit = $env:REEFVISION_EDGE_BUILD_COMMIT
    $previousRuntimeProfile = $env:REEFVISION_EDGE_RUNTIME_PROFILE
    try {
        $env:REEFVISION_EDGE_BUILD_COMMIT = $bundleMetadata.commit
        $env:REEFVISION_EDGE_RUNTIME_PROFILE = $bundleMetadata.runtime_profile
        & $buildPython -m PyInstaller --noconfirm --clean `
            --distpath $backendDist --workpath $backendWork $spec
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller backend build failed." }
    }
    finally {
        if ($null -eq $previousBuildCommit) {
            Remove-Item Env:REEFVISION_EDGE_BUILD_COMMIT -ErrorAction SilentlyContinue
        } else {
            $env:REEFVISION_EDGE_BUILD_COMMIT = $previousBuildCommit
        }
        if ($null -eq $previousRuntimeProfile) {
            Remove-Item Env:REEFVISION_EDGE_RUNTIME_PROFILE -ErrorAction SilentlyContinue
        } else {
            $env:REEFVISION_EDGE_RUNTIME_PROFILE = $previousRuntimeProfile
        }
    }

    $backendExe = Join-Path $backendDist "reefvision-edge-backend\reefvision-edge-backend.exe"
    if (-not (Test-Path -LiteralPath $backendExe)) {
        throw "Expected packaged backend was not produced: $backendExe"
    }
    Write-Host "Packaged backend: $backendExe"

    if ($BackendOnly) {
        return
    }

    foreach ($tool in "heat.exe", "candle.exe", "light.exe") {
        if (-not (Test-Path -LiteralPath (Join-Path $wixTools $tool))) {
            throw "Required WiX tool is missing: $(Join-Path $wixTools $tool)"
        }
    }
    if (-not (Test-Path -LiteralPath $WebView2Installer)) {
        throw "The supplied offline WebView2 installer is missing: $WebView2Installer"
    }
    $webView2Sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $WebView2Installer).Hash
    if ($webView2Sha256 -ne $expectedWebView2Sha256) {
        throw "WebView2 installer checksum mismatch. Expected $expectedWebView2Sha256, got $webView2Sha256."
    }

    & cargo build --release --manifest-path (Join-Path $repoRoot "packaging\windows\launcher\Cargo.toml")
    if ($LASTEXITCODE -ne 0) { throw "ReefVision-Edge launcher build failed." }
    if (-not (Test-Path -LiteralPath $launcherTarget)) {
        throw "Expected launcher was not produced: $launcherTarget"
    }

    Clear-BuildDirectory $stageRoot
    Clear-BuildDirectory $wixObjectRoot
    New-Item -ItemType Directory -Path (Join-Path $stageRoot "backend") -Force | Out-Null
    Copy-Item -LiteralPath $launcherTarget -Destination $launcherStage -Force
    Copy-Item -LiteralPath (Join-Path $backendDist "reefvision-edge-backend") `
        -Destination (Join-Path $stageRoot "backend") -Recurse -Force

    & (Join-Path $wixTools "heat.exe") dir (Join-Path $stageRoot "backend") `
        -nologo -gg -sfrag -srd -sreg -platform x64 `
        -dr BackendDir -cg BackendPayload -var "var.BackendSource" `
        -out $wixBackendSource
    if ($LASTEXITCODE -ne 0) { throw "WiX backend harvesting failed." }

    $candleArguments = @(
        "-nologo",
        "-arch", "x64",
        "-dLauncherSource=$launcherStage",
        "-dBackendSource=$(Join-Path $stageRoot 'backend')",
        "-dIconSource=$(Join-Path $repoRoot 'packaging\windows\launcher\icons\icon.ico')",
        "-dWebView2Installer=$WebView2Installer",
        "-out", ($wixObjectRoot + [System.IO.Path]::DirectorySeparatorChar),
        $wixSource,
        $wixBackendSource
    )
    & (Join-Path $wixTools "candle.exe") @candleArguments
    if ($LASTEXITCODE -ne 0) { throw "WiX source compilation failed." }

    $builtMsi = Join-Path $workRoot "ReefVision-Edge-0.1.0-windows-x64.msi"
    & (Join-Path $wixTools "light.exe") -nologo -sice:ICE38 -sice:ICE61 -sice:ICE64 -sice:ICE91 `
        -out $builtMsi `
        (Join-Path $wixObjectRoot "main.wixobj") `
        (Join-Path $wixObjectRoot "backend-payload.wixobj")
    if ($LASTEXITCODE -ne 0) { throw "WiX MSI linking failed." }
    if (-not (Test-Path -LiteralPath $builtMsi)) {
        throw "Expected MSI was not produced: $builtMsi"
    }

    Copy-Item -LiteralPath $builtMsi -Destination $downloadsMsi -Force
    Write-Host "Windows MSI: $downloadsMsi"
}
finally {
    Pop-Location
}
