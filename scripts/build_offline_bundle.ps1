param(
    [string]$OutputPath = ".artifacts\reefvision-edge-offline",
    [ValidateSet("cpu", "cuda")]
    [string]$RuntimeProfile = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$bundlePath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".artifacts"))
if (-not $bundlePath.StartsWith($artifactRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Output must be a child of $artifactRoot"
}
& (Join-Path $PSScriptRoot "build_windows_runtime_bundle.ps1") `
    -Profile $RuntimeProfile -OutputPath $OutputPath
if ($LASTEXITCODE -ne 0) { throw "Windows runtime bundle build failed." }
& (Join-Path $PSScriptRoot "build_windows_converter_bundle.ps1") `
    -OutputPath $OutputPath
if ($LASTEXITCODE -ne 0) { throw "Windows converter bundle build failed." }
$converterWheels = Join-Path $bundlePath "converter-wheels"
if (-not (Test-Path -LiteralPath $converterWheels)) {
    throw "Converter bundle did not produce converter-wheels."
}
New-Item -ItemType Directory -Path (Join-Path $bundlePath "app") | Out-Null

Push-Location $repoRoot
try {
    npm --prefix web ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed with exit code $LASTEXITCODE" }
    npm --prefix web run build
    if ($LASTEXITCODE -ne 0) { throw "npm build failed with exit code $LASTEXITCODE" }
    $generatedPaths = @(
        (Join-Path $repoRoot "build"),
        (Join-Path $repoRoot "src\reefvision_edge.egg-info")
    )
    foreach ($generatedPath in $generatedPaths) {
        $absoluteGeneratedPath = [System.IO.Path]::GetFullPath($generatedPath)
        if (-not $absoluteGeneratedPath.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to clean generated output outside the repository: $absoluteGeneratedPath"
        }
        if (Test-Path -LiteralPath $generatedPath) {
            Remove-Item -LiteralPath $generatedPath -Recurse -Force
        }
    }
    & $python -m build --wheel --outdir (Join-Path $bundlePath "app")
    if ($LASTEXITCODE -ne 0) { throw "wheel build failed with exit code $LASTEXITCODE" }
    & $python -m pip download --dest (Join-Path $bundlePath "wheels") ".[windows-package]"
    if ($LASTEXITCODE -ne 0) { throw "wheel download failed with exit code $LASTEXITCODE" }
    $researchFiles = @(
        & git ls-files "*.pt"
        & git ls-files "*.pth"
        & git ls-files "*.onnx"
    ) | Where-Object { $_ }
    if ($researchFiles.Count) {
        throw "Research weights or derived models are tracked and would enter release assets: $($researchFiles -join ', ')"
    }
    $sourceArchive = Join-Path $bundlePath "reefvision-edge-source.zip"
    git archive --format=zip -o $sourceArchive HEAD
    if ($LASTEXITCODE -ne 0) { throw "source archive failed with exit code $LASTEXITCODE" }
    $bundleMetadata = [ordered]@{
        product = "ReefVision-Edge"
        version = "0.1.0"
        commit = (& git rev-parse HEAD).Trim()
        runtime_profile = $RuntimeProfile
        platform = "windows-x64"
    }
    $bundleMetadata | ConvertTo-Json | Set-Content `
        -LiteralPath (Join-Path $bundlePath "BUNDLE-METADATA.json") -Encoding utf8
    Get-ChildItem -Path $bundlePath -Recurse -File |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
        Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash.ToLower())  $($_.Path.Substring($bundlePath.Length + 1).Replace('\', '/'))" } |
        Set-Content -LiteralPath (Join-Path $bundlePath "SHA256SUMS.txt") -Encoding utf8
}
finally {
    Pop-Location
}

Write-Host "Offline bundle created at $bundlePath"
