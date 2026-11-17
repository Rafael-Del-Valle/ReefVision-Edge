[CmdletBinding()]
param(
    [string]$OutputPath = ".artifacts\reefvision-edge-offline",
    [switch]$ValidateOnly
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
$lock = Join-Path $repoRoot "packaging\windows\model-converter.lock"
$review = Join-Path $repoRoot "packaging\windows\model-converter-licenses.json"
$wheels = Join-Path $bundlePath "converter-wheels"
$licenses = Join-Path $bundlePath "converter-licenses"
$requirements = Get-Content -LiteralPath $lock | Where-Object { $_ -and -not $_.StartsWith("#") }
if ($requirements.Count -lt 40 -or $requirements.Where({ $_ -notmatch '--hash=sha256:[0-9a-f]{64}$' }).Count) {
    throw "Converter lock is incomplete or contains an unhashed requirement."
}
$licenseReview = Get-Content -Raw -LiteralPath $review | ConvertFrom-Json
if ($licenseReview.project_license -ne "AGPL-3.0-only" -or $licenseReview.ultralytics_license -ne "AGPL-3.0-only") {
    throw "Converter license review is missing the AGPL compatibility decision."
}
if ($ValidateOnly) {
    [ordered]@{
        platform = "windows-x64"
        python = "3.12"
        requirements = $requirements
        lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $lock).Hash.ToLowerInvariant()
        reviewed_packages = $licenseReview.packages.Count
    } | ConvertTo-Json -Depth 3
    return
}
New-Item -ItemType Directory -Path $wheels -Force | Out-Null
& $python -m pip download --disable-pip-version-check --require-hashes `
    --only-binary=:all: --platform win_amd64 --python-version 312 `
    --implementation cp --abi cp312 --dest $wheels --requirement $lock
if ($LASTEXITCODE -ne 0) { throw "Pinned converter wheel download failed." }
& $python (Join-Path $repoRoot "scripts\collect_converter_licenses.py") `
    --wheels $wheels --output $licenses --lock $lock --review $review
if ($LASTEXITCODE -ne 0) { throw "Converter license collection failed." }
Copy-Item -LiteralPath $lock -Destination (Join-Path $bundlePath "model-converter.lock") -Force
Copy-Item -LiteralPath $review -Destination (Join-Path $licenses "REVIEW.json") -Force
Write-Host "Pinned model converter bundle created at $wheels"
