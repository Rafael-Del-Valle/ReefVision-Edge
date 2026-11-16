param(
    [string]$OutputPath = ".artifacts\reefvision-edge-offline"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundlePath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".artifacts"))
if (-not $bundlePath.StartsWith($artifactRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Output must be a child of $artifactRoot"
}
if (Test-Path -LiteralPath $bundlePath) {
    throw "Refusing to overwrite existing bundle: $bundlePath"
}

New-Item -ItemType Directory -Path $bundlePath | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundlePath "app") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundlePath "wheels") | Out-Null

Push-Location $repoRoot
try {
    npm --prefix web ci
    npm --prefix web run build
    python -m build --wheel --outdir (Join-Path $bundlePath "app")
    python -m pip download --dest (Join-Path $bundlePath "wheels") .
    $sourceArchive = Join-Path $bundlePath "reefvision-edge-source.zip"
    git archive --format=zip -o $sourceArchive HEAD
    Get-ChildItem -Path $bundlePath -Recurse -File |
        Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash.ToLower())  $($_.Path.Substring($bundlePath.Length + 1).Replace('\', '/'))" } |
        Set-Content -LiteralPath (Join-Path $bundlePath "SHA256SUMS.txt") -Encoding utf8
}
finally {
    Pop-Location
}

Write-Host "Offline bundle created at $bundlePath"
