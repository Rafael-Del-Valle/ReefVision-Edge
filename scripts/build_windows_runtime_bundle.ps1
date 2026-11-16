[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Profile = "cuda",
    [string]$OutputPath,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$lockName = if ($Profile -eq "cuda") { "runtime-windows-cuda.lock" } else { "runtime-windows-cpu.lock" }
$lockPath = Join-Path $repoRoot "packaging\windows\$lockName"
if (-not (Test-Path -LiteralPath $lockPath)) {
    throw "Runtime lock is missing: $lockPath"
}

$requirements = [System.Collections.Generic.List[string]]::new()
$allHashed = $true
foreach ($line in Get-Content -LiteralPath $lockPath) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
    $match = [regex]::Match(
        $trimmed,
        "^([A-Za-z0-9_.-]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})$"
    )
    if (-not $match.Success) {
        $allHashed = $false
        throw "Runtime lock entry is not exact and SHA-256 hashed: $trimmed"
    }
    $requirements.Add("$($match.Groups[1].Value)==$($match.Groups[2].Value)")
}
if ($requirements.Count -eq 0) { throw "Runtime lock contains no requirements." }

$lockSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath).Hash.ToLowerInvariant()
$metadata = [ordered]@{
    profile = $Profile
    platform = "windows-x64"
    python = "cp312"
    lock_file = $lockName
    lock_sha256 = $lockSha256
    all_requirements_hashed = $allHashed
    requirements = @($requirements)
}
if ($ValidateOnly) {
    $metadata | ConvertTo-Json -Depth 4 -Compress
    return
}

$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".artifacts"))
$resolvedOutput = if ($OutputPath) {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
} else {
    Join-Path $artifactRoot "windows-runtime-$Profile"
}
if (-not $resolvedOutput.StartsWith(
    $artifactRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Runtime output must be a child of $artifactRoot"
}
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Refusing to overwrite existing runtime bundle: $resolvedOutput"
}

$wheels = Join-Path $resolvedOutput "wheels"
New-Item -ItemType Directory -Path $wheels -Force | Out-Null
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
& $python -m pip download --disable-pip-version-check `
    --only-binary=:all: --require-hashes `
    --platform win_amd64 --python-version 312 --implementation cp --abi cp312 `
    --dest $wheels --requirement $lockPath
if ($LASTEXITCODE -ne 0) { throw "Pinned Windows runtime download failed." }

Copy-Item -LiteralPath $lockPath -Destination (Join-Path $resolvedOutput "runtime.lock")
$metadata | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $resolvedOutput "runtime-profile.json") -Encoding utf8
Get-ChildItem -Path $wheels -File |
    Get-FileHash -Algorithm SHA256 |
    Sort-Object Path |
    ForEach-Object {
        "$($_.Hash.ToLowerInvariant())  wheels/$([System.IO.Path]::GetFileName($_.Path))"
    } |
    Set-Content -LiteralPath (Join-Path $resolvedOutput "RUNTIME-SHA256SUMS.txt") -Encoding utf8

Write-Host "Windows $Profile runtime bundle created at $resolvedOutput"
