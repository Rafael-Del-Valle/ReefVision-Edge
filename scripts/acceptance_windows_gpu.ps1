[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BackendExe,
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,
    [double]$DurationSeconds = 60.0,
    [string]$OutputPath = ".artifacts\windows-gpu-acceptance"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backend = (Resolve-Path -LiteralPath $BackendExe).Path
$bundle = (Resolve-Path -LiteralPath $BundlePath).Path
$bundleMetadata = Get-Content -Raw -LiteralPath (Join-Path $bundle "BUNDLE-METADATA.json") | ConvertFrom-Json
$output = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".artifacts"))
if (-not $output.StartsWith(
    $artifactRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "GPU acceptance output must be under $artifactRoot"
}
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Recurse -Force
}
New-Item -ItemType Directory -Path $output -Force | Out-Null

$diagnostics = & $backend --runtime-diagnostics | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Packaged runtime diagnostics failed." }
if ($diagnostics.nvidia_runtime_preload -ne "loaded") {
    throw "Packaged NVIDIA runtime did not preload: $($diagnostics.preload_error)"
}
if ($diagnostics.available_providers -notcontains "CUDAExecutionProvider") {
    throw "Packaged runtime does not expose CUDAExecutionProvider."
}
if ($diagnostics.build_commit -ne $bundleMetadata.commit) {
    throw "Packaged backend commit $($diagnostics.build_commit) does not match bundle commit $($bundleMetadata.commit)."
}
if ($diagnostics.runtime_profile -ne $bundleMetadata.runtime_profile) {
    throw "Packaged backend runtime profile does not match the supplied bundle."
}

$listener = [System.Net.Sockets.TcpListener]::new(
    [System.Net.IPAddress]::Loopback,
    0
)
$listener.Start()
$port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()
$dataDir = Join-Path $output "data"
$logPath = Join-Path $output "backend.log"
$process = $null
try {
    $process = Start-Process -FilePath $backend -ArgumentList @(
        "--host", "127.0.0.1",
        "--port", "$port",
        "--data-dir", $dataDir
    ) -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError (Join-Path $output "backend-error.log") -PassThru

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        if ($process.HasExited) {
            throw "Packaged backend exited early with code $($process.ExitCode)."
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 1
        } catch {
            $health = $null
            Start-Sleep -Milliseconds 100
        }
    } while ($null -eq $health -and [DateTime]::UtcNow -lt $deadline)
    if ($null -eq $health -or $health.product -ne "ReefVision-Edge") {
        throw "Packaged backend did not become healthy on loopback."
    }

    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    & $python (Join-Path $PSScriptRoot "acceptance_windows_gpu.py") `
        --base-url "http://127.0.0.1:$port" `
        --output-dir $output `
        --bundle-path $bundle `
        --duration-seconds $DurationSeconds
    if ($LASTEXITCODE -ne 0) { throw "RTX 3090 acceptance harness failed." }
}
finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit(10000)
    }
}
