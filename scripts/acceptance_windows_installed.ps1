[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8765"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-That([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workRoot = Join-Path $repoRoot ".artifacts\windows-msi"
$acceptanceRoot = Join-Path $workRoot "installed-acceptance"
$installRoot = Join-Path $env:LOCALAPPDATA "Programs\ReefVision-Edge"
$expectedBackend = Join-Path $installRoot "backend\reefvision-edge-backend\reefvision-edge-backend.exe"

$resolvedAcceptance = [System.IO.Path]::GetFullPath($acceptanceRoot)
$resolvedWork = [System.IO.Path]::GetFullPath($workRoot)
if (-not $resolvedAcceptance.StartsWith($resolvedWork + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing acceptance output outside Windows build artifacts: $resolvedAcceptance"
}
if (Test-Path -LiteralPath $acceptanceRoot) {
    Remove-Item -LiteralPath $acceptanceRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $acceptanceRoot -Force | Out-Null

$health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 5
Assert-That ($health.product -eq "ReefVision-Edge") "Installed service returned the wrong product."
Assert-That ($health.version -eq "0.1.0") "Installed service returned the wrong version."

$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction Stop
Assert-That ($listener.LocalAddress -eq "127.0.0.1") "Installed service is not bound exclusively to IPv4 loopback."
$backendProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
Assert-That ($null -ne $backendProcess) "Could not resolve the installed backend process."
Assert-That ([System.IO.Path]::GetFullPath($backendProcess.ExecutablePath) -eq [System.IO.Path]::GetFullPath($expectedBackend)) "The running service is not the installed backend executable."

$fixtureBuilder = @'
import json
import sys
from pathlib import Path

from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
build_tiny_yolo_model(root / "tiny.onnx")
build_mjpeg_video(root / "recorded.avi")
metadata = {
    "name": "Installed ReefVision-Edge Smoke Model",
    "format": "ultralytics_yolo_detect_v1",
    "input_width": 640,
    "input_height": 640,
    "classes": {"0": "coral", "1": "fish"},
    "preprocessing": {"color_order": "rgb", "scale": "0_1", "letterbox": "centered"},
    "output": {"layout": "1x(4+classes)xN", "box_format": "xywh", "score_format": "class_probabilities"},
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
}
(root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
'@
$sourcePython = Join-Path $repoRoot ".venv\Scripts\python.exe"
& $sourcePython -c $fixtureBuilder $acceptanceRoot
if ($LASTEXITCODE -ne 0) { throw "Could not create deterministic acceptance fixtures." }

$modelPath = Join-Path $acceptanceRoot "tiny.onnx"
$metadataPath = Join-Path $acceptanceRoot "metadata.json"
$videoPath = Join-Path $acceptanceRoot "recorded.avi"

try {
    $model = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/models" -Form @{
        model = Get-Item -LiteralPath $modelPath
        metadata_json = Get-Content -Raw -LiteralPath $metadataPath
    }
}
catch {
    $model = Invoke-RestMethod -Uri "$BaseUrl/api/models" | Where-Object name -eq "Installed ReefVision-Edge Smoke Model" | Select-Object -First 1
    if ($null -eq $model) { throw }
}

try {
    $source = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/sources/video" -Form @{
        video = Get-Item -LiteralPath $videoPath
    }
}
catch {
    $source = Invoke-RestMethod -Uri "$BaseUrl/api/sources" | Where-Object name -eq "recorded.avi" | Select-Object -First 1
    if ($null -eq $source) { throw }
}

$session = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/sessions" `
    -ContentType "application/json" `
    -Body (@{model_id=$model.id;source_id=$source.id;sample_every=2} | ConvertTo-Json -Compress)
$null = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/sessions/$($session.id)/start"

$deadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Milliseconds 100
    $session = Invoke-RestMethod -Uri "$BaseUrl/api/sessions/$($session.id)"
} while ($session.status -in @("queued", "running") -and [DateTime]::UtcNow -lt $deadline)

Assert-That ($session.status -eq "completed") "Installed inference did not complete: $($session.status) $($session.error_message)"
Assert-That ($session.frame_count -eq 3) "Installed inference returned the wrong frame count."
Assert-That ($session.detection_count -eq 6) "Installed inference returned the wrong detection count."
Assert-That ($session.provider -eq "CPUExecutionProvider") "Installed inference used the wrong provider."

$previewPath = Join-Path $acceptanceRoot "preview.jpg"
$preview = Invoke-WebRequest -Uri "$BaseUrl/api/previews/$($session.id)" -OutFile $previewPath -PassThru
Assert-That ($preview.StatusCode -eq 200) "Installed preview request failed."
Assert-That ((Get-Item -LiteralPath $previewPath).Length -gt 100) "Installed preview is empty."

$exportPath = Join-Path $acceptanceRoot "$($session.id).zip"
$export = Invoke-WebRequest -Uri "$BaseUrl/api/sessions/$($session.id)/export" -OutFile $exportPath -PassThru
Assert-That ($export.StatusCode -eq 200) "Installed export request failed."

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($exportPath)
try {
    $names = @($archive.Entries | ForEach-Object FullName)
    Assert-That (($names -join ",") -eq "manifest.json,detections.csv") "Installed export contains unexpected files."
    $manifestEntry = $archive.GetEntry("manifest.json")
    $reader = [System.IO.StreamReader]::new($manifestEntry.Open())
    try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    Assert-That ($manifest.schema -eq "reefvision-edge-session/v1") "Installed export schema is invalid."
    Assert-That ($manifest.session.id -eq $session.id) "Installed export references the wrong session."
    Assert-That ($manifest.session.detection_count -eq 6) "Installed export detection count is invalid."
}
finally {
    $archive.Dispose()
}

$result = [ordered]@{
    product = $health.product
    version = $health.version
    service_executable = $backendProcess.ExecutablePath
    bind_address = $listener.LocalAddress
    data_directory = Join-Path $env:LOCALAPPDATA "ReefVision-Edge"
    session_id = $session.id
    status = $session.status
    frame_count = $session.frame_count
    detection_count = $session.detection_count
    provider = $session.provider
    preview = $previewPath
    export = $exportPath
}
$report = Join-Path $acceptanceRoot "result.json"
$result | ConvertTo-Json | Set-Content -LiteralPath $report -Encoding utf8
$result | ConvertTo-Json -Compress
