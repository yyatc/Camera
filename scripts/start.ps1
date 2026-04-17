$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}

if (-not (Test-Path ".\config\secrets.env")) {
    throw "Missing config\secrets.env. Copy from config\secrets.env.example and fill camera credentials."
}

$mediaMtxExe = "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\bluenviron.mediamtx_Microsoft.Winget.Source_8wekyb3d8bbwe\mediamtx.exe"
if (-not (Test-Path $mediaMtxExe)) {
    throw "MediaMTX not found at expected path. Install with: winget install --id bluenviron.mediamtx -e"
}

$logDir = Join-Path $Root "runtime-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"

# Stop stale processes first.
Get-Process -Name python,mediamtx -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$mtxOut = Join-Path $logDir ("mediamtx-" + $ts + ".out.log")
$mtxErr = Join-Path $logDir ("mediamtx-" + $ts + ".err.log")
$appOut = Join-Path $logDir ("app-" + $ts + ".out.log")
$appErr = Join-Path $logDir ("app-" + $ts + ".err.log")
$meta = Join-Path $logDir ("session-" + $ts + ".txt")

$mtx = Start-Process -FilePath $mediaMtxExe -ArgumentList (Join-Path $Root "mediamtx.yml") -RedirectStandardOutput $mtxOut -RedirectStandardError $mtxErr -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 1
$app = Start-Process -FilePath (Join-Path $Root "venv\Scripts\python.exe") -ArgumentList "-m", "src.app.main" -WorkingDirectory $Root -RedirectStandardOutput $appOut -RedirectStandardError $appErr -PassThru -WindowStyle Hidden

@(
    "started_at=$((Get-Date).ToString('o'))"
    "mediamtx_pid=$($mtx.Id)"
    "app_pid=$($app.Id)"
    "mediamtx_out=$mtxOut"
    "mediamtx_err=$mtxErr"
    "app_out=$appOut"
    "app_err=$appErr"
) | Set-Content -Path $meta -Encoding UTF8

Write-Host "Started."
Write-Host "Session: $meta"
Write-Host "Stream URL: rtsp://127.0.0.1:8554/tracking"
