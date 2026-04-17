param(
    [switch]$InstallMediaMtx
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ not found in PATH."
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

Write-Host "Installing Python dependencies..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found in PATH. Install ffmpeg and ensure command 'ffmpeg' is available."
}

if ($InstallMediaMtx) {
    Write-Host "Installing MediaMTX via winget..."
    winget install --id bluenviron.mediamtx -e --accept-package-agreements --accept-source-agreements
}

if (-not (Test-Path ".\config\secrets.env")) {
    Copy-Item ".\config\secrets.env.example" ".\config\secrets.env"
    Write-Host "Created config\secrets.env from example. Fill camera credentials before start."
}

Write-Host "Setup completed."
