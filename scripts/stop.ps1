$ErrorActionPreference = "Stop"

Get-Process -Name python,mediamtx -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 300

Write-Host "Stopped python and mediamtx processes."
