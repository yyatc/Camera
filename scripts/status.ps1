$ErrorActionPreference = "SilentlyContinue"

$procs = Get-Process -Name python,mediamtx | Select-Object Id, ProcessName, Path
if ($procs) {
    Write-Host "Processes:"
    $procs | Format-Table -AutoSize
} else {
    Write-Host "No python/mediamtx processes found."
}

Write-Host ""
Write-Host "RTSP 8554 sockets:"
netstat -ano | findstr :8554
