@echo off
setlocal
set ROOT=%~dp0..
powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\start.ps1"
endlocal
