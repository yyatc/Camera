@echo off
setlocal
set ROOT=%~dp0..
powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\stop.ps1"
endlocal
