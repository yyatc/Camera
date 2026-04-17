@echo off
setlocal
set ROOT=%~dp0..
powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\setup.ps1" %*
endlocal
