REM =============================================================================
REM File: scripts/stop_all_services.bat
REM Purpose: Stop all running ML monitoring services
REM =============================================================================
@echo off
echo.
echo ========================================
echo Stopping All ML Services
echo ========================================
echo.

REM Stop Grafana
echo [1/4] Stopping Grafana...
net stop Grafana 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    Grafana stopped successfully
) else (
    echo    Grafana was not running or failed to stop
)

REM Kill Prometheus
echo.
echo [2/4] Stopping Prometheus...
taskkill /FI "WindowTitle eq Prometheus*" /T /F 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    Prometheus stopped successfully
) else (
    echo    Prometheus was not running
)

REM Kill BentoML
echo.
echo [3/4] Stopping BentoML Service...
taskkill /FI "WindowTitle eq BentoML Service*" /T /F 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    BentoML stopped successfully
) else (
    echo    BentoML was not running
)

REM Kill FastAPI
echo.
echo [4/4] Stopping FastAPI Service...
taskkill /FI "WindowTitle eq FastAPI Service*" /T /F 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    FastAPI stopped successfully
) else (
    echo    FastAPI was not running
)

echo.
echo ========================================
echo All Services Stopped
echo ========================================
echo.
pause