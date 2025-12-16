REM =============================================================================
REM File: scripts/open_service_urls.bat
REM Purpose: Open all service URLs in browser
REM =============================================================================
@echo off
echo.
echo Opening all service URLs in browser...
echo.

start http://localhost:9090
echo Opened: Prometheus

timeout /t 1 /nobreak >nul
start http://localhost:3000
echo Opened: Grafana

timeout /t 1 /nobreak >nul
start http://localhost:3001/docs
echo Opened: BentoML Docs

timeout /t 1 /nobreak >nul
start http://localhost:8000/docs
echo Opened: FastAPI Docs

echo.
echo All service URLs opened!
echo.
pause