REM =============================================================================
REM File: scripts/check_services.bat
REM Purpose: Check status of all services
REM =============================================================================
@echo off
echo.
echo ========================================
echo Service Status Check
echo ========================================
echo.

REM Check Prometheus
echo [1/4] Checking Prometheus (Port 9090)...
curl -s http://localhost:9090/-/healthy >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: RUNNING
) else (
    echo    Status: NOT RUNNING
)

REM Check Grafana
echo.
echo [2/4] Checking Grafana (Port 3003)...
curl -s http://localhost:3000/api/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: RUNNING
) else (
    echo    Status: NOT RUNNING
)

REM Check BentoML
echo.
echo [3/4] Checking BentoML (Port 3001)...
curl -s http://localhost:3001/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: RUNNING
) else (
    echo    Status: NOT RUNNING
)

REM Check FastAPI
echo.
echo [4/4] Checking FastAPI (Port 8000)...
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: RUNNING
) else (
    echo    Status: NOT RUNNING
)

echo.
echo ========================================
echo Check Complete
echo ========================================
echo.

REM Show open ports
echo Active Network Connections:
echo --------------------------------------------
netstat -ano | findstr "9090 3000 3001 8000" | findstr "LISTENING"

echo.
pause
