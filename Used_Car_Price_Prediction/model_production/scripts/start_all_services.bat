@echo off
echo ========================================
echo Starting ML Model Monitoring Stack
echo ========================================

REM Get the script directory (where this .bat file is located)
set "SCRIPT_DIR=%~dp0"
REM Get project root (parent of scripts folder)
set "PROJECT_ROOT=%SCRIPT_DIR%"..
cd /d "%PROJECT_ROOT%"

echo Script directory: %SCRIPT_DIR%
echo Project root: %PROJECT_ROOT%
echo.

echo [1/5] Starting Prometheus...
REM Check if Prometheus is already running
netstat -ano | findstr ":9090" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: Already running on port 9090
) else (
    if exist "C:\prometheus\prometheus.exe" (
        start "Prometheus" cmd /k "cd /d C:\prometheus && prometheus.exe --config.file=prometheus.yml --web.listen-address=:9090"
        echo    Status: Started on http://localhost:9090
        timeout 5 sleep 1
    ) else (
        echo    WARNING: Prometheus not found at C:\prometheus\
        echo    Please install Prometheus first!
    )
)
echo.

echo [2/5] Starting Grafana...
REM Check if Grafana is already running
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: Already running on port 3000
    echo    Access at: http://localhost:3000
) else (
    REM Try to start Grafana service
    net start Grafana >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo    Status: Started successfully
        echo    Access at: http://localhost:3000
        timeout 5 sleep 1
    ) else (
        REM Check if access denied or already running
        net start Grafana 2>&1 | findstr "denied" >nul
        if %ERRORLEVEL% EQU 0 (
            echo    Status: Access denied - requires Administrator privileges
            echo    Please start Grafana manually or run this script as Administrator
            echo    Or start from Windows Services as Administrator
        ) else (
            echo    Status: Could not start (may already be running)
            echo    Check Windows Services or start manually
        )
    )
)
echo.

echo [3/5] Starting BentoML Service...
REM Check if BentoML is already running
netstat -ano | findstr ":3001" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: Already running on port 3001
) else (
    if exist "%SCRIPT_DIR%start_bentoml_service.bat" (
        start "BentoML Service" cmd /k "cd /d %SCRIPT_DIR% && start_bentoml_service.bat"
        echo    Status: Starting on http://localhost:3001
        echo    Docs: http://localhost:3001/docs
        timeout 10 sleep 1
    ) else (
        echo    ERROR: start_bentoml_service.bat not found
        echo    Expected location: %SCRIPT_DIR%start_bentoml_service.bat
        echo    Please run deployment pipeline first: python deployment_pipeline.py
    )
)
echo.

echo [4/5] Starting FastAPI Service...
REM Check if FastAPI is already running
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: Already running on port 8000
) else (
    if exist "%SCRIPT_DIR%start_fastapi_service.bat" (
        start "FastAPI Service" cmd /k "cd /d %SCRIPT_DIR% && start_fastapi_service.bat"
        echo    Status: Starting on http://localhost:8000
        echo    Docs: http://localhost:8000/docs
        timeout 5 sleep 1
    ) else (
        echo    ERROR: start_fastapi_service.bat not found
        echo    Expected location: %SCRIPT_DIR%start_fastapi_service.bat
        echo    Please run deployment pipeline first: python deployment_pipeline.py
    )
)
echo.

echo [5/5] Starting Alertmanager...
REM Check if Alertmanager is already running
netstat -ano | findstr ":9093" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Status: Already running on port 9093
) else (
    if exist "C:\alertmanager\alertmanager.exe" (
        start "Alertmanager" cmd /k "cd /d C:\alertmanager && alertmanager.exe --config.file=alertmanager.yml --web.listen-address=:9093"
        echo    Status: Started on http://localhost:9093
        timeout 5 sleep 1
    ) else (
        echo    WARNING: Alertmanager not found at C:\alertmanager\
        echo    Please install Alertmanager first!
    )
)
echo.

echo ========================================
echo All services started!
echo ========================================
echo.
echo Service URLs:
echo - Prometheus:  http://localhost:9090
echo - Alertmanager: http://localhost:9093
echo - Grafana:     http://localhost:3000
echo - BentoML:     http://localhost:3001
echo - FastAPI:     http://localhost:8000
echo - FastAPI Docs: http://localhost:8000/docs
echo.
echo NOTES:
echo --------------------------------------------
echo - Each service runs in its own window
echo - DO NOT close these windows to keep services running
echo - Grafana may require Administrator privileges
echo - Check individual windows for any errors
echo.
echo To stop all services, close each window individually
echo or run: scripts\stop_all_services.bat
echo.
echo ========================================
echo.

REM Show service status
echo Current Service Status:
echo --------------------------------------------
netstat -ano | findstr ":9093 :9090 :3000 :3001 :8000" | findstr "LISTENING"
echo.

pause