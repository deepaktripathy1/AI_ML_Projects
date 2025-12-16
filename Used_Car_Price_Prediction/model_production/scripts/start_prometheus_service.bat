@echo off
echo ==================================================
echo Starting Prometheus and Alertmanager
echo ==================================================

echo [1/2] Starting Prometheus...
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
        echo    ERROR: Prometheus not found at C:\prometheus\
        echo    Please install Prometheus first!
    )
)
echo.

echo [2/2] Starting Alertmanager...
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

pause