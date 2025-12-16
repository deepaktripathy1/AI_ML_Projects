REM =============================================================================
REM Alertmanager
REM =============================================================================
echo Starting Alertmanager...
echo --------------------------------------------
if exist "C:\alertmanager\alertmanager.exe" (
    start "Alertmanager" cmd /k "cd /d C:\alertmanager && alertmanager.exe --config.file=alertmanager.yml --web.listen-address=:9093"
    echo    Status: Started on http://localhost:9093
    timeout /t 5 /nobreak >nul
) else (
    echo    ERROR: Alertmanager not found at C:\alertmanager\
    echo    Please install Alertmanager first!
    echo.
)