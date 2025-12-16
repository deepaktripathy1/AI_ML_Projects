REM =============================================================================
REM Grafana
REM =============================================================================
echo.
echo Starting Grafana...
echo --------------------------------------------
net start Grafana 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    Status: Started on http://localhost:3000
) else (
    echo    Status: Already running or not installed as service
    echo    Try starting manually from Windows Services
)
timeout /t 5 /nobreak >nul