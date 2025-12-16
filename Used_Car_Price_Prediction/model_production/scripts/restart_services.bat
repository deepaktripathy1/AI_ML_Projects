REM =============================================================================
REM File: scripts/restart_services.bat
REM Purpose: Restart all services
REM =============================================================================
@echo off
echo.
echo ========================================
echo Restarting All ML Services
echo ========================================
echo.

echo Step 1: Stopping services...
call "%~dp0stop_all_services.bat"

echo.
echo Step 2: Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo.
echo Step 3: Starting services...
call "%~dp0start_all_services.bat"