@echo off
title BPP download agent - leave this window open
cd /d "%~dp0"
echo Starting the BPP download agent in: %CD%
echo.
set "PYEXE="
if exist "%USERPROFILE%\MiniConda3\python.exe" set "PYEXE=%USERPROFILE%\MiniConda3\python.exe"
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE (
  where py >nul 2>nul && set "PYEXE=py"
)
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo Could not find Python on this laptop. Tell Claude.
  pause
  exit /b 1
)
:run
"%PYEXE%" "%~dp0bpp_fetch.py"
if %errorlevel%==3 (
  echo.
  echo Restarting with the new version...
  goto run
)
echo.
echo The agent has stopped. You can close this window.
pause
