@echo off
setlocal
cd /d "%~dp0"
title Manhua Studio

REM Everything past the environment lives in manhua/launch.py: finding and
REM starting ComfyUI, checking Ollama, opening the browser. Batch is a bad
REM language for socket probes and this file only has to get Python running.

where python >nul 2>&1
if errorlevel 1 goto :nopython

if not exist ".venv\Scripts\python.exe" (
    echo(
    echo   Creating the virtual environment. This happens once.
    python -m venv .venv
    if errorlevel 1 goto :venvfail
)
set PY=.venv\Scripts\python.exe

"%PY%" -c "import fastapi, uvicorn, pydantic, PIL, yaml" >nul 2>&1
if errorlevel 1 (
    echo(
    echo   Installing dependencies. This happens once.
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :depfail
)

"%PY%" -m manhua.launch %*
if errorlevel 1 goto :runfail
goto :eof

REM ---------------------------------------------------------------- errors
:nopython
echo(
echo   Python was not found on PATH.
echo(
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   During setup, TICK "Add Python to PATH".
echo(
pause
goto :eof

:venvfail
echo(
echo   Could not create the virtual environment.
echo   Delete the .venv folder and run this again.
echo(
pause
goto :eof

:depfail
echo(
echo   Dependency installation failed. Check your internet connection,
echo   then run this file again.
echo(
pause
goto :eof

:runfail
echo(
echo   The studio exited with an error. Scroll up for the reason.
echo(
pause
goto :eof
