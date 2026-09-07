@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Manhua Studio

echo(
echo  ========================================
echo    Manhua Studio
echo  ========================================
echo(

REM ---------------------------------------------------------------- python
where python >nul 2>&1
if errorlevel 1 goto :nopython

REM ---------------------------------------------------------------- venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment ^(one time^)...
    python -m venv .venv
    if errorlevel 1 goto :venvfail
) else (
    echo [1/4] Virtual environment OK
)
set PY=.venv\Scripts\python.exe

REM ---------------------------------------------------------------- core deps
"%PY%" -c "import fastapi, pydantic, PIL, yaml" >nul 2>&1
if errorlevel 1 (
    echo [2/4] Installing core dependencies...
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :depfail
) else (
    echo [2/4] Core dependencies OK
)

REM ---------------------------------------------------------------- gpu + torch
REM Native rendering needs torch built for the installed CUDA runtime. Without
REM an NVIDIA GPU we skip it entirely -- the studio still runs with the mock
REM backend, or against ComfyUI.
"%PY%" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    echo [3/4] PyTorch OK
    goto :launch
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [3/4] No NVIDIA GPU detected - skipping PyTorch
    echo       The studio will run, but rendering needs a GPU or ComfyUI.
    goto :launch
)

echo [3/4] NVIDIA GPU detected. PyTorch is a ~2.5 GB download.
set /p TORCHYN="      Install it now for local rendering? [Y/n] "
if /i "!TORCHYN!"=="n" goto :launch

echo       Installing PyTorch ^(CUDA 12.4^)...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo       CUDA wheel failed; falling back to the default index...
    "%PY%" -m pip install torch
)
"%PY%" -m pip install -r requirements-local.txt

:launch
echo [4/4] Starting studio...
echo(
"%PY%" -m manhua.cli studio %*
if errorlevel 1 goto :runfail
goto :eof

REM ---------------------------------------------------------------- errors
:nopython
echo(
echo  ERROR: Python was not found on PATH.
echo(
echo  Install Python 3.10 or newer from https://www.python.org/downloads/
echo  During setup, TICK "Add Python to PATH".
echo(
pause
goto :eof

:venvfail
echo(
echo  ERROR: Could not create the virtual environment.
echo  Try deleting the .venv folder and running this again.
echo(
pause
goto :eof

:depfail
echo(
echo  ERROR: Dependency installation failed.
echo  Check your internet connection, then run this file again.
echo(
pause
goto :eof

:runfail
echo(
echo  The studio exited with an error. Scroll up for details.
echo(
pause
goto :eof
