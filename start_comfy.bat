@echo off
REM Launch ComfyUI as the render engine for Manhua Studio.
REM
REM --lowvram is what lets Anima (2B DiT + Qwen-3 encoder + Qwen VAE) run on a
REM 6GB card. Without it ComfyUI still works, but Anima loads to near-zero free
REM VRAM and generation slows to a crawl. run_nvidia_gpu.bat does NOT pass it.

set COMFY=C:\ComfyUI\ComfyUI_windows_portable

if not exist "%COMFY%\python_embeded\python.exe" (
    echo ComfyUI not found at %COMFY%
    echo Edit the COMFY path at the top of this file.
    pause
    exit /b 1
)

echo Starting ComfyUI with --lowvram on http://127.0.0.1:8188
echo Leave this window open while you work. Closing it stops rendering.
echo(

cd /d "%COMFY%"
python_embeded\python.exe -s ComfyUI\main.py --lowvram --port 8188

pause
