@echo off
REM Kept so the old habit still works. start.bat is the real launcher: it
REM starts ComfyUI first if it is not already up, which this never did.
cd /d "%~dp0"
call start.bat %*
