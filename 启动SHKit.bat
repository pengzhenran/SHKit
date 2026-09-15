@echo off
rem ===================================================================
rem  SHKit GUI launcher
rem  Double-click this file to start the SHKit desktop GUI.
rem  No console window is shown.
rem
rem  Notes for maintenance:
rem    - Keep this file ASCII-only.  cmd.exe reads .bat in the OEM code
rem      page (GBK on a Chinese Windows) and mis-parses UTF-8 Chinese in
rem      comment lines, which produces spurious "not recognized" errors.
rem    - Python is hard-coded because the bare `python` on PATH is the
rem      Microsoft Store stub, which cannot import numpy/PySide6.
rem ===================================================================
setlocal
pushd "%~dp0"

set "PYW=C:\Users\pengzhenran\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=C:\Users\pengzhenran\anaconda3\python.exe"
if not exist "%PYW%" set "PYW=pythonw"

start "" "%PYW%" -m shkit.gui %*

popd
endlocal
