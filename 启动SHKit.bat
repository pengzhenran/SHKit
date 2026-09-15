@echo off
rem ===================================================================
rem  SHKit GUI launcher  (portable)
rem  Double-click this file to start the SHKit desktop GUI.
rem  No console window is shown.
rem
rem  Notes for maintenance:
rem    - Keep this file ASCII-only.  cmd.exe reads .bat in the OEM code
rem      page (GBK on a Chinese Windows) and mis-parses UTF-8 Chinese in
rem      comment lines, which produces spurious "not recognized" errors.
rem    - `python` on PATH may be the Microsoft Store stub, which cannot
rem      import numpy/PySide6.  This launcher therefore tries, in order:
rem        1. %SHKIT_PYTHON%  (set it to your interpreter if you have one)
rem        2. the Anaconda install under %USERPROFILE%
rem        3. the py launcher (py -3)
rem        4. pythonw / python on PATH
rem    - Run this file from the project root so that the `shkit` package
rem      next to it is importable.
rem ===================================================================
setlocal
pushd "%~dp0"

set "PYW="
if defined SHKIT_PYTHON if exist "%SHKIT_PYTHON%" set "PYW=%SHKIT_PYTHON%"
if not defined PYW if exist "%USERPROFILE%\anaconda3\pythonw.exe" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not defined PYW if exist "%USERPROFILE%\miniconda3\pythonw.exe" set "PYW=%USERPROFILE%\miniconda3\pythonw.exe"
if not defined PYW if exist "%LOCALAPPDATA%\Programs\Python" for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined PYW if exist "%%~fD\pythonw.exe" set "PYW=%%~fD\pythonw.exe"
if not defined PYW if exist "%USERPROFILE%\anaconda3\python.exe" set "PYW=%USERPROFILE%\anaconda3\python.exe"
if not defined PYW if exist "%USERPROFILE%\miniconda3\python.exe" set "PYW=%USERPROFILE%\miniconda3\python.exe"

if defined PYW (
  start "" "%PYW%" -m shkit.gui %*
) else (
  where py >nul 2>nul && ( start "" py -3 -m shkit.gui %* & goto :done )
  where pythonw >nul 2>nul && ( start "" pythonw -m shkit.gui %* & goto :done )
  echo [x] 没找到可用的 Python 解释器。
  echo     请先安装 Python 3.10+ 并装好依赖：
  echo         python -m pip install -r requirements.txt PySide6-Essentials matplotlib
  echo     或者设置环境变量 SHKIT_PYTHON 指向你的 pythonw.exe，
  echo     例如： set SHKIT_PYTHON=D:\Python312\pythonw.exe
  echo     想看完整报错请改用「启动SHKit-查看报错.bat」。
  pause
)

:done
popd
endlocal
