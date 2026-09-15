@echo off
rem ===================================================================
rem  SHKit GUI launcher, WITH console.
rem  Use this one when the GUI does not start: it keeps the window open
rem  and prints the traceback.
rem
rem  Keep this file ASCII-only: cmd.exe reads .bat in the OEM code page
rem  (GBK on a Chinese Windows) and would mis-parse UTF-8 Chinese here.
rem  Interpreter lookup order: SHKIT_PYTHON, Anaconda under %USERPROFILE%,
rem  the py launcher, then python on PATH.
rem ===================================================================
setlocal
pushd "%~dp0"

set "PY="
if defined SHKIT_PYTHON if exist "%SHKIT_PYTHON%" set "PY=%SHKIT_PYTHON%"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python" for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined PY if exist "%%~fD\python.exe" set "PY=%%~fD\python.exe"

echo ===============================================================
echo   Starting SHKit GUI ...
if defined PY (
  echo   Python: %PY%
) else (
  echo   Python: py -3  ^(from PATH^)
)
echo ===============================================================
echo.

if defined PY (
  "%PY%" -m shkit.gui %*
) else (
  py -3 -m shkit.gui %*
)

echo.
echo ---------------------------------------------------------------
echo   The GUI has exited (exit code = %ERRORLEVEL%).
echo   If there is a traceback above, please copy the whole block.
echo ---------------------------------------------------------------
pause
popd
endlocal
