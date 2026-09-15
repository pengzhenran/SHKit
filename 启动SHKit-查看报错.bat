@echo off
rem ===================================================================
rem  SHKit GUI launcher, WITH console.
rem  Use this one when the GUI does not start: it keeps the window open
rem  and prints the traceback.
rem
rem  Keep the lines above the `chcp` call ASCII-only: cmd.exe parses a
rem  .bat file in the OEM code page until `chcp 65001` takes effect.
rem ===================================================================
chcp 65001 >nul
setlocal
pushd "%~dp0"

set "PY=C:\Users\pengzhenran\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"

echo ===============================================================
echo   正在启动 SHKit 图形界面 ...
echo   使用的 Python: %PY%
echo ===============================================================
echo.

"%PY%" -m shkit.gui %*

echo.
echo ---------------------------------------------------------------
echo   程序已退出 (exit code = %ERRORLEVEL%)
echo   若上面有报错，请把整段内容发给我。
echo ---------------------------------------------------------------
pause
popd
endlocal
