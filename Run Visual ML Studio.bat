@echo off
setlocal
set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"
set "PYTHON=C:\Python314\python.exe"

if not exist "%PYTHON%" set "PYTHON=python"

echo Starting Visual ML Studio...

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; try { $null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2; $ok=$true } catch {}; if (-not $ok) { Start-Process cmd.exe -ArgumentList '/k','cd /d ""%BACKEND%"" ^&^& ""%PYTHON%"" -m uvicorn app.main:app --host 127.0.0.1 --port 8000' -WindowStyle Minimized }"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; try { $null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/ -TimeoutSec 2; $ok=$true } catch {}; if (-not $ok) { Start-Process cmd.exe -ArgumentList '/k','cd /d ""%FRONTEND%"" ^&^& npm run dev -- --host 127.0.0.1' -WindowStyle Minimized }"

echo Waiting for the application...
set /a attempts=0
:wait
set /a attempts+=1
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2; $null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/ -TimeoutSec 2; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto ready
if %attempts% GEQ 30 goto failed
timeout /t 1 /nobreak >nul
goto wait

:ready
echo Visual ML Studio is ready.
start "" "http://127.0.0.1:5173/"
exit /b 0

:failed
echo Could not start Visual ML Studio.
echo Check the minimized Backend and Frontend windows for the error.
pause
exit /b 1
