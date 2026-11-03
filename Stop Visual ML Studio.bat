@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports=8000,5173; $ids=Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue ^| Where-Object { $ports -contains $_.LocalPort } ^| Select-Object -ExpandProperty OwningProcess -Unique; foreach ($id in $ids) { if ($id -and $id -ne 0) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue } }"
echo Visual ML Studio has been stopped.
timeout /t 2 /nobreak >nul
