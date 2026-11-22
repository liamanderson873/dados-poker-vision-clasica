@echo off
cd /d "%~dp0"
py -c "import sys" >nul 2>&1
if errorlevel 1 (
  if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    "%LocalAppData%\Programs\Python\Python313\python.exe" programa.py
  ) else (
    echo No se encuentra Python. Instala Python y las dependencias de requirements.txt.
    pause
  )
) else (
  py programa.py
)
if errorlevel 1 pause
