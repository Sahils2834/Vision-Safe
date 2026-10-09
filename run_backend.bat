@echo off
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
  echo Backend environment is missing. Run the setup commands in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8000
