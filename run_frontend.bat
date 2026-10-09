@echo off
cd /d "%~dp0frontend"
if not exist "node_modules\.bin\vite.cmd" (
  echo Frontend packages are missing. Run pnpm install inside the frontend folder first.
  pause
  exit /b 1
)
"node_modules\.bin\vite.cmd" --host 0.0.0.0
