@echo off
REM Second Life AI - double-click to run on Windows. Builds the environment the
REM first time (needs internet once), then serves the site.
cd /d "%~dp0"
set "PYEXE=.venv\Scripts\python.exe"
if exist "%PYEXE%" goto :run
echo [setup] first run - building the Python environment (a few minutes)...
py -3 -m venv .venv 2>nul || python -m venv .venv
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 ^
  || "%PYEXE%" -m pip install torch torchvision
"%PYEXE%" -m pip install timm scikit-learn pandas pyyaml pillow flask
:run
start "" http://127.0.0.1:5001
"%PYEXE%" app.py %*
pause
