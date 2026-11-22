@echo off
rem GGS starter for Windows: installs Python 3.13, the libraries and the motion library into .venv.
rem   setup.cmd                    everything except the AI models (the app offers them on first start)
rem   setup.cmd --models required  also download the AI models this PC needs (about 11 GB with an NVIDIA GPU)
rem Running it again repairs or updates the install. Then start the app:  .venv\Scripts\pythonw -m ggs.ui.app
setlocal
cd /d "%~dp0"
set "UV="
for /f "usebackq delims=" %%U in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\get_uv.ps1"`) do set "UV=%%U"
if not defined UV (
  echo Could not get uv, the tool that installs Python. Check the internet connection and run setup.cmd again.
  exit /b 1
)
"%UV%" run --no-project --python 3.13 "%~dp0tools\setup_env.py" %*
exit /b %ERRORLEVEL%
