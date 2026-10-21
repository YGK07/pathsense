@echo off
rem Start the PathSense website on this computer (and for phones on the same Wi-Fi).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Setting up Python packages, first time only...
  python -m venv .venv || goto :nopython
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :eof
)
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:5000"
.venv\Scripts\python.exe app\server.py --lan
pause
goto :eof
:nopython
echo Python 3 was not found. Install it from python.org and run this again.
pause
