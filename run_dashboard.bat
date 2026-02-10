@echo off
echo =========================================
echo BRENT OIL ANALYSIS DASHBOARD
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/3] Installing backend dependencies...
cd dashboard\backend
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install backend dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Installing frontend dependencies...
cd ..\frontend
npm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Starting servers...
echo.

REM Start backend in new window
start "Backend Server" cmd /k "cd dashboard\backend && python app.py"

REM Wait for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend in new window
start "Frontend Server" cmd /k "cd dashboard\frontend && npm start"

echo.
echo =========================================
echo DASHBOARD STARTED SUCCESSFULLY!
echo =========================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to stop all servers...
pause >nul

REM Kill all processes
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

echo.
echo Dashboard stopped.
pause
