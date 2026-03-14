@echo off
echo =======================================================
echo 🏏 ICC T20 World Cup Predictor - Starting Services...
echo =======================================================

:: Start FastAPI Backend
echo [1/2] Starting FastAPI Backend on Port 8000...
start "ICC T20 - FastAPI Backend" cmd /c "uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait 3 seconds for the API to initialize
timeout /t 3 /nobreak >nul

:: Start Streamlit Dashboard
echo [2/2] Starting Streamlit Dashboard...
start "ICC T20 - Streamlit Dashboard" cmd /c "streamlit run dashboards/streamlit_app.py"

echo.
echo ✅ Services started successfully in new windows!
echo - API Docs: http://127.0.0.1:8000/docs
echo - Dashboard: Automatically opening in your browser...
echo.
echo Press any key to exit this launcher...
pause >nul
