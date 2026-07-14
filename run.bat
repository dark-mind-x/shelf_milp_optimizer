@echo off
echo Starting milp shelf Optimizer...
echo Please wait, this takes a few seconds.
echo.

cd /d "%~dp0"

call venv\Scripts\activate.bat

start http://localhost:8501

streamlit run app.py

pause