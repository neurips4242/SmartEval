@echo off
REM Launch demo automatically with translation API

echo ========================================
echo IBM Agentics - Smart Contract Translator
echo Research & Quality Evaluation Demo
echo ========================================
echo.
echo Starting translation API and demo...
echo.

python launch_demo.py

if errorlevel 1 (
    echo.
    echo Error starting demo. Make sure Python is installed.
    pause
)
