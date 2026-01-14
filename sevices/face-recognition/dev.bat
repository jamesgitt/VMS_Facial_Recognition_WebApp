@echo off
REM Quick development startup script for Windows

echo ========================================
echo  Face Recognition API - Dev Mode
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found.
    echo [INFO] Run: python -m venv venv
    echo.
)

REM Check if models exist
if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo [WARNING] Models not found!
    echo [INFO] Downloading models...
    python app\download_models.py
    if errorlevel 1 (
        echo [ERROR] Failed to download models!
        pause
        exit /b 1
    )
)

echo [INFO] Starting development server with auto-reload...
echo [INFO] API will be available at: http://localhost:8000
echo [INFO] API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the server with reload
python app\main.py --reload

pause
