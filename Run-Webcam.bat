@echo off
cd /d "%~dp0"
python -c "import cv2, sounddevice" 2>nul || (
  echo Installing dependencies...
  python -m pip install -r requirements.txt
)
python app.py
