@echo off
setlocal

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

python -m PyInstaller --onefile --noconsole --name "FVTT-Journal-to-PDF" app_with_dividers.py

echo.
echo Build complete!
echo EXE is in the dist folder.
pause
