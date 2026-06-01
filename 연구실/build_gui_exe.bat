@echo off
cd /d "%~dp0"
python -m pip install --user pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name SUMO_Net_Generator generate_sumo_net_gui.pyw
echo.
echo EXE output: %~dp0dist\SUMO_Net_Generator.exe
pause
