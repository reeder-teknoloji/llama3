@echo off
cd /d "%~dp0"

REM --- Force localhost distributed init (Windows 10049 fix) ---
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500
set RANK=0
set WORLD_SIZE=1
set LOCAL_RANK=0

if not exist ".venv\Scripts\python.exe" (
  echo [HATA] .venv bulunamadi.
  echo Lutfen once:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "knowledge\reev_technical_index.json" (
  echo [HATA] knowledge\reev_technical_index.json bulunamadi.
  echo Lutfen once index olusturun:
  echo   .venv\Scripts\python.exe ingest.py --input_files "C:\Users\Efehan Admin\Downloads\ReeV Fancy Teknik.pdf" --output_path knowledge\reev_technical_index.json
  pause
  exit /b 1
)

echo Uygulama baslatiliyor...
echo.
echo   Localhost  : http://localhost:8501
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP: =%
echo   Ag Erisimi : http://%IP%:8501
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.headless true
