@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo Start Streamlit + Public Link
echo ========================================
echo.

set "CLOUDFLARED_CMD=cloudflared"
where cloudflared >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
        set "CLOUDFLARED_CMD=C:\Program Files (x86)\cloudflared\cloudflared.exe"
    ) else if exist "C:\Program Files\cloudflared\cloudflared.exe" (
        set "CLOUDFLARED_CMD=C:\Program Files\cloudflared\cloudflared.exe"
    ) else (
        echo ERROR: cloudflared is not installed.
        echo Install with:
        echo winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
        echo.
        pause
        exit /b 1
    )
)

echo Starting Streamlit app on http://localhost:8501 ...
start "Streamlit App" cmd /k "cd /d %CD% && python -m streamlit run app.py"

echo Waiting for app startup...
timeout /t 6 /nobreak >nul

echo Starting Cloudflare public tunnel...
start "Cloudflare Tunnel" cmd /k "cd /d %CD% && call "%CLOUDFLARED_CMD%" tunnel --url http://localhost:8501"

echo.
echo Open the "Cloudflare Tunnel" window and copy the https://*.trycloudflare.com link.
echo Keep both windows open while sharing the app.
echo.
pause
