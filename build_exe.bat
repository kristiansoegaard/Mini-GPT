@echo off
cd /d "%~dp0"
echo ==== Installing tools (can take a few minutes) ====
python -m pip install --upgrade pyinstaller torch tiktoken
if errorlevel 1 goto fejl

set ICONARGS=
if exist mini-gpt.ico set ICONARGS=--icon mini-gpt.ico --add-data "mini-gpt.ico;."

echo ==== Building mini-gpt.exe (can take 5-10 minutes) ====
python -m PyInstaller --noconfirm --onedir --windowed --name mini-gpt --collect-all tiktoken --hidden-import tiktoken_ext.openai_public %ICONARGS% ai_app.py
if errorlevel 1 goto fejl

set FOUND=0
if exist mini_chat_gpt.pth (
    copy /Y mini_chat_gpt.pth dist\mini-gpt\ >nul
    set FOUND=1
)
if exist mini_code_gpt.pth (
    copy /Y mini_code_gpt.pth dist\mini-gpt\ >nul
    set FOUND=1
)
if "%FOUND%"=="0" echo WARNING: no .pth model found. Copy mini_chat_gpt.pth into dist\mini-gpt\ yourself.

echo.
echo ==== DONE ====
echo Your program: dist\mini-gpt\mini-gpt.exe
pause
exit /b 0

:fejl
echo.
echo Something went wrong. Read the error above.
pause
exit /b 1