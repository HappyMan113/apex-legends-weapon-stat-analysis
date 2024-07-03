cd "%~dp0\venv\Scripts"
.\python .\apex-assistant.exe
if not errorlevel 1 goto end
pause
:end
