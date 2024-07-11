SET XDG_CACHE_HOME=%~dp0.cache
SET HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%~dp0venv\Scripts\python" -W ignore "%~dp0venv\Scripts\apex-assistant.exe"
if not errorlevel 1 goto end
pause
:end
