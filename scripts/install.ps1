$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

python .\scripts\python_version_check.py
if ($LASTEXITCODE -ne 0) { exit }

$package_path = "package"
$venv_path = "$package_path\venv"

# Clean.
try {
    Remove-Item -Recurse $package_path
} catch [System.Management.Automation.ItemNotFoundException] {}

# Create virtual environement.
python -m venv "$venv_path"
if ($LASTEXITCODE -ne 0) { exit }

# Activate virtual environment.
& "$venv_path\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) { exit }

# Install RealtimeSTT and RealtimeTTS without installing required packages yet.
pip install -r .\requirements-speech.txt --ignore-requires-python --no-deps --require-virtualenv --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { exit }

# setuptools & wheel are undeclared dependencies of nvidia-pyindex.
pip install setuptools wheel --require-virtualenv --disable-pip-version-check

# Install required packages.
# https://download.pytorch.org/whl/cu121 has torch packages.
# https://pypi.ngc.nvidia.com has NVIDIA packages.
pip install -r .\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.ngc.nvidia.com --require-virtualenv --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { exit }

# Install apex assistant.
pip install . --require-virtualenv --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { exit }

# Copy setup script into package.
Copy-Item .\scripts\apex-assistant.bat $package_path

echo ''
echo 'Success!'