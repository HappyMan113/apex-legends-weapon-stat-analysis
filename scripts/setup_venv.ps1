$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

python .\scripts\python_version_check.py
if ($LASTEXITCODE -ne 0) { exit }

$install_path = $Env:APEX_ASSISTANT_INSTALL_PATH
if (!($install_path)) {
    $install_path = "install"
}
$install_path = "$install_path\Apex Assistant"

$venv_path = "$install_path\venv"
echo "Setting up venv in $install_path"

# Clean.
try {
    Remove-Item -Recurse $venv_path
} catch [System.Management.Automation.ItemNotFoundException] {}

# Create virtual environement.
python -m venv $venv_path
if ($LASTEXITCODE -ne 0) { exit }

# Activate virtual environment.
& "$venv_path\Scripts\Activate.ps1"

# Upgrade pip.
python -m pip install --upgrade pip

# Sanity check that we're in a virtual environment.
pip config set --site install.require-virtualenv true
# Ensure that only the packages that are actually required get installed. This way, the compressed virtual
# environment ends up being under 2 GB and can be attached to a GitHub release.
pip config set --site install.no-deps true
# Setting no-compile to true here doesn't seem to work, even though setting other options this way works. We have
# to resort to using the --no-compile flag instead.
# pip config set --site install.no-compile true

# Install required packages.
# https://download.pytorch.org/whl/cu121 has torch packages.
pip install -r .\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --ignore-requires-python --no-compile
if ($LASTEXITCODE -ne 0) { exit }

.\scripts\install.ps1

echo ''
echo 'Success!'