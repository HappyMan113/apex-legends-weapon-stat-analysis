$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

$install_path = $Env:APEX_ASSISTANT_INSTALL_PATH
if (!($install_path)) {
    $install_path = "install"
}
$install_path = "$install_path\Apex Assistant"

# Install Apex Assistant.
& "$install_path\venv\Scripts\pip" install . $e
if ($LASTEXITCODE -ne 0) { exit }

# Copy setup script into package.
Copy-Item ".\scripts\apex-assistant.bat" $install_path