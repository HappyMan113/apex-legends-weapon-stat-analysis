$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

$install_path = $Env:APEX_ASSISTANT_INSTALL_PATH
if (!($install_path)) {
    $install_path = "install"
}

# Install Apex Assistant.
& "$install_path\venv\Scripts\pip" install .
if ($LASTEXITCODE -ne 0) { exit }

# Copy setup script into package.
$start_script_name = "apex-assistant.bat"
Copy-Item ".\scripts\$start_script_name" $install_path

$archive_name = "apex-assistant.7z"
try {
    Remove-Item $archive_name
} catch [System.Management.Automation.ItemNotFoundException] {}

echo 'Zipping...'
& "C:\Program Files\7-Zip\7z.exe" a "$install_path\$archive_name" "$install_path\venv" "$install_path\$start_script_name"
if ($LASTEXITCODE -ne 0) { exit }

echo 'Zipped!'