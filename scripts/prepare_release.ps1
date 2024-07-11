$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

# Install Apex Assistant.
.\scripts\install.ps1

$install_path = $Env:APEX_ASSISTANT_INSTALL_PATH
if (!($install_path)) {
    $install_path = "install"
}
$install_path = "$install_path\Apex Assistant"

$archive_name = "apex-assistant.7z"
try {
    Remove-Item $archive_name
} catch [System.Management.Automation.ItemNotFoundException] {}

echo 'Zipping...'
& "C:\Program Files\7-Zip\7z.exe" a "$install_path\$archive_name" $install_path
if ($LASTEXITCODE -ne 0) { exit }

echo 'Zipped!'