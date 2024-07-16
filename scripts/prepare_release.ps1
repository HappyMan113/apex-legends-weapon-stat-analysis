$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

# Install Apex Assistant.
.\scripts\install.ps1

$install_path = $Env:APEX_ASSISTANT_INSTALL_PATH
if (!($install_path)) {
    $install_path = "install"
}
$archive_path = "$install_path\apex-assistant.7z"
$install_path = "$install_path\Apex Assistant"

try {
    Remove-Item $archive_path
} catch [System.Management.Automation.ItemNotFoundException] {}

echo 'Zipping...'
& "C:\Program Files\7-Zip\7z.exe" a $archive_path $install_path -mx9 -mmt=4 "-xr!.cache" "-xr!*.log" "-xr!*.json" "-w$archive_path\.."
if ($LASTEXITCODE -ne 0) { exit }

echo 'Zipped!'