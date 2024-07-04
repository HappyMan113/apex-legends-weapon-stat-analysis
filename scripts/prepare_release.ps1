$ErrorActionPreference = 'Stop'
cd "$PSScriptRoot\.."

$package_path = ".\package"

# Install Apex Assistant.
& "$package_path\venv\Scripts\pip" install .
if ($LASTEXITCODE -ne 0) { exit }

# Copy setup script into package.
Copy-Item .\scripts\apex-assistant.bat $package_path

$archive_name = "apex-assistant.7z"
try {
    Remove-Item $archive_name
} catch [System.Management.Automation.ItemNotFoundException] {}

echo 'Zipping...'
& "C:\Program Files\7-Zip\7z.exe" a $archive_name "$package_path\*"
if ($LASTEXITCODE -ne 0) { exit }

echo 'Zipped!'