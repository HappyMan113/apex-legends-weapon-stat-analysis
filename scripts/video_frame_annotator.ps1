$ErrorActionPreference = 'Stop'
cd $PSScriptRoot

# Install Apex Assistant.
python -m pip install ffmpeg-python
if ($LASTEXITCODE -ne 0) { exit }

# Copy setup script into package.
python .\python\video_frame_annotator.py $args