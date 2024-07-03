
# Prerequisites
- [Install NVIDIA Drivers](https://www.nvidia.com/download/index.aspx?lang=en-us)
<!--
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file)
  - Install CUDA
    - [Install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
    - [Install CUDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html)
      - https://developer.nvidia.com/cudnn-downloads
      - `pip install nvidia-cudnn-cu12`
-->
- Install FFMpeg
  ```shell
  winget install Gyan.FFmpeg
  ```

## OpenAI API key
- [Create a Project Key](https://platform.openai.com/api-keys)
- Set environment variable `OPENAI_API_KEY`
- [Add funds](https://platform.openai.com/settings/organization/billing/overview) if needed.

<!--
## Get your Azure API keys:
- [Azure Portal](https://portal.azure.com/?quickstart=true#home)
- [Guide to creating API keys](https://docs.merkulov.design/how-to-get-microsoft-azure-tts-api-key/)

## Get ElevenLabs
- `python -m pip install --upgrade elevenlabs==0.2.27`
- `python -m pip install --upgrade elevenlabs==1.2.2`
- Get your ElevenLabs API key: https://elevenlabs.io/app/speech-synthesis
    - Get MPV https://sourceforge.net/projects/mpv-player-windows/
    - Run bootstrapper
- Run install script
- Add install location with exe to your Path
-->

# Developer Guide
<!--
## [Build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Just get "Desktop development with C++"
-->

## Set up Virtual Environment
```shell
.\scripts\install.ps1
```

<!--
## Install [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- May have to install from source instead of pip install RealtimeTTS to get the latest version

## Create Standalone Executable
- `pip install nuitka`
- `python -m nuitka --standalone ./apex_stat_analysis/main.py --include-package-data=apex_stat_analysis --noinclude-numba-mode=nofollow --noinclude-custom-mode=transformers:nofollow` 
- 
- `pip install py2exe`
- 
- `pip install pypiwin32 pyinstaller`
- https://github.com/mhammond/pywin32/releases
- `python -m pip install pywin32 --upgrade`
- `pip install --upgrade pvporcupine`
- `pip uninstall enum34`
- https://www.gtk.org/docs/installations/ ?
- ```shell
  pyinstaller -c -F ./apex_stat_analysis/main.py -n apex-assistant --collect-data apex_stat_analysis --clean --collect-data pvporcupine --collect-binaries azure
  ```
-->

## Re-Installation of Apex Assistant Package Only
```shell
.\package\venv\Scripts\python -m pip install .
```
### Editable Install
```shell
.\package\venv\Scripts\python -m pip install -e .
```

# Usage
```shell
.\package\apex-assistant.bat
```