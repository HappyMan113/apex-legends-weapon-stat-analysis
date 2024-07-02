[Build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Just get "Desktop development with C++"

# Developer Guide

For installing the latest version of RealtimeTTS, it seems that you need to be on Python 3.11,
not 3.12 (as of 2024-06-29)

## [Install RealtimeSTT](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file)
* `pip install RealtimeSTT`
* `pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`
* Install CUDA
  * [Install NVIDIA Drivers](https://www.nvidia.com/download/index.aspx?lang=en-us)
  * [Install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  * [Install CUDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html)
    * https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic
    * `pip install nvidia-cudnn-cu12`
* ```shell
  winget install Gyan.FFmpeg
  ```

## [Install RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- May have to install from source instead of pip install RealtimeTTS
- Need to be Python <3.12

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

# Usage

Keywords: 