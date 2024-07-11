<!-- 
TODO: Explain what the assistant does and the methods used to calculate "best" weapons.
# About
## Methods
- Speech-to-Text is done locally through faster-whisper, which is a reimplementation of OpenAI's 
whisper model. Ideally you will be able to do this with a GPU. See prerequisites below.
- Text-to-Speech is done through OpenAI's tts-1 model, with a fallback to being done locally. See 
Prerequisites below.
-->

# Prerequisites
See the
[RealtimeSST README](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file#steps-that-might-be-necessary-before) 
for more information. 
- Must be on Windows.
- Must have a microphone cause how else are you going to use voice commands?
- Enable developer mode for the distil-large-v3 optimal performance.
  https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development 
- Set up OpenAI API key (for text-to-speech)
  - [Create a Project Key](https://platform.openai.com/api-keys)
  - Set environment variable `OPENAI_API_KEY` to your project key.
  - [Add funds](https://platform.openai.com/settings/organization/billing/overview) if needed (I
    wasn't able to make any free requests, but maybe you'll have better luck.)
  - NoteL Must have an internet connection for OpenAI Text-to-Speech engine to work.
- Must have NVIDIA CUDA GPU to allow GPU-based Speech-to-Text to work on your machine (CPU-based STT
may slow down your machine which wouldn't be ideal for playing Apex).
  - Install [NVIDIA GPU Drivers](https://www.nvidia.com/download/index.aspx?lang=en-us)
- Install FFMpeg
  ```shell
  winget install Gyan.FFmpeg
  ```
- Install 7-Zip.
  ```shell
  winget install -e --id 7zip.7zip
  ```

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

## User Installation
Head over to the Releases page and download apex-assistant.7z. Extract the archive using 
[7-Zip](https://www.7-zip.org/download.html).

## Usage
Run `apex-assistant.bat` which you just extracted. Wait for audio recording to start.

### Voice Command Syntax
Note: Filler words between key terms will be ignored, but filler words within key terms with two or
more words may result in that term failing to be translated.
- `compare <weapon name>`

### Example Voice Commands:
- "Compare wingman peacekeeper"
- "Compare RE-45 with purple mag with P2020 with hammerpoint"
- "Compare no reloads R-99 with level 2 stock with Flatline with level 1 extended heavy magazine"
- "Compare R-301 level 2 sidearm wingman boosted loader Flatline with no mag purple standard stock
sidearm peacekeeper with a white shotgun bolt and no standard stock"
- "Best 5"


# Developer Guide
<!--
## [Build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Just get "Desktop development with C++"
-->

## Set up Virtual Environment
This virtual environment will contain all required dependencies. It's the virtual environment that
gets copied into the release archive. It can be used for development & testing, but if you install 
packages for testing purposes only, be sure to uninstall them prior to release. Rerunning
setup_venv.ps1 will accomplish this.
```shell
.\scripts\setup_venv.ps1
```

## Install this package
You'll need to install this package before you can run it. While a regular install is done as part 
of the prepare_release script, you may want to install for development and testing purposes before
preparing a release.

### Regular Install
```shell
.\scripts\install.ps1
```
### Editable Install
```shell
.\scripts\install.ps1 -e
```

## Create Release Archive
<!--
### Prerequisite
# TODO: Create process for creating an installer: http://ntsblog.homedev.com.au/index.php/2015/05/14/self-extracting-archive-runs-setup-exe-7zip-sfx-switch/
Install 7-Zip LZMA SDK:
https://www.7-zip.org/sdk.html

###
Zip up release package.
-->
```shell
.\scripts\prepare_release.ps1
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
