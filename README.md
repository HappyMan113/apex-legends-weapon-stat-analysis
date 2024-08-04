# About

The Apex Assistant client allows you to ask it to compare two or more weapons from the video game
[Apex Legends](https://www.ea.com/games/apex-legends) via the "compare" voice command. It will then
respond with a synthetic voice telling you which of the compared weapons will in theory give you the
highest expected average damage ser second in close quarters combat.

## Expected Mean DPS Calculation Algorithm

1) For any given weapon, the assistant calculates the mean DPS up till time "t", assuming perfectly
   accurate hits to an opponent's body. It does this for each "t" in a set of historic TTK values.
    - These historic TTK values were derived from clips of my in-game deaths. They were calculated
      starting at the time I first started taking damage and ending when I died, except that if
      there was a long enough time gap to reset before I died, that start of my taking damage did
      not count.
    - Weapon stats were compiled from various sources and used to calculate cumulative damage up
      till
      time "t". Sources were as follows:
        - [Apex Legends Wiki](https://apexlegends.fandom.com/wiki/Weapon#General)
        - [S20 patch notes](https://www.ea.com/games/apex-legends/news/breakout-patch-notes)
        - [S21 patch notes](https://www.ea.com/games/apex-legends/news/upheaval-patch-notes)
        - [S21 mid-season patch notes](https://www.ea.com/games/apex-legends/news/double-take-collection-event)
        - [Reddit comment with the new G7 Scout fire rate](https://www.reddit.com/r/apexlegends/comments/1dwbf4o/comment/lbtsc84/)
    - Reload times are assumed to be tactical reload times with only 1 bullet left in the magazine.
    - For Hammerpoint and Disruptor hop-ups, the DPS is calculated as the average of the DPS when
      no bonus damage is applied and the DPS when the bonus damage does apply. Essentially a 50/50
      split for shielded/unshielded shots.
2) The mean of these mean values is then calculated.
3) This becomes the metric which determines which weapons are "best".

## Speech-to-Text and Text-to-Speech

- Speech-to-Text is done locally through faster-whisper, which is a reimplementation of OpenAI's
  whisper model. Ideally you will be able to do this with a GPU. See prerequisites below.
- Text-to-Speech is done through OpenAI's tts-1 model, with a fallback to being done locally. See
  Prerequisites below.

# Prerequisites

- Must be on Windows.
- Must have a microphone.
- Must set up OpenAI API key for text-to-speech to be done using the OpenAI's tts-1 engine as
  opposed to TTS being done locally on your machine.
    - [Create a Project Key](https://platform.openai.com/api-keys)
    - Set environment variable `OPENAI_API_KEY` to your project key.
    - [Add funds](https://platform.openai.com/settings/organization/billing/overview) if needed (I
      wasn't able to make API requests for free, but maybe you'll have better luck).
    - Must have an internet connection to be able to make OpenAI API requests.
- Must have NVIDIA CUDA GPU to allow GPU-based Speech-to-Text to work on your machine (CPU-based STT
  may slow down your machine, which wouldn't be ideal for playing Apex).
    - Make sure you've installed
      [NVIDIA GPU Drivers](https://www.nvidia.com/download/index.aspx?lang=en-us)
- Install FFMpeg
  ```shell
  winget install Gyan.FFmpeg
  ```
- Install 7-Zip.
  ```shell
  winget install -e --id 7zip.7zip
  ```
- (Recommended)
  [Enable developer mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
  to allow for symbolic links to work which is what's ideal when using the distil-large-v3 model. If
  you leave developer mode disabled, the slower large-v2 model will be used.

See the
[RealtimeSST README](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file#steps-that-might-be-necessary-before)
for more information.

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

Head over to the
[Releases](https://github.com/HappyMan113/apex-legends-weapon-stat-analysis/releases) page and
download apex-assistant.7z. Extract the archive using [7-Zip](https://www.7-zip.org/download.html).

## Usage

Run `apex-assistant.bat` which you just extracted. Wait for audio recording to start.

### Voice Command Syntax

Filler words are ignored, and a lot of variations are allowed. However, the documented syntax is as
follows:

| Command | Description                                                                                                                                                                                                                                                                                                                                                           | Syntax                                                | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Compare | Tells you loadout with the highest expected mean DPS...<ul><li>(if no arguments are specified) ...among all available loadouts.</li><li>(if exactly one weapon is specified) ...among all loadouts containing the given weapon</li><li>(in all other cases) ...among all loadouts that contain only the given weapons, loadouts, and/or classes of weapons.</li></ul> | `COMPARE [<<weapon\|class> [AND <weapon\|class>]]...` | <ul><li>"Compare"</li><li>"Compare volt"</li><li>"Compare wingman with peacekeeper"</li><li>"Compare RE-45 with purple mag with P2020 with level 1 light mag with hammerpoint"</li><li>"Compare flatline level 2 mag no stock sidearm RE-45 same main sidearm Wingman"</li><li>"Compare R-99 with level 2 stock with Flatline with level 1 extended heavy magazine"</li><li>"Compare R-301 level 2 sidearm wingman boosted loader Flatline with no mag purple standard stock"</li><li>"Compare peacekeeper and RE-45"</li><li>"Compare shotgun"</li><li>"Compare Marksman or Sniper"</li><li>"Compare Flatline or Prowler"</li><li>"Compare Flatline with blue mag, RE-45 with level 1 mag with hammerpoint, or Prowler with level 2 mag"</li></ul> | 

Argument syntaxes are further broken down as follows:

| Argument   | Syntax                                                                                                                                                                                                                                                 |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `<weapon>` | <ul><li><code>[BASE\|LEVEL &lt;number&gt;\|FULLY KITTED] &lt;weapon name&gt; [&lt;&lt;LEVEL &lt;level&gt;\|NO&gt; &lt;attachment name&gt;&gt;\|&lt;hop-up name&gt;]...</code></li>--- OR ---<li>"SAME MAIN"</li>--- OR ---<li>"SAME SIDEARM"</li></ul> |
| `<class>`  | `<AR\|LMG\|MARKSMAN\|PISTOL\|SHOTGUN\|SMG\|SNIPER>`                                                                                                                                                                                                    |

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
## Create Standalone Executable
- `pip install nuitka`
- `python -m nuitka --standalone ./apex_assistant/main.py --include-package-data=apex_assistant --noinclude-numba-mode=nofollow --noinclude-custom-mode=transformers:nofollow` 
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
  pyinstaller -c -F ./apex_assistant/main.py -n apex-assistant --collect-data apex_assistant --clean --collect-data pvporcupine --collect-binaries azure
  ```
-->
