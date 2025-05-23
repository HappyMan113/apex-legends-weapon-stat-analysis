# About

The Apex Assistant client allows you to ask it to compare two or more weapons from the video game
[Apex Legends](https://www.ea.com/games/apex-legends) via the "compare" voice command. It will then
respond with a synthetic voice telling you which of the compared weapons will in theory give you the
highest expected average damage ser second in close quarters combat.

## Expected Mean DPS Calculation Algorithm

Probability of kill (P(kill)) is calculated for a loadout using binomial distributions against a
sampling of time values taken from a Weibull distribution. This is done for various distances (and
their corresponding accuracy values), and the mean of all of these probability values is considered
the final P(kill) value.

## Algorithm Notes
- Weapon stats were compiled from various sources. Sources were as follows:
  - [Apex Legends Wiki](https://apexlegends.fandom.com/wiki/Weapon#General)
  - [S20 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/breakout-patch-notes)
  - [S21 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/upheaval-patch-notes)
  - [S21 mid-season patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/double-take-collection-event)
  - [S22 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/shockwave-patch-notes)
  - [Reddit comment with the new G7 Scout fire rate](https://www.reddit.com/r/apexlegends/comments/1dwbf4o/comment/lbtsc84/)
  - [S22 mid-season patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/space-hunt-event)
  - [X Post about Mozambique Nerf](https://x.com/Respawn/status/1844427285916680379)
  - [S23 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/from-the-rift-season-updates)
  - [S23 mid-season patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/astral-anomaly-event)
  - [S24 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/from-the-rift-season-updates)
  - [X Post with S24 mid-season patch notes](https://x.com/Respawn/status/1909722102892314681)
  - [S25 patch notes](https://www.ea.com/games/apex-legends/apex-legends/news/prodigy-patch-notes)
  - [X Post with S25 early-season patch notes](https://x.com/Respawn/status/1922450385207455919)
- Reload times are assumed to be tactical reload times with only 1 bullet left in the magazine.
- Swap times were calculated as "Holster Time" + "Ready to Fire" time that I calculated from my
  recordings. The values I got were close to the values in the Reddit post
  [here](https://www.reddit.com/r/apexlegends/comments/13rtny9/the_foundational_flaw_of_apex_legends/)
  which describes Ready to Fire time (referred to as "Draw Time" in the post).
- For <!--Hammerpoint and -->Disruptor hop-ups, the DPS is calculated as the average of the DPS when
  no bonus damage is applied and the DPS when the bonus damage does apply. Essentially a 50/50
  split for shielded/unshielded shots.
- Accuracy at various distances was measured against fully kitted variants of weapons. Dummies
  were standing still, and I (the player) was strafing back and forth to try to take into
  account fire spread. Accuracy was calculated simply as the interpolated value between the two
  nearest accuracy data points.
  - For measurements of accuracy at 10 meter and closer ranges, I used hipfire. 
  - For measurements of accuracy at 160 meter measurements and further, for most weapons I figured
    the fire spread would still be taken into account with me standing still and that the risk of
    dying due to standing still was pretty low, so I stood still and crouched in most cases.

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

| Command               | Description                                                                                                                                                                                                                                                                                                                                                           | Syntax                                                | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Compare               | Tells you loadout with the highest expected mean DPS...<ul><li>(if no arguments are specified) ...among all available loadouts.</li><li>(if exactly one weapon is specified) ...among all loadouts containing the given weapon</li><li>(in all other cases) ...among all loadouts that contain only the given weapons, loadouts, and/or classes of weapons.</li></ul> | `COMPARE [<<weapon\|class> [AND <weapon\|class>]]...` | <ul><li>"Compare"</li><li>"Compare volt"</li><li>"Compare wingman with peacekeeper"</li><li>"Compare RE-45 with purple mag with P2020 with level 1 light mag with hammerpoint"</li><li>"Compare flatline level 2 mag no stock sidearm RE-45 same main sidearm Wingman"</li><li>"Compare R-99 with level 2 stock with Flatline with level 1 extended heavy magazine"</li><li>"Compare R-301 level 2 sidearm wingman boosted loader Flatline with no mag purple standard stock"</li><li>"Compare peacekeeper and RE-45"</li><li>"Compare shotgun"</li><li>"Compare Marksman or Sniper"</li><li>"Compare Flatline or Prowler"</li><li>"Compare Flatline with blue mag, RE-45 with level 1 mag with hammerpoint, or Prowler with level 2 mag"</li></ul> | 
| Configure Legend      | Allows you to take into account which legend you have selected. Only "Rampart" and "none" are currently supported.                                                                                                                                                                                                                                                    | `CONFIGURE LEGEND <legend>...`                        | <ul><li>"Configure legend Rampart"</li><li>"Configure legend none"</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 
| Create Summary Report | Creates a Summary Report CSV File.                                                                                                                                                                                                                                                                                                                                    | `CREATE SUMMARY REPORT`                               | "Create summary report"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 

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
