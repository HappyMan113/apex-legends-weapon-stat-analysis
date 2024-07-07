import logging
import os
import sys

from pydub.utils import which

from apex_stat_analysis.speech.best_command import BestCommand
from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.compare_command import CompareCommand
from apex_stat_analysis.speech.speech_client import SpeechClient
from apex_stat_analysis.weapon_database import ApexDatabase


logger = logging.getLogger()

def register_commands():
    registry = CommandRegistry.get_instance()
    registry.register_command(CompareCommand())
    registry.register_command(BestCommand())


def set_up_logger(level):
    logger.handlers.clear()
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s:%(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def ensure_ffmpeg_installed():
    if which("ffmpeg"):
        return

    print('FFMpeg not found. Installing...')
    os.system('winget install Gyan.FFmpeg')
    print('===================================================================')
    print('FFMPEG installed. Please restart your shell and rerun this program.')
    print('===================================================================')
    sys.exit(0)


def main():
    ensure_ffmpeg_installed()
    log_level = logging.INFO
    set_up_logger(log_level)
    register_commands()

    # Load everything in.
    ApexDatabase.get_instance()

    try:
        with SpeechClient() as client:
            client.start()
    except KeyboardInterrupt:
        logger.info('Done.')
        sys.exit(0)
