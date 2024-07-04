import logging
import os
import sys

from pydub.utils import which

from apex_stat_analysis.speech.best_command import BestCommand
from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.compare_command import CompareCommand
from apex_stat_analysis.speech.speech_client import SpeechClient


logger = logging.getLogger()

def register_commands():
    registry = CommandRegistry.get_instance()
    registry.register_command(CompareCommand())
    registry.register_command(BestCommand())


def setup_logger(level):
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
    setup_logger(logging.WARNING)
    register_commands()

    try:
        with SpeechClient() as client:
            # Something in SpeechClient initializer is actually changing the log level; that's one
            # reason we need to change the log level after client initialization. The other reason
            # is that we can avoid logging messages we don't care about.
            logger.setLevel(logging.INFO)

            client.start()
    except KeyboardInterrupt:
        logger.info('Done.')
        sys.exit(0)


if __name__ == '__main__':
    main()
