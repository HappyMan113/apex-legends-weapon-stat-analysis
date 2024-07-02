import logging
import sys

from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.compare_command import CompareCommand
from apex_stat_analysis.speech.speech_client import SpeechClient


logger = logging.getLogger()

def register_commands():
    registry = CommandRegistry.get_instance()
    registry.register_command(CompareCommand())


def setup_logger(level):
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s:%(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def main():
    setup_logger(logging.WARNING)
    register_commands()

    try:
        with SpeechClient() as client:
            # Something in SpeechClient initializer is actually changing the log level; that's one
            # reason we need to change the log level after client initialization. The other reason
            # is that we can avoid logging messages we don't care about.
            logger.setLevel(logging.DEBUG)

            client.start()
    except KeyboardInterrupt:
        sys.exit('Done.')


if __name__ == '__main__':
    main()
