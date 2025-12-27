import logging
from typing import TypeVar

from apex_assistant.speech.command_registry import CommandRegistry
from apex_assistant.speech.term import Words

# Logger for this module.
logger = logging.getLogger()

T = TypeVar('T')


class SpeechClient:
    def __init__(self, command_registry: CommandRegistry):
        self._command_registry = command_registry
        self._closed: bool = False

    def start(self):
        logger.info('Initialized. Type a command.')
        try:
            while not self._closed:
                text: str = input('\n').strip()
                try:
                    words = Words(text)
                    self.process_text(words)
                except ValueError:
                    # It's okay.
                    pass
        finally:
            self.stop()

    def process_text(self, words: Words):
        logger.info(f'Heard: {words}')
        message = self._command_registry.process_command(words)
        logger.info(message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if not hasattr(self, '_closed') or self._closed:
            return
        self._closed = True
        logger.debug('Closed.')

    def __del__(self):
        self.stop()
