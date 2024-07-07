import logging
from threading import Lock
from typing import Generator

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.term import Words
from apex_stat_analysis.speech.term_translator import Translator, ParsedAndFollower

logger = logging.getLogger()


class CommandRegistry:
    INSTANCE: 'CommandRegistry | None' = None
    LOCK = Lock()

    def __init__(self):
        self._translator = Translator()

    def register_command(self, command: Command):
        name = command.get_name()
        if name in self._translator:
            raise RuntimeError(f'Another command is registered as "{name}".')
        self._translator.add_terms({command.get_name(): command})

    @staticmethod
    def get_instance() -> 'CommandRegistry':
        with CommandRegistry.LOCK:
            if CommandRegistry.INSTANCE is None:
                CommandRegistry.INSTANCE = CommandRegistry()

        return CommandRegistry.INSTANCE

    def get_commands(self, words: Words) -> Generator[ParsedAndFollower[Command], None, None]:
        for command_and_args in self._translator.translate_terms(words):
            logger.debug(f'Command: {command_and_args}')
            yield command_and_args
