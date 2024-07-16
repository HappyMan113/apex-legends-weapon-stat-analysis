import logging
from typing import Generator

from apex_assistant.checker import check_tuple
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import TranslatedTerm, Translator

logger = logging.getLogger()


class CommandRegistry:
    def __init__(self, *commands: Command):
        check_tuple(Command, allow_empty=False, commands=commands)
        term_set = set(command.get_term() for command in commands)
        if len(term_set) < len(commands):
            raise ValueError('All commands must have unique terms.')

        self._translator = Translator({command.get_term(): command for command in commands})

    def get_commands(self, words: Words) -> Generator[TranslatedTerm[Command], None, None]:
        for command_and_args in self._translator.translate_terms(words):
            logger.debug(f'Command: {command_and_args.get_value()}')
            yield command_and_args
