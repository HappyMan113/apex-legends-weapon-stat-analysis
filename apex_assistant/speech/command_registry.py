import logging
from typing import Optional

from apex_assistant.checker import check_tuple
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import TranslatedTerm, Translator

_LOGGER = logging.getLogger()


class CommandRegistry:
    def __init__(self, *commands: Command):
        check_tuple(Command, allow_empty=False, commands=commands)
        term_set = set(command.get_term() for command in commands)
        if len(term_set) < len(commands):
            raise ValueError('All commands must have unique terms.')

        self._translator = Translator({command.get_term(): command for command in commands})

    def process_command(self, words: Words) -> str:
        parsed = self._get_command(words)
        if parsed is None:
            return ''

        command = parsed.get_value()
        arguments = parsed.get_following_words()
        args_str = f' {arguments}' if len(arguments) > 0 else ''
        _LOGGER.info(f'Issuing command: {command}{args_str}')
        message = command.execute(arguments)
        return message

    def _get_command(self, words: Words) -> Optional[TranslatedTerm[Command]]:
        translation = self._translator.translate_terms(words)
        commands = translation.terms()
        if len(commands) == 0:
            _LOGGER.debug(f'No command found in "{words}".')
            return None

        first_command = commands[0]
        starts_with_command = len(first_command.get_preceding_words()) == 0
        if not starts_with_command:
            _LOGGER.debug('Words do not start with command. Skipping.')
            return None

        if len(commands) > 1:
            _LOGGER.warning('Multiple commands specified. Only the first one will be executed.')

        return first_command
