import logging
from typing import Callable, Optional, TypeAlias

from apex_assistant.checker import check_type
from apex_assistant.speech import apex_terms
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.term import RequiredTerm, Term, Words
from apex_assistant.speech.term_translator import TranslatedTerm, Translator
from apex_assistant.weapon import ConcreteWeapon
from apex_assistant.weapon_comparer import WeaponComparer
from apex_assistant.weapon_translator import WeaponTranslator


_LOGGER = logging.getLogger()
_METHOD: TypeAlias = Callable[[Words], str]


class ConfigureCommand(ApexCommand):
    def __init__(self, weapon_translator: WeaponTranslator, weapon_comparer: WeaponComparer):
        super().__init__(apex_terms.CONFIGURE,
                         weapon_translator=weapon_translator,
                         weapon_comparer=weapon_comparer)
        log_level = Term('log level', 'logging', 'logging level')
        self._defaults_translator: Translator[_METHOD] = Translator({
            apex_terms.WITH_RELOAD: self._parse_with_reload_term,
            apex_terms.WITHOUT_RELOAD: self._parse_without_reload_term,
            apex_terms.SIDEARM: self._parse_with_sidearm_term,
            log_level: self._parse_log_level_term,
        })
        self._bool_translator: Translator[bool] = Translator({apex_terms.TRUE: True,
                                                              apex_terms.FALSE: False})
        self._log_level_translator: Translator[int] = Translator({
            Term('debug', 'verbose', 'trace'): logging.DEBUG,
            Term('info'): logging.INFO,
            Term('warn', 'warning'): logging.WARNING,
            Term('error'): logging.ERROR,
            Term('critical', 'quiet'): logging.CRITICAL
        })

    def _parse_with_reload_term(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)
        values: set[bool] = {term.get_value()
                             for term in self._bool_translator.translate_terms(arguments)}
        if len(values) > 1:
            _LOGGER.debug('More than one boolean value specified for reload by default.')
            return 'Must specify either true or false for reload by default'
        reload_by_default = next(iter(values), True)
        return self._set_reload_by_default(reload_by_default)

    def _parse_without_reload_term(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)
        return self._set_reload_by_default(False)

    def _set_reload_by_default(self, reload_by_default: bool) -> str:
        old_value = self.get_translator().set_reload_by_default(reload_by_default)
        _LOGGER.info(f'Set reload_by_default to {reload_by_default} (was {old_value}).')
        return f'Set reload by default to {reload_by_default}. Was {old_value}.'

    def _parse_with_sidearm_term(self, arguments: Words) -> str:
        translator = self.get_translator()
        if len(arguments) == 0:
            sidearm = translator.get_default_sidearm()
            _LOGGER.info(f'Default sidearm is {sidearm}).')
            return f'Current default sidearm is {self._get_term(sidearm)}.'

        if apex_terms.NONE.find_term(arguments):
            old_sidearm = translator.set_default_sidearm(None)
            _LOGGER.info(f'Set default sidearm to None (was {old_sidearm}).')
            return f'Set default sidearm to none. Was {self._get_term(old_sidearm)}.'

        sidearm = translator.translate_concrete_weapon(arguments)
        if sidearm is None:
            _LOGGER.debug(f'Could not find concrete weapon for "{arguments}".')
            return f'No weapon found named {arguments}.'

        old_sidearm = translator.set_default_sidearm(sidearm)
        _LOGGER.info(f'Set default sidearm to {sidearm} (was {old_sidearm}).')
        return (f'Set default sidearm to {self._get_term(sidearm)}. Was'
                f' {self._get_term(old_sidearm)}.')

    @staticmethod
    def _get_term(weapon: Optional[ConcreteWeapon]) -> str:
        check_type(ConcreteWeapon, optional=True, weapon=weapon)
        return weapon.get_term().to_audible_str() if weapon is not None else 'none'

    def _parse_log_level_term(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)
        values: dict[int, RequiredTerm] = {
            term.get_value(): term.get_term()
            for term in self._log_level_translator.translate_terms(arguments)}
        if len(values) > 1:
            _LOGGER.debug('More than one log level specified.')
            return 'Must specify only one log level.'
        elif len(values) == 0:
            return f'Current logging level is {logging.getLevelName(_LOGGER.getEffectiveLevel())}.'
        log_level, term = next(iter(values.items()))
        _LOGGER.setLevel(log_level)
        _LOGGER.info(f'Set log level to {logging.getLevelName(log_level)}.')
        return f'Set log level to {term.to_audible_str()}.'

    def _execute(self, arguments: Words) -> str:
        results: dict[_METHOD, TranslatedTerm[_METHOD]] = {}
        for translated_term in self._defaults_translator.translate_terms(arguments):
            value = translated_term.get_value()
            if value not in results:
                results[value] = translated_term
        actions: tuple[str] = tuple(self._parse_argument(argument)
                                    for argument in results.values())
        result = ' '.join(actions) if len(actions) > 0 else 'Nothing was configured.'
        return result

    @staticmethod
    def _parse_argument(argument: TranslatedTerm[_METHOD]) -> str:
        method = argument.get_value()
        return method(argument.get_following_words())
