import logging
from typing import Callable, TypeAlias

from apex_assistant.checker import check_type
from apex_assistant.legend import Legend
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech import apex_terms
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.term import RequiredTerm, Term, Words
from apex_assistant.speech.term_translator import TranslatedTerm, Translator


_LOGGER = logging.getLogger()
_METHOD: TypeAlias = Callable[[Words], str]


class ConfigureCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(apex_terms.CONFIGURE,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)
        legend = Term('legend')
        log_level = Term('log level', 'logging', 'logging level')
        self._defaults_translator: Translator[_METHOD] = Translator({
            legend: self._parse_legend_term,
            log_level: self._parse_log_level_term,
        })
        self._legend_translator = Translator[Legend]({
            Term(legend): legend
            for legend in Legend
        })
        self._log_level_translator: Translator[int] = Translator({
            Term('debug', 'verbose', 'trace'): logging.DEBUG,
            Term('info'): logging.INFO,
            Term('warn', 'warning'): logging.WARNING,
            Term('error'): logging.ERROR,
            Term('critical', 'quiet'): logging.CRITICAL
        })

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

    def _parse_legend_term(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)
        translation = self._legend_translator.translate_terms(arguments)
        if len(translation) > 1:
            return 'Must specify only one legend.'
        legend = translation.get_latest_value()
        return self._set_legend(legend)

    def _set_legend(self, legend: Legend) -> str:
        old_legend = self.get_translator().set_legend(legend)
        _LOGGER.info(f'Set legend to {legend} (was {old_legend}).')
        if old_legend is legend:
            return f'Legend was already set to {legend}.'
        return f'Set legend to {legend}. Was {old_legend}.'

    @staticmethod
    def _parse_argument(argument: TranslatedTerm[_METHOD]) -> str:
        method = argument.get_value()
        return method(argument.get_following_words())
