import logging

from apex_stat_analysis.speech.apex_terms import BEST, NUMBER_TERMS
from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.term import Words
from apex_stat_analysis.speech.term_translator import Translator
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()

class BestCommand(Command):
    def __init__(self):
        super().__init__(BEST)

        numbers = NUMBER_TERMS
        self._number_translator = Translator[int]({
            number_term: int(number_term)
            for number_term in numbers})

    def _execute(self, arguments: Words) -> str:
        numbers = set(self._number_translator.translate_terms(arguments))
        if len(numbers) > 1:
            return 'Too many numbers specified. Only specify one.'
        if len(numbers) == 0:
            vals = self._number_translator.values()
            min_val = min(vals)
            max_val = max(vals)
            return f'Must specify a number between {min_val} and {max_val}'
        number = next(num.get_parsed() for num in numbers)

        LOGGER.debug(f'Getting {number} best weapons.')
        comparison_result = \
            ApexDatabase.get_instance().compare_all_weapons(reload=True).limit_to_best_num(number)
        LOGGER.info(f'Best {number} weapons:\n'
                    f'  {comparison_result.get_archetypes()}')
        audible_names = ' '.join([weapon_archetype.get_name()
                                  for weapon_archetype in comparison_result.get_archetypes()])
        return audible_names
