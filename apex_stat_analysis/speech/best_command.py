import logging

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.compare_command import CompareCommand
from apex_stat_analysis.speech.term_translator import ApexTranslator
from apex_stat_analysis.speech.terms import ApexTerms, IntTerm, Words
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()

class BestCommand(Command):
    def __init__(self):
        super().__init__(ApexTerms.BEST)

        numbers = (IntTerm(1, 'one'),
                   IntTerm(2, 'two'),
                   IntTerm(3, 'three'),
                   IntTerm(4, 'four'),
                   IntTerm(5, 'five'),
                   IntTerm(6, 'six'),
                   IntTerm(7, 'seven'),
                   IntTerm(8, 'eight'),
                   IntTerm(9, 'nine'),
                   IntTerm(10, 'ten'))
        self._number_translator = ApexTranslator({
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
        number = next(iter(numbers))

        LOGGER.debug(f'Getting {number} best weapons.')
        comparison_result = \
            ApexDatabase.get_instance().compare_all_weapons(reload=True).limit_to_best_num(number)
        LOGGER.info(f'Best {number} weapons:\n'
                    f'  {comparison_result.get_archetypes()}')
        audible_names = ' '.join([CompareCommand.make_audible(weapon_archetype)
                                  for weapon_archetype in comparison_result.get_archetypes()])
        return audible_names
