import logging
import typing

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.compare_command import CompareCommand
from apex_stat_analysis.speech.terms import ApexTermBase, ApexTerms, extract_terms
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()

class BestCommand(Command):
    def __init__(self):
        super().__init__(ApexTerms.BEST)

    def _execute(self, arguments: typing.Iterable[ApexTermBase]) -> str:
        arguments = extract_terms(arguments)
        number_terms = set(arguments) & set(ApexTerms.NUMBERS)
        if len(number_terms) > 1:
            return 'Too many numbers specified. Only specify one.'
        if len(number_terms) == 0:
            return f'Must specify {ApexTerms.NUMBERS}'
        number = int(str(next(iter(number_terms))))

        LOGGER.debug(f'Getting {number} best weapons.')
        comparison_result = \
            ApexDatabase.get_instance().compare_all_weapons().limit_to_best_num(number)
        LOGGER.debug(f'Best: {comparison_result.get_archetypes()}')
        audible_names = ' '.join([CompareCommand.make_audible(weapon_archetype)
                                  for weapon_archetype in comparison_result.get_archetypes()])
        return audible_names
