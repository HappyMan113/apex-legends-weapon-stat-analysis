import logging

from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import COMPARE
from apex_assistant.speech.term import Words


LOGGER = logging.getLogger()


class CompareCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(term=COMPARE,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)

    def _execute(self, arguments: Words) -> str:
        loadouts = tuple(self.get_translator().translate_loadouts(arguments))
        unique_loadouts = tuple(set(loadouts))
        if len(unique_loadouts) < 2:
            if len(unique_loadouts) == 1:
                LOGGER.info(f'All weapons are the same: {unique_loadouts[0]}')
            else:
                LOGGER.info(f'No weapons found matching voice command.')
            return 'Must specify two or more unique weapons to compare.'
        if len(unique_loadouts) < len(loadouts):
            LOGGER.warning('Duplicate weapons found. Only unique weapons will be compared.')
            loadouts = unique_loadouts

        uniqueness = self._get_uniqueness(loadouts)

        comparison_result = self._comparator.compare_loadouts(loadouts)
        best_weapon, score = comparison_result.get_best_loadout()
        LOGGER.info(f'Comparison result: {comparison_result}')
        audible_name = self._make_audible(best_weapon, uniqueness=uniqueness)

        if len(loadouts) == 2:
            second_best_weapon, second_best_score = comparison_result.get_nth_best_weapon(2)
            second_audible_name = self._make_audible(
                second_best_weapon,
                uniqueness=uniqueness)
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return (f'{audible_name} is {better_percentage} percent better than '
                    f'{second_audible_name}.')

        return f'{audible_name} is best.'
