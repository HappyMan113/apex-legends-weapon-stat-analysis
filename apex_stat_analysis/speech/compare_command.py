import logging
import re

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.term_translator import ApexTranslator
from apex_stat_analysis.speech.terms import ApexTerms, Words
from apex_stat_analysis.weapon import WeaponBase, add_sidearm_and_reload
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()

class CompareCommand(Command):
    def __init__(self):
        super().__init__(ApexTerms.COMPARE)
        self._no_reload_translator = ApexTranslator({ApexTerms.NO_RELOAD_TERM: False})

    def _execute(self, arguments: Words) -> str:
        weapons = tuple(ApexDatabase.get_instance().get_base_weapons(arguments))
        if len(weapons) < 2:
            return 'Must specify two or more weapons to compare.'
        archetypes = set(weapon.get_archetype() for weapon in weapons)
        unique_archetypes = len(archetypes) == len(weapons)
        with_reloads = next(map(lambda arg: arg.get_parsed(),
                                self._no_reload_translator.translate_terms(arguments)),
                            True)
        weapons = add_sidearm_and_reload(weapons, reload=with_reloads)

        LOGGER.info(f'Comparing: {weapons}')
        comparison_result = ApexDatabase.get_instance().compare_weapons(weapons)
        best_weapon, score = comparison_result.get_best_weapon()
        LOGGER.info(f'Best: {best_weapon}')
        audible_name = self.make_audible(best_weapon, unique_archetypes=unique_archetypes)

        if len(weapons) == 2:
            second_best_weapon, second_best_score = comparison_result.get_nth_best_weapon(2)
            second_audible_name = self.make_audible(second_best_weapon,
                                                    unique_archetypes=unique_archetypes)
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return (f'{audible_name} is {better_percentage:.0f} percent better than'
                    f' {second_audible_name}.')

        return f'{audible_name} is best.'

    @staticmethod
    def make_audible(weapon: WeaponBase, unique_archetypes: bool):
        if unique_archetypes:
            weapon_name = weapon.get_archetype().get_name()
        else:
            weapon_name = weapon.get_name()
        audible_name = re.sub('[()]', '', weapon_name)
        return audible_name

