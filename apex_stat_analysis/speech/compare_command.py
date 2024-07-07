import logging
import re
from enum import Enum

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.term import Words
from apex_stat_analysis.speech.terms import COMPARE
from apex_stat_analysis.weapon import CombinedWeapon, WeaponBase
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()


class Uniqueness(Enum):
    SAY_MAIN_ARCHETYPE_NAMES = 0
    SAY_SIDEARM_ARCHETYPE_NAMES = 1
    SAY_EVERYTHING = 2


class CompareCommand(Command):
    def __init__(self):
        super().__init__(COMPARE)

    def _execute(self, arguments: Words) -> str:
        weapons = tuple(ApexDatabase.get_instance().get_base_weapons(arguments))
        unique_weapons = tuple(set(weapons))
        if len(unique_weapons) < 2:
            if len(unique_weapons) == 1:
                LOGGER.info(f'All weapons are the same: {unique_weapons[0]}')
            else:
                LOGGER.info(f'No weapons found matching voice command.')
            return 'Must specify two or more unique weapons to compare.'
        if len(unique_weapons) < len(weapons):
            LOGGER.warning('Duplicate weapons found. Only unique weapons will be compared.')
            weapons = unique_weapons

        uniqueness = self.get_uniqueness(weapons)

        delim = '\n  - '
        weapons_str = delim.join(map(str, weapons))
        LOGGER.info(f'Comparing:{delim}{weapons_str}')
        comparison_result = ApexDatabase.get_instance().compare_weapons(weapons)
        best_weapon, score = comparison_result.get_best_weapon()
        LOGGER.info(f'Best: {best_weapon}')
        audible_name = self.make_audible(best_weapon, uniqueness=uniqueness)

        if len(weapons) == 2:
            _, second_best_score = comparison_result.get_nth_best_weapon(2)
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return f'{audible_name} is {better_percentage:.0f} percent better.'

        return f'{audible_name} is best.'

    @staticmethod
    def get_uniqueness(weapons) -> Uniqueness:
        archetypes = set(weapon.get_archetype() for weapon in weapons)
        unique_archetypes = len(archetypes) == len(weapons)
        if unique_archetypes:
            return Uniqueness.SAY_MAIN_ARCHETYPE_NAMES

        # This is kinda dumb, but I can't think of a better way right now.
        if all(isinstance(weapon, CombinedWeapon) for weapon in weapons):
            main_weapons = set(map(CombinedWeapon.get_main_weapon, weapons))
            main_weapons_all_same = len(main_weapons) == 1
            if not main_weapons_all_same:
                return Uniqueness.SAY_EVERYTHING

            sidearm_archetypes = set(map(CombinedWeapon.get_sidearm, weapons))
            unique_sidearm_archetypes = len(sidearm_archetypes) == len(weapons)
            if unique_sidearm_archetypes:
                return Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        return Uniqueness.SAY_EVERYTHING

    @staticmethod
    def make_audible(weapon: WeaponBase, uniqueness: Uniqueness):
        if uniqueness is Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            weapon_name = weapon.get_archetype().get_name()
        elif uniqueness is Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            assert isinstance(weapon, CombinedWeapon)
            weapon_name = f'sidearm {weapon.get_sidearm().get_archetype().get_name()}'
        else:
            weapon_name = weapon.get_name()
        audible_name = re.sub('[()]', '', weapon_name)
        return audible_name
