import logging
from enum import Enum

from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import COMPARE
from apex_assistant.speech.term import Words
from apex_assistant.weapon import CombinedWeapon, WeaponBase
from apex_assistant.weapon_comparer import WeaponComparer
from apex_assistant.weapon_translator import WeaponTranslator


LOGGER = logging.getLogger()


class _Uniqueness(Enum):
    SAY_MAIN_ARCHETYPE_NAMES = 0
    SAY_SIDEARM_ARCHETYPE_NAMES = 1
    SAY_EVERYTHING = 2


class CompareCommand(ApexCommand):
    def __init__(self, weapon_translator: WeaponTranslator, weapon_comparer: WeaponComparer):
        super().__init__(term=COMPARE,
                         weapon_translator=weapon_translator,
                         weapon_comparer=weapon_comparer)

    def _execute(self, arguments: Words) -> str:
        weapons = tuple(self.get_translator().translate_weapon_terms(arguments))
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

        uniqueness = self._get_uniqueness(weapons)

        delim = '\n  - '
        weapons_str = delim.join(map(str, weapons))
        LOGGER.info(f'Comparing:{delim}{weapons_str}')
        comparison_result = self._comparer.compare_weapons(weapons)
        best_weapon, score = comparison_result.get_best_weapon()
        LOGGER.info(f'Comparison result: {comparison_result}')
        audible_name = self._make_audible(best_weapon, uniqueness=uniqueness)\

        if len(weapons) == 2:
            _, second_best_score = comparison_result.get_nth_best_weapon(2)
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return f'{audible_name} is {better_percentage:.0f} percent better.'

        return f'{audible_name} is best.'

    @staticmethod
    def _get_uniqueness(weapons) -> _Uniqueness:
        archetypes = set(weapon.get_archetype() for weapon in weapons)
        unique_archetypes = len(archetypes) == len(weapons)
        if unique_archetypes:
            return _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES

        # This is kinda dumb, but I can't think of a better way right now.
        if all(isinstance(weapon, CombinedWeapon) for weapon in weapons):
            main_weapons = set(map(CombinedWeapon.get_main_weapon, weapons))
            main_weapons_all_same = len(main_weapons) == 1
            if not main_weapons_all_same:
                return _Uniqueness.SAY_EVERYTHING

            sidearm_archetypes = set(map(CombinedWeapon.get_sidearm, weapons))
            unique_sidearm_archetypes = len(sidearm_archetypes) == len(weapons)
            if unique_sidearm_archetypes:
                return _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        return _Uniqueness.SAY_EVERYTHING

    @staticmethod
    def _make_audible(weapon: WeaponBase, uniqueness: _Uniqueness):
        if uniqueness is _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            weapon_name: str = weapon.get_archetype().get_term().to_audible_str()
        elif uniqueness is _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            assert isinstance(weapon, CombinedWeapon)
            sidearm_name: str = weapon.get_sidearm().get_archetype().get_term().to_audible_str()
            weapon_name: str = f'sidearm {sidearm_name}'
        else:
            weapon_name: str = weapon.get_term().to_audible_str()
        return weapon_name
