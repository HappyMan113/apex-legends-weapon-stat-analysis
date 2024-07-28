import logging
from enum import IntEnum
from typing import Optional, Tuple

from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import COMPARE, MAIN, SIDEARM, WITHOUT
from apex_assistant.speech.term import TermBase, Words
from apex_assistant.weapon import Loadout


LOGGER = logging.getLogger()


class _Uniqueness(IntEnum):
    SAY_MAIN_ARCHETYPE_NAMES = 0
    SAY_MAIN_ARCHETYPE_NAMES_AND_MAIN_WORD = 1
    SAY_SIDEARM_ARCHETYPE_NAMES = 2
    SAY_MAIN_LOADOUT_NAMES = 3
    SAY_MAIN_LOADOUT_NAMES_AND_MAIN_WORD = 4
    SAY_SIDEARM_WEAPON_NAMES = 5
    SAY_EVERYTHING = 6


class CompareCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(term=COMPARE,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)

    def _execute(self, arguments: Words) -> str:
        loadouts = tuple(self.get_translator().translate_any_loadouts(arguments))
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
        audible_name = self._make_audible(best_weapon, uniqueness=uniqueness).to_audible_str()

        if len(loadouts) == 2:
            second_best_weapon, second_best_score = comparison_result.get_nth_best_weapon(2)
            second_audible_name = self._make_audible(
                second_best_weapon,
                uniqueness=uniqueness).to_audible_str()
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return (f'{audible_name} is {better_percentage} percent better than '
                    f'{second_audible_name}.')

        return f'{audible_name} is best.'

    @staticmethod
    def _get_uniqueness(loadouts: Tuple[Loadout, ...]) -> _Uniqueness:
        main_weapon_archetypes = set(loadout.get_archetype() for loadout in loadouts)
        main_weapon_archetypes_unique = len(main_weapon_archetypes) == len(loadouts)
        sidearms = set(weapon.get_sidearm() for weapon in loadouts)

        if main_weapon_archetypes_unique:
            sidearms_all_none = all(sidearm is None for sidearm in sidearms)
            return (_Uniqueness.SAY_MAIN_ARCHETYPE_NAMES if sidearms_all_none else
                    _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES_AND_MAIN_WORD)

        sidearms_all_same = len(sidearms) == 1
        if sidearms_all_same:
            sidearms_all_none = all(sidearm is None for sidearm in sidearms)
            return (_Uniqueness.SAY_MAIN_LOADOUT_NAMES if sidearms_all_none else
                    _Uniqueness.SAY_MAIN_LOADOUT_NAMES_AND_MAIN_WORD)

        main_loadouts = set(weapon.get_main_loadout() for weapon in loadouts)
        main_loadouts_all_same = len(main_loadouts) == 1
        sidearm_archetypes = set(
            loadout.get_sidearm().get_archetype() if loadout.get_sidearm() is not None else None
            for loadout in loadouts)
        sidearm_archetypes_unique = len(sidearm_archetypes) == len(loadouts)

        if main_loadouts_all_same and sidearm_archetypes_unique:
            return _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        sidearms_unique = len(sidearms) == len(loadouts)
        if main_loadouts_all_same and sidearms_unique:
            return _Uniqueness.SAY_SIDEARM_WEAPON_NAMES

        return _Uniqueness.SAY_EVERYTHING

    @staticmethod
    def _make_audible(loadout: Loadout, uniqueness: _Uniqueness) -> TermBase:
        if uniqueness is _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            loadout_term = loadout.get_archetype().get_term()
        elif uniqueness is _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES_AND_MAIN_WORD:
            loadout_term = MAIN.append(loadout.get_archetype().get_term())
        elif uniqueness is _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            loadout_term = CompareCommand._make_sidearm_audible(
                loadout.get_sidearm().get_archetype().get_term()
                if loadout.get_sidearm() is not None
                else None)
        elif uniqueness is _Uniqueness.SAY_MAIN_LOADOUT_NAMES:
            loadout_term = loadout.get_main_loadout().get_term()
        elif uniqueness is _Uniqueness.SAY_MAIN_LOADOUT_NAMES_AND_MAIN_WORD:
            loadout_term = MAIN.append(loadout.get_main_loadout().get_term())
        elif uniqueness is _Uniqueness.SAY_SIDEARM_WEAPON_NAMES:
            loadout_term = CompareCommand._make_sidearm_audible(
                loadout.get_sidearm().get_term()
                if loadout.get_sidearm() is not None
                else None)
        else:
            loadout_term = loadout.get_term()
        return loadout_term

    @staticmethod
    def _make_sidearm_audible(sidearm_term: Optional[TermBase]) -> TermBase:
        if sidearm_term is None:
            loadout_term = WITHOUT + SIDEARM
        else:
            loadout_term = SIDEARM + sidearm_term
        return loadout_term
