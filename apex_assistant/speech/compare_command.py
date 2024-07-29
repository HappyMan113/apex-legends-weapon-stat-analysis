import logging
from enum import IntEnum
from typing import Dict, Iterable, Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import COMPARE, MAIN, SIDEARM
from apex_assistant.speech.term import TermBase, Words
from apex_assistant.weapon import FullLoadout, Weapon, WeaponArchetype


LOGGER = logging.getLogger()


class _Uniqueness(IntEnum):
    SAY_MAIN_ARCHETYPE_NAMES = 0
    SAY_SIDEARM_ARCHETYPE_NAMES = 1
    SAY_LOADOUT_ARCHETYPE_NAMES = 2
    SAY_MAIN_LOADOUT_NAMES = 3
    SAY_SIDEARM_WEAPON_NAMES = 4
    SAY_LOADOUT_NAMES = 5


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
    def _attachments_all_same(weapons: Iterable[Weapon]) -> bool:
        archetype_to_loadout_dict: Dict[WeaponArchetype, Weapon] = {}
        for weapon in weapons:
            archetype = weapon.get_archetype()
            if (archetype in archetype_to_loadout_dict and
                    archetype_to_loadout_dict[archetype] != weapon):
                return False
            archetype_to_loadout_dict[archetype] = weapon
        return True

    @staticmethod
    def _get_uniqueness(loadouts: Tuple[FullLoadout, ...]) -> _Uniqueness:
        check_tuple(FullLoadout, loadouts=loadouts)

        main_weapon_archetypes = set(loadout.get_archetype() for loadout in loadouts)
        main_weapon_archetypes_unique = len(main_weapon_archetypes) == len(loadouts)
        sidearms = set(weapon.get_sidearm() for weapon in loadouts)

        main_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_main_weapon()
                                                                         for loadout in loadouts)

        if main_weapon_archetypes_unique:
            return _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES

        sidearms_all_same = len(sidearms) == 1
        if sidearms_all_same:
            if main_attachments_all_same:
                return _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES
            else:
                return _Uniqueness.SAY_MAIN_LOADOUT_NAMES

        main_loadouts = set(weapon.get_main_loadout() for weapon in loadouts)
        main_loadouts_all_same = len(main_loadouts) == 1
        sidearm_archetypes = set(
            loadout.get_sidearm().get_archetype() if loadout.get_sidearm() is not None else None
            for loadout in loadouts)
        sidearm_archetypes_unique = len(sidearm_archetypes) == len(loadouts)

        if main_loadouts_all_same and sidearm_archetypes_unique:
            return _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        sidearms_unique = len(sidearms) == len(loadouts)
        sidearm_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_sidearm()
                                                                            for loadout in loadouts)

        if main_loadouts_all_same and sidearms_unique:
            if sidearm_attachments_all_same:
                return _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES
            else:
                return _Uniqueness.SAY_SIDEARM_WEAPON_NAMES

        if main_attachments_all_same and sidearm_attachments_all_same:
            return _Uniqueness.SAY_LOADOUT_ARCHETYPE_NAMES
        return _Uniqueness.SAY_LOADOUT_NAMES

    @staticmethod
    def _make_audible(loadout: FullLoadout, uniqueness: _Uniqueness) -> TermBase:
        check_type(FullLoadout, loadout=loadout)
        check_type(_Uniqueness, uniqueness=uniqueness)

        if uniqueness is _Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            loadout_term = MAIN.append(loadout.get_archetype().get_term())
        elif uniqueness is _Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_archetype().get_term()
        elif uniqueness is _Uniqueness.SAY_MAIN_LOADOUT_NAMES:
            loadout_term = MAIN.append(loadout.get_main_loadout().get_term())
        elif uniqueness is _Uniqueness.SAY_LOADOUT_ARCHETYPE_NAMES:
            loadout_term = loadout.get_main_loadout().get_archetype().get_term().append(
                SIDEARM,
                loadout.get_sidearm().get_archetype().get_term())
        elif uniqueness is _Uniqueness.SAY_SIDEARM_WEAPON_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_term()
        else:
            loadout_term = loadout.get_term()
        return loadout_term
