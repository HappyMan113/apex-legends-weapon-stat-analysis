import logging
from enum import IntEnum
from typing import Dict, Iterable, Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import BEST, CARE_PACKAGE, COMPARE, LOADOUT, LOADOUTS
from apex_assistant.speech.term import Term, Words
from apex_assistant.speech.term_translator import SingleTermFinder, Translator
from apex_assistant.weapon import FullLoadout, Weapon, WeaponArchetype


LOGGER = logging.getLogger()


class _Uniqueness(IntEnum):
    SAY_A_ARCHETYPE_NAMES = 0
    SAY_B_ARCHETYPE_NAMES = 1
    SAY_FULL_LOADOUT_ARCHETYPE_NAMES = 2
    SAY_A_WEAPON_NAMES = 3
    SAY_B_WEAPON_NAMES = 4
    SAY_FULL_LOADOUT_NAMES = 5


class _Excludes(IntEnum):
    CARE_PACKAGE = 0
    NON_HOPPED_UP = 1


class CompareCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(term=COMPARE | (BEST + (LOADOUT | LOADOUTS)),
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)
        show = Term('show', 'so')
        plots = Term('plots', 'plot')
        self._show_plots_finder = SingleTermFinder(show + plots)

        exclude = Term('exclude')
        exclude_care_package = exclude + CARE_PACKAGE
        exclude_non_hopped_up = exclude + Term('non hopped up', 'non-hopped up')
        self._exclude_translator = Translator({exclude_care_package: _Excludes.CARE_PACKAGE,
                                               exclude_non_hopped_up: _Excludes.NON_HOPPED_UP})

    def _execute(self, arguments: Words) -> str:
        translator = self.get_translator()
        comparator = self.get_comparator()
        loadouts: Tuple[FullLoadout, ...] = tuple(translator.translate_full_loadouts(arguments))
        show_plots = bool(self._show_plots_finder.find_all(arguments))
        unique_loadouts = set(loadouts)
        if len(unique_loadouts) < len(loadouts):
            LOGGER.warning('Duplicate loadout found. Only unique weapons will be compared.')
            loadouts = tuple(unique_loadouts)

        if len(loadouts) == 1:
            LOGGER.info(f'All weapons are the same: {loadouts[0]}')
            return 'Only one unique weapon was specified.'

        if len(loadouts) == 0:
            excludes = self._exclude_translator.translate_terms(arguments).values()
            include_care_package = _Excludes.CARE_PACKAGE not in excludes
            include_non_hopped_up = _Excludes.NON_HOPPED_UP not in excludes

            weapons = translator.get_fully_kitted_weapons(
                include_care_package_weapons=include_care_package,
                include_non_hopped_up_weapons=include_non_hopped_up)
            loadouts = tuple(comparator.get_best_loadouts(weapons))
            uniqueness = _Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
        else:
            uniqueness = self._get_uniqueness(loadouts)

        comparison_result = comparator.compare_loadouts(loadouts, show_plots=show_plots)
        best_loadout, score = comparison_result.get_best_loadout()

        LOGGER.info(f'Comparison result: {comparison_result}')
        audible_name = self._make_audible(best_loadout, uniqueness=uniqueness)

        if len(loadouts) == 2:
            second_best_weapon, second_best_score = comparison_result.get_nth_best_loadout(2)
            second_audible_name = self._make_audible(second_best_weapon, uniqueness=uniqueness)
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

        main_weapon_archetypes = set(loadout.get_weapon_a().get_archetype()
                                     for loadout in loadouts)
        main_weapon_archetypes_unique = len(main_weapon_archetypes) == len(loadouts)
        sidearms = set(weapon.get_weapon_b() for weapon in loadouts)

        main_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_weapon_a()
                                                                         for loadout in loadouts)

        if main_weapon_archetypes_unique:
            return _Uniqueness.SAY_A_ARCHETYPE_NAMES

        sidearms_all_same = len(sidearms) == 1
        if sidearms_all_same:
            return (_Uniqueness.SAY_A_ARCHETYPE_NAMES if main_attachments_all_same else
                    _Uniqueness.SAY_A_WEAPON_NAMES)

        main_loadouts = set(weapon.get_weapon_a() for weapon in loadouts)
        main_loadouts_all_same = len(main_loadouts) == 1
        sidearm_archetypes = set(
            loadout.get_weapon_b().get_archetype() if loadout.get_weapon_b() is not None else None
            for loadout in loadouts)
        sidearm_archetypes_unique = len(sidearm_archetypes) == len(loadouts)

        if main_loadouts_all_same and sidearm_archetypes_unique:
            return _Uniqueness.SAY_B_ARCHETYPE_NAMES

        sidearms_unique = len(sidearms) == len(loadouts)
        sidearm_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_weapon_b()
                                                                            for loadout in loadouts)

        if main_loadouts_all_same and sidearms_unique:
            return (_Uniqueness.SAY_B_ARCHETYPE_NAMES if sidearm_attachments_all_same else
                    _Uniqueness.SAY_B_WEAPON_NAMES)

        if main_attachments_all_same and sidearm_attachments_all_same:
            return _Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
        return _Uniqueness.SAY_FULL_LOADOUT_NAMES

    @staticmethod
    def _make_audible(
            loadout: FullLoadout,
            uniqueness: _Uniqueness = _Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES) -> str:
        check_type(FullLoadout, loadout=loadout)
        check_type(_Uniqueness, uniqueness=uniqueness)

        if uniqueness is _Uniqueness.SAY_A_ARCHETYPE_NAMES:
            main_weapon = loadout.get_weapon_a()
            loadout_term = main_weapon.get_archetype().get_term()
        elif uniqueness is _Uniqueness.SAY_B_ARCHETYPE_NAMES:
            loadout_term = loadout.get_weapon_b().get_archetype().get_term()
        elif uniqueness is _Uniqueness.SAY_A_WEAPON_NAMES:
            loadout_term = loadout.get_weapon_a().get_term()
        elif uniqueness is _Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES:
            main_weapon = loadout.get_weapon_a()
            main_term = main_weapon.get_archetype().get_term()
            weapon_b_term = loadout.get_weapon_b().get_archetype().get_term()
            loadout_term = main_term.append(weapon_b_term)
        elif uniqueness is _Uniqueness.SAY_B_WEAPON_NAMES:
            loadout_term = loadout.get_weapon_b().get_term()
        else:
            loadout_term = loadout.get_term()

        return loadout_term.to_audible_str()
