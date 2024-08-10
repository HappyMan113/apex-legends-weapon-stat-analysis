import logging
from typing import Dict, Iterable, Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand, Uniqueness
from apex_assistant.speech.apex_terms import BEST, COMPARE, LOADOUT, LOADOUTS, MAIN, SIDEARM
from apex_assistant.speech.term import Term, Words
from apex_assistant.speech.term_translator import SingleTermFinder
from apex_assistant.weapon import FullLoadout, Weapon, WeaponArchetype


LOGGER = logging.getLogger()


class CompareCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(term=COMPARE | (BEST + (LOADOUT | LOADOUTS)),
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)
        show = Term('show', 'so')
        plots = Term('plots')
        self._show_plots_finder = SingleTermFinder(show + plots)

    def _execute(self, arguments: Words) -> str:
        translator = self.get_translator()
        comparator = self.get_comparator()
        loadouts: Tuple[FullLoadout, ...] = tuple(translator.translate_full_loadouts(arguments))
        show_plots = bool(self._show_plots_finder.find_all(arguments))
        unique_loadouts = set(loadouts)
        if len(unique_loadouts) < len(loadouts):
            LOGGER.warning('Duplicate weapons found. Only unique weapons will be compared.')
            loadouts = tuple(unique_loadouts)

        if len(loadouts) == 1:
            LOGGER.info(f'All weapons are the same: {loadouts[0]}')
            return 'Only one unique weapon was specified.'

        if len(loadouts) == 0:
            weapons = translator.get_fully_kitted_weapons()
            loadouts = tuple(comparator.get_best_loadouts(weapons))
            uniqueness = Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
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
    def _get_uniqueness(loadouts: Tuple[FullLoadout, ...]) -> Uniqueness:
        check_tuple(FullLoadout, loadouts=loadouts)

        main_weapon_archetypes = set(loadout.get_main_loadout().get_archetype()
                                     for loadout in loadouts)
        main_weapon_archetypes_unique = len(main_weapon_archetypes) == len(loadouts)
        sidearms = set(weapon.get_sidearm() for weapon in loadouts)

        main_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_main_weapon()
                                                                         for loadout in loadouts)

        if main_weapon_archetypes_unique:
            return Uniqueness.SAY_MAIN_ARCHETYPE_NAMES

        sidearms_all_same = len(sidearms) == 1
        if sidearms_all_same:
            return (Uniqueness.SAY_MAIN_ARCHETYPE_NAMES if main_attachments_all_same else
                    Uniqueness.SAY_MAIN_LOADOUT_NAMES)

        main_loadouts = set(weapon.get_main_loadout() for weapon in loadouts)
        main_loadouts_all_same = len(main_loadouts) == 1
        sidearm_archetypes = set(
            loadout.get_sidearm().get_archetype() if loadout.get_sidearm() is not None else None
            for loadout in loadouts)
        sidearm_archetypes_unique = len(sidearm_archetypes) == len(loadouts)

        if main_loadouts_all_same and sidearm_archetypes_unique:
            return Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        sidearms_unique = len(sidearms) == len(loadouts)
        sidearm_attachments_all_same = CompareCommand._attachments_all_same(loadout.get_sidearm()
                                                                            for loadout in loadouts)

        if main_loadouts_all_same and sidearms_unique:
            return (Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES if sidearm_attachments_all_same else
                    Uniqueness.SAY_SIDEARM_WEAPON_NAMES)

        if main_attachments_all_same and sidearm_attachments_all_same:
            return Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
        return Uniqueness.SAY_FULL_LOADOUT_NAMES

    @staticmethod
    def _make_audible(loadout: FullLoadout,
                      uniqueness: Uniqueness = Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES) -> str:
        check_type(FullLoadout, loadout=loadout)
        check_type(Uniqueness, uniqueness=uniqueness)

        if uniqueness is Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            main_loadout = loadout.get_main_loadout()
            loadout_term = MAIN.append(main_loadout.get_archetype().get_term())
            if main_loadout.get_variant_term() is not None:
                loadout_term = loadout_term.append(main_loadout.get_variant_term())
        elif uniqueness is Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_archetype().get_term()
        elif uniqueness is Uniqueness.SAY_MAIN_LOADOUT_NAMES:
            loadout_term = MAIN.append(loadout.get_main_loadout().get_term())
        elif uniqueness is Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES:
            main_loadout = loadout.get_main_loadout()
            main_term = main_loadout.get_archetype().get_term()
            if main_loadout.get_variant_term() is not None:
                main_term = main_term.append(main_loadout.get_variant_term())
            sidearm_term = loadout.get_sidearm().get_archetype().get_term()
            loadout_term = main_term.append(SIDEARM, sidearm_term)
        elif uniqueness is Uniqueness.SAY_SIDEARM_WEAPON_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_term()
        else:
            loadout_term = loadout.get_term()

        return loadout_term.to_audible_str()
