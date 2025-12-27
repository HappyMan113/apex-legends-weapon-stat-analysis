import logging
from enum import IntEnum
from typing import Dict, Iterable, Tuple

from apex_assistant.loadout_comparator import compare_loadouts, get_best_loadouts
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import (AKIMBO,
                                              BEST,
                                              CARE_PACKAGE,
                                              COMPARE,
                                              LOADOUT,
                                              LOADOUTS,
                                              REVVED,
                                              TURBOCHARGED)
from apex_assistant.speech.term import Term, Words
from apex_assistant.speech.term_translator import SingleTermFinder, Translator
from apex_assistant.weapon import (ExcludeFlag,
                                   FullLoadout,
                                   Weapon,
                                   WeaponArchetype,
                                   WeaponArchetypes)


LOGGER = logging.getLogger()


class CompareCommand(ApexCommand):
    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(term=COMPARE | (BEST + (LOADOUT | LOADOUTS)),
                         loadout_translator=loadout_translator)
        show = Term('show', 'so')
        plots = Term('plots', 'plot')
        self._show_plots_finder = SingleTermFinder(show + plots)

        non = Term('non')
        care_package = CARE_PACKAGE
        hopped_up = Term('hopped up', 'hopped-up')
        non_hopped_up = (non + hopped_up) | Term('non-hopped up')
        revved_up = REVVED
        non_revved_up = (non + revved_up) | Term('non-revved up')
        akimbo = AKIMBO
        non_akimbo = non + akimbo
        turbocharged = TURBOCHARGED

        exclude = Term('exclude', 'no')
        self._exclude_translator = Translator[ExcludeFlag]({
            exclude + term: flag
            for term, flag in {
                care_package: ExcludeFlag.CARE_PACKAGE,
                hopped_up: ExcludeFlag.HOPPED_UP,
                non_hopped_up: ExcludeFlag.NON_HOPPED_UP,
                revved_up: ExcludeFlag.REVVED_UP,
                non_revved_up: ExcludeFlag.NON_REVVED_UP,
                akimbo: ExcludeFlag.AKIMBO,
                non_akimbo: ExcludeFlag.NON_AKIMBO,
                turbocharged: ExcludeFlag.TURBOCHARGED
            }.items()})

    def _execute(self, arguments: Words) -> str:
        translator = self.get_translator()
        loadouts: Tuple[FullLoadout, ...] = tuple(translator.translate_full_loadouts(arguments))
        show_plots = bool(self._show_plots_finder.find_all(arguments))
        unique_loadouts = set(loadouts)
        if len(unique_loadouts) < len(loadouts):
            LOGGER.debug('Duplicate loadout found. Only unique weapons will be compared.')
            loadouts = tuple(unique_loadouts)

        exclude_flags: int = 0
        for exclude_flag in self._exclude_translator.translate_terms(arguments).values():
            exclude_flags |= exclude_flag

        if len(loadouts) == 0:
            weapons = translator.get_fully_kitted_weapons(exclude_flags=exclude_flags)
            if len(weapons) == 0:
                exclude_flags_set = tuple(flag for flag in ExcludeFlag if flag.find(exclude_flags))
                LOGGER.debug(f'Exclude flags: {exclude_flags_set}')
                return 'All weapons were filtered out.'
            loadouts = tuple(get_best_loadouts(weapons))
        elif len(loadouts) == 1:
            loadouts = tuple(WeaponArchetypes.filter_loadouts(loadouts,
                                                              exclude_flags=exclude_flags))
        else:
            loadouts = tuple(WeaponArchetypes.filter_loadouts(loadouts,
                                                              exclude_flags=exclude_flags))

        comparison_result = compare_loadouts(loadouts, show_plots=show_plots)
        return f'Comparison result: {comparison_result}'

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
