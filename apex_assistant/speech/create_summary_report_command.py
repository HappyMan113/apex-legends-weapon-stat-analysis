import csv
import logging
import os
from typing import List, Optional, Tuple, TypeAlias

import numpy as np

from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech import apex_terms
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.term import Words
from apex_assistant.weapon import FullLoadout


_LOGGER = logging.getLogger()

TTK: TypeAlias = float
RANGE: TypeAlias = int
LOADOUT: TypeAlias = FullLoadout


class CreateSummaryReportCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(apex_terms.CREATE_SUMMARY_REPORT,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)

    def _execute(self, arguments: Words) -> str:
        translator = self.get_translator()

        weapons = translator.get_fully_kitted_weapons()
        loadout_to_range_dict: dict[LOADOUT, RANGE] = {
            FullLoadout(main_loadout, sidearm): max(main_weapon.get_eighty_percent_accuracy_range(),
                                                    sidearm.get_eighty_percent_accuracy_range())
            for main_weapon in weapons
            for main_loadout in main_weapon.get_main_loadout_variants(allow_skipping=True)
            for sidearm in weapons}

        range_inc = 10
        max_range = max(loadout_to_range_dict.values())
        min_ttk = 0.0
        max_ttk = 5
        ttk_step = 0.1
        ttks = np.arange(min_ttk, max_ttk + ttk_step, ttk_step)

        # Create table with collapsed accuracy ranges.
        table: dict[Tuple[RANGE, RANGE], Tuple[LOADOUT, ...]] = {}
        prev_best_loadouts: Optional[Tuple[LOADOUT]] = None
        prev_min_range: int = 0
        for accuracy_range in range(range_inc, max_range + range_inc, range_inc):
            loadouts = tuple(loadout
                             for loadout, loadout_range in loadout_to_range_dict.items()
                             if loadout_range >= accuracy_range)

            best_loadouts_for_ttks: Tuple[LOADOUT, ...] = tuple(
                max(loadouts, key=lambda loadout: loadout.get_cumulative_damage(ttk))
                for ttk in ttks
            )

            if ((prev_best_loadouts is None or best_loadouts_for_ttks == prev_best_loadouts) and
                    accuracy_range < max_range):
                prev_best_loadouts = best_loadouts_for_ttks
                continue

            table[(prev_min_range, accuracy_range)] = prev_best_loadouts
            prev_best_loadouts = best_loadouts_for_ttks
            prev_min_range = accuracy_range

        # Collapse TTK times.
        table2: dict[Tuple[RANGE, RANGE], dict[Tuple[TTK, TTK], LOADOUT]] = {key: {}
                                                                             for key in table}
        prev_best_loadouts = None
        prev_ttk = min_ttk
        ttk_ranges: List[Tuple[TTK, TTK]] = []
        for ttk_idx, ttk in enumerate(ttks):
            loadouts_for_ttk: Tuple[LOADOUT, ...] = tuple(loadouts[ttk_idx]
                                                          for loadouts in table.values())
            if ((prev_best_loadouts is None or loadouts_for_ttk == prev_best_loadouts) and
                    ttk_idx < len(ttks) - 1):
                prev_best_loadouts = loadouts_for_ttk
                continue

            ttk_range = (prev_ttk, ttk)
            for range_idx, ttk_range_to_loadout_dict in enumerate(table2.values()):
                ttk_range_to_loadout_dict[ttk_range] = prev_best_loadouts[range_idx]

            ttk_ranges.append(ttk_range)
            prev_best_loadouts = loadouts_for_ttk
            prev_ttk = ttk

        # Transpose.
        table3: dict[Tuple[TTK, TTK], dict[Tuple[RANGE, RANGE], LOADOUT]] = \
            {key: {} for key in ttk_ranges}
        for accuracy_range, ttk_range_to_loadout_dict in table2.items():
            for ttk_range, loadout in ttk_range_to_loadout_dict.items():
                table3[ttk_range][accuracy_range] = loadout

        # Write the results.
        filename = os.path.abspath('weapon_summary.csv')
        ttk_key = 'Time'
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = (ttk_key,) + tuple(self._get_range_str(min_range, max_range)
                                                 for min_range, max_range in table2)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ((min_ttk, max_ttk), accuracy_range_to_loadout_dict) in table3.items():
                row_dict: dict[str, str] = {
                    self._get_range_str(min_range, max_range): self._get_loadout_name(loadout)
                    for (min_range, max_range), loadout in accuracy_range_to_loadout_dict.items()
                }
                row_dict[ttk_key] = self._get_ttk_range_str(min_ttk, max_ttk)
                writer.writerow(row_dict)

        return f'Wrote summary to {filename}.'

    @staticmethod
    def _get_range_str(min_range: RANGE, max_range: RANGE) -> str:
        return f'{min_range}-{max_range} m'

    @staticmethod
    def _get_ttk_range_str(min_ttk: TTK, max_ttk: TTK) -> str:
        return f'{min_ttk:.1f}-{max_ttk:.1f} secs'

    @staticmethod
    def _get_loadout_name(loadout: LOADOUT) -> str:
        main_loadout = loadout.get_main_loadout()
        main_name = main_loadout.get_archetype().get_name()
        if main_loadout.get_variant_term() is not None:
            main_name = f'{main_name} ({main_loadout.get_variant_term()})'
        sidearm = loadout.get_sidearm()
        sidearm_name = sidearm.get_archetype().get_name()
        full_term = f'{main_name} {sidearm_name}'
        return full_term
