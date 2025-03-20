import csv
import logging
import os

from apex_assistant.loadout_comparator import ComparisonResult, compare_loadouts
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech import apex_terms
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.term import Words
from apex_assistant.weapon import FullLoadout, SingleWeaponLoadout, Weapon


class CreateSummaryReportCommand(ApexCommand):
    _LOGGER = logging.getLogger()

    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(apex_terms.CREATE_SUMMARY_REPORT,
                         loadout_translator=loadout_translator)

    def _execute(self, arguments: Words) -> str:
        translator = self.get_translator()

        weapons = translator.get_fully_kitted_weapons()

        comparison_results: dict[Weapon, ComparisonResult] = {}
        for weapon_a in weapons:
            loadouts = tuple(FullLoadout(weapon_a, weapon_b) for weapon_b in weapons)
            comparison_results[weapon_a] = compare_loadouts(loadouts)

        max_n = len(weapons)

        # Write the results.
        filename = os.path.abspath('weapon_summary.csv')
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = tuple(
                function(weapon)
                for weapon in weapons
                for function in (self._get_loadout_score_header, self._get_loadout_sidearm_header))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            row_dict: dict[str, str] = {}
            for n in range(1, max_n + 1):
                for loadout, comparison_result in comparison_results.items():
                    score_header = self._get_loadout_score_header(loadout)
                    sidearm_header = self._get_loadout_sidearm_header(loadout)

                    nth_best_loadout, score = comparison_result.get_nth_best_loadout(n)
                    sidearm = nth_best_loadout.get_weapon_b()

                    row_dict[score_header] = f'{score:6.2f}'
                    row_dict[sidearm_header] = self._get_loadout_name(sidearm)

                writer.writerow(row_dict)

        message = f'Wrote summary report to {filename}.'
        self._LOGGER.info(message)
        return message

    @staticmethod
    def _get_loadout_name(loadout: SingleWeaponLoadout) -> str:
        return loadout.get_archetype().get_name()

    @staticmethod
    def _get_loadout_score_header(loadout: Weapon) -> str:
        name = CreateSummaryReportCommand._get_loadout_name(loadout)
        return f'Score ({name})'

    @staticmethod
    def _get_loadout_sidearm_header(loadout: Weapon) -> str:
        name = CreateSummaryReportCommand._get_loadout_name(loadout)
        return f'{name} Sidearm'
