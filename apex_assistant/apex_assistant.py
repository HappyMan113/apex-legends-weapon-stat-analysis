import logging
import os
import sys
from typing import Tuple

from pydub.utils import which

from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.best_command import BestLoadoutsCommand
from apex_assistant.speech.command_registry import CommandRegistry
from apex_assistant.speech.compare_command import CompareCommand
from apex_assistant.speech.configure_command import ConfigureCommand
from apex_assistant.speech.create_summary_report_command import CreateSummaryReportCommand
from apex_assistant.speech.speech_client import SpeechClient
from apex_assistant.weapon import WeaponArchetypes
from apex_assistant.weapon_csv_parser import TTKCsvReader, WeaponCsvReader


logger = logging.getLogger()


def register_commands() -> CommandRegistry:
    # Load everything in.
    self_path = os.path.dirname(__file__)

    # TODO: Measure TTK in terms of duration of your active firing (i.e. not counting short pauses).
    #  Active firing means counting one round period per round fired. i.e. You can multiply number
    #  of rounds fired with round period and add reload time if you're in the open when reloading.
    ttks_filename = os.path.join(self_path, 'historic_ttks.csv')
    with open(ttks_filename, encoding='utf-8-sig') as fp:
        dr = TTKCsvReader(fp)
        ttk_entries = tuple(dr)
    comparator = LoadoutComparator(ttk_entries)

    apex_stats_filename = os.path.join(self_path, 'weapon_stats.csv')
    with open(apex_stats_filename, encoding='utf-8-sig') as fp:
        dr = WeaponCsvReader(fp)
        weapons: Tuple[WeaponArchetypes, ...] = WeaponArchetypes.group_archetypes(dr)

    apex_config_filename = os.path.join(self_path, 'apex_config.json')
    apex_config = ApexConfig.load(apex_config_filename)
    translator = LoadoutTranslator(weapon_archetypes=weapons, apex_config=apex_config)

    registry = CommandRegistry(
        CompareCommand(loadout_translator=translator, loadout_comparator=comparator),
        BestLoadoutsCommand(loadout_translator=translator, loadout_comparator=comparator),
        ConfigureCommand(loadout_translator=translator, loadout_comparator=comparator),
        CreateSummaryReportCommand(loadout_translator=translator, loadout_comparator=comparator))
    return registry


def set_up_logger(level):
    logger.handlers.clear()
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s:%(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def ensure_ffmpeg_installed():
    if which("ffmpeg"):
        return

    print('FFMpeg not found. Installing...')
    os.system('winget install Gyan.FFmpeg')
    print('===================================================================')
    print('FFMPEG installed. Please restart your shell and rerun this program.')
    print('===================================================================')
    sys.exit(0)


def main():
    ensure_ffmpeg_installed()
    log_level = logging.INFO
    set_up_logger(log_level)
    registry = register_commands()

    try:
        with SpeechClient(registry) as client:
            client.start()
    except KeyboardInterrupt:
        logger.info('Done.')
        sys.exit(0)
