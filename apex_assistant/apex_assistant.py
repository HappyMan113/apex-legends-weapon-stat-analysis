import logging
import os
import sys
from typing import Tuple

from pydub.utils import which

from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.command_registry import CommandRegistry
from apex_assistant.speech.compare_command import CompareCommand
from apex_assistant.speech.configure_command import ConfigureCommand
from apex_assistant.speech.create_summary_report_command import CreateSummaryReportCommand
from apex_assistant.speech.speech_client import SpeechClient
from apex_assistant.weapon import WeaponArchetypes
from apex_assistant.weapon_csv_parser import WeaponCsvReader


logger = logging.getLogger()


def load_weapon_archetypes() -> Tuple[WeaponArchetypes, ...]:
    module_path = os.path.dirname(__file__)

    apex_stats_filename = os.path.join(module_path, 'weapon_stats.csv')
    with open(apex_stats_filename, encoding='utf-8-sig') as fp:
        dr = WeaponCsvReader(fp)
        weapons: Tuple[WeaponArchetypes, ...] = WeaponArchetypes.group_archetypes(dr)
    return weapons


def register_commands(weapons: Tuple[WeaponArchetypes, ...]) -> \
        Tuple[LoadoutTranslator, CommandRegistry]:
    # Load everything in.
    user_data_path = os.path.join(os.getenv('APPDATA'), 'Apex Assistant')
    if not os.path.exists(user_data_path):
        os.mkdir(user_data_path)
    elif not os.path.isdir(user_data_path):
        raise IOError(f'{user_data_path} is not a directory!')

    apex_config_filename = os.path.join(user_data_path, 'config.json')
    apex_config = ApexConfig.load(apex_config_filename)
    logger.info(f'Loaded config: {os.path.realpath(apex_config_filename)}')
    translator = LoadoutTranslator(weapon_archetypes=weapons, apex_config=apex_config)

    registry = CommandRegistry(
        CompareCommand(loadout_translator=translator),
        ConfigureCommand(loadout_translator=translator),
        CreateSummaryReportCommand(loadout_translator=translator))
    return translator, registry


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

    archetypes = load_weapon_archetypes()
    translator, registry = register_commands(archetypes)
    for_legend_str = (f' for {translator.get_legend()}' if translator.get_legend() is not None else
                      '')
    logger.info(f'Preloading fully-kitted loadouts{for_legend_str}...')
    translator.preload_loadouts()

    try:
        with SpeechClient(registry) as client:
            client.start()
    except KeyboardInterrupt:
        logger.info('Done.')
        sys.exit(0)
