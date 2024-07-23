import logging
from enum import IntEnum
from typing import Generator, Optional, Tuple

from apex_assistant.checker import check_bool, check_tuple, check_type
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.apex_terms import (BASE,
                                              FULLY_KITTED,
                                              LEVEL_1,
                                              LEVEL_2,
                                              LEVEL_3,
                                              LEVEL_4,
                                              THE_SAME_MAIN_WEAPON,
                                              WITHOUT_RELOAD,
                                              WITH_RELOAD,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import BoolTranslator, SingleTermFinder, Translator
from apex_assistant.speech.translations import FindResult
from apex_assistant.weapon import Loadout, Weapon, WeaponArchetypes


class _LookingForSidearm(IntEnum):
    NOT_REQUESTED = 0
    REQUESTED_BY_DEFAULT = 1
    REQUESTED_EXPLICITLY = 2


class LoadoutTranslator:
    _LOGGER = logging.getLogger()
    _OVERALL_LEVEL_TRANSLATOR = Translator({
        BASE: OverallLevel.BASE,
        LEVEL_1: OverallLevel.LEVEL_1,
        LEVEL_2: OverallLevel.LEVEL_2,
        LEVEL_3: OverallLevel.LEVEL_3,
        (FULLY_KITTED | LEVEL_4): OverallLevel.FULLY_KITTED
    })
    _RELOAD_FINDER = BoolTranslator(WITH_RELOAD, WITHOUT_RELOAD)
    _SIDEARM_FINDER = SingleTermFinder(WITH_SIDEARM)

    def __init__(self, weapon_archetypes: Tuple[WeaponArchetypes, ...], apex_config: ApexConfig):
        check_tuple(WeaponArchetypes, weapon_archetypes=weapon_archetypes)
        check_type(ApexConfig, apex_config=apex_config)

        self._base_weapons: Tuple[Weapon, ...] = tuple(
            base_weapon
            for weapon_archetype in weapon_archetypes
            for base_weapon in weapon_archetype.get_all_base_weapons())
        self._fully_kitted_weapons: Tuple[Weapon, ...] = tuple(
            weapon_archetype.get_fully_kitted_weapon()
            for weapon_archetype in weapon_archetypes)
        archetype_translator = Translator[WeaponArchetypes | None]({
            weapon_archetype.get_base_term(): weapon_archetype
            for weapon_archetype in weapon_archetypes
            if weapon_archetype.get_base_term() is not None})
        archetype_translator.add_term(THE_SAME_MAIN_WEAPON, None)
        self._archetype_translator = archetype_translator
        default_sidearm_name_prop = apex_config.get_default_sidearm_name()
        self._default_sidearm_name_prop = default_sidearm_name_prop
        self._default_sidearm_prop = default_sidearm_name_prop.map(
            self._weapon_name_to_weapon)
        self._reload_by_default_prop = apex_config.get_reload_default()

    def _weapon_name_to_weapon(self, weapon_name: Optional[str]) -> Optional[Weapon]:
        if weapon_name is None:
            return None
        return self.translate_weapon(Words(weapon_name))

    def set_reload_by_default(self, reload_by_default: bool) -> bool:
        check_bool(reload_by_default=reload_by_default)
        return self._reload_by_default_prop.set_value(reload_by_default)

    def set_default_sidearm(self, default_sidearm: Weapon | None) -> Weapon | None:
        check_type(Weapon, optional=True, default_sidearm=default_sidearm)
        old_value = self._default_sidearm_prop.get_value()
        self._default_sidearm_name_prop.set_value(default_sidearm.get_name())
        return old_value

    def translate_weapon(self, words: Words) -> Optional[Weapon]:
        check_type(Words, words=words)
        translation = self._archetype_translator.translate_terms(words)
        words = translation.get_untranslated_words()
        if len(translation) > 1:
            self._LOGGER.warning('More than one value found. Only the last one will be used.')
        archetype = translation.get_latest_value()
        if archetype is None:
            return None
        return archetype.get_best_match(words).get_value()

    def get_default_sidearm(self) -> Weapon | None:
        return self._default_sidearm_prop.get_value()

    def get_reload_by_default(self) -> bool:
        return self._reload_by_default_prop.get_value()

    def get_base_weapons(self) -> tuple[Weapon, ...]:
        return self._base_weapons

    def _to_default_loadout(self, weapon: Weapon) -> Loadout:
        loadout = weapon
        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            loadout = loadout.add_sidearm(default_sidearm)
        if self.get_reload_by_default():
            loadout = loadout.reload()
        return loadout

    def get_fully_kitted_weapons(self) -> Tuple[Weapon, ...]:
        return self._fully_kitted_weapons

    def get_fully_kitted_loadouts(self) -> Tuple[Loadout, ...]:
        return tuple(map(self._to_default_loadout, self._fully_kitted_weapons))

    def translate_loadouts(self, words: Words) -> Generator[Loadout, None, None]:
        check_type(Words, words=words)

        with_reloads = bool(self.is_asking_for_reloads(words))

        # The most recent weapon parsed.
        main_weapon: Weapon | None = None
        # Whether that most recent main weapon is intended to have a sidearm.
        looking_for_sidearm: _LookingForSidearm = _LookingForSidearm.NOT_REQUESTED
        preceding_args: Words | None = None

        for parsed_archetype in self._archetype_translator.translate_terms(words):
            archetype = parsed_archetype.get_value()
            if preceding_args is None:
                preceding_args = parsed_archetype.get_preceding_words()
            following_args = parsed_archetype.get_following_words()
            overall_level = self.get_overall_level(preceding_args)

            if archetype is None:
                # Asking for previous main weapon.
                if looking_for_sidearm is _LookingForSidearm.REQUESTED_EXPLICITLY:
                    self._LOGGER.warning(
                        'Cannot have sidearm of a main weapon, that makes no sense. Ignoring '
                        'sidearm term.')

                if main_weapon is None:
                    self._LOGGER.warning(
                        f'No previous main weapon specified for "{parsed_archetype.get_term()}" to '
                        'make sense. Term will be ignored.')
                    preceding_args = following_args
                    continue
                sidearm_translation = self.is_asking_for_sidearm(following_args)
                # We assume that the only reason you would say "the same main weapon" is to specify
                # a different sidearm.
                looking_for_sidearm = (
                    _LookingForSidearm.REQUESTED_EXPLICITLY if sidearm_translation else
                    _LookingForSidearm.REQUESTED_BY_DEFAULT)
                preceding_args = sidearm_translation.get_untranslated_words()
                continue

            parsed_weapon = archetype.get_best_match(
                words=following_args,
                overall_level=overall_level)
            weapon = parsed_weapon.get_value()
            preceding_args = parsed_weapon.get_untranslated_words()
            if looking_for_sidearm in (_LookingForSidearm.REQUESTED_EXPLICITLY,
                                       _LookingForSidearm.REQUESTED_BY_DEFAULT):
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                looking_for_sidearm = _LookingForSidearm.NOT_REQUESTED
                sidearm = weapon
                loadout = main_weapon.add_sidearm(sidearm)
            elif self.is_asking_for_sidearm(following_args):
                looking_for_sidearm = _LookingForSidearm.REQUESTED_EXPLICITLY
                main_weapon = weapon
                continue
            else:
                main_weapon = weapon
                loadout = self.add_default_sidearm(main_weapon)

            if with_reloads:
                loadout = loadout.reload()

            yield loadout

        if looking_for_sidearm is _LookingForSidearm.REQUESTED_EXPLICITLY:
            self._LOGGER.warning(f'Requested sidearm for {main_weapon}, but didn\'t find one. This '
                                 'incomplete loadout will be ignored.')
        elif looking_for_sidearm is _LookingForSidearm.REQUESTED_BY_DEFAULT:
            yield main_weapon

    @staticmethod
    def get_overall_level(preceding_args: Words) -> OverallLevel:
        overall_level_translations = \
            LoadoutTranslator._OVERALL_LEVEL_TRANSLATOR.translate_terms(preceding_args).values()
        if len(set(overall_level_translations)) > 1:
            LoadoutTranslator._LOGGER.warning('More than one overall level specified. Only the '
                                              'last one will be used.')
        return (overall_level_translations[-1]
                if len(overall_level_translations) != 0
                else OverallLevel.PARSE_WORDS)

    def is_asking_for_reloads(self, arguments: Words) -> bool:
        return LoadoutTranslator._RELOAD_FINDER.translate_term(
            arguments,
            default=self.get_reload_by_default())

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> FindResult:
        return LoadoutTranslator._SIDEARM_FINDER.find_all(archetype_args)

    def add_default_sidearm(self, main_weapon: Weapon) -> Loadout:
        check_type(Weapon, main_weapon=main_weapon)

        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            weapon = main_weapon.add_sidearm(default_sidearm)
        else:
            weapon = main_weapon
        return weapon
