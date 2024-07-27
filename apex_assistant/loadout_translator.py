import logging
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
                                              SINGLE_SHOT,
                                              THE_SAME_MAIN_WEAPON,
                                              WITHOUT_RELOAD,
                                              WITH_RELOAD,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import BoolTranslator, SingleTermFinder, Translator
from apex_assistant.weapon import (Loadout,
                                   MainLoadout,
                                   NonReloadingLoadout,
                                   Weapon,
                                   WeaponArchetypes)


class ArchetypeParseResult:
    pass


class LoadoutRequestingSidearm:
    def __init__(self, main_loadout: MainLoadout, requested_explicitly: bool = True):
        check_bool(requested_explicitly=requested_explicitly)
        check_type(MainLoadout, main_loadout=main_loadout)
        self._main_loadout = main_loadout
        self._requested_explicitly = requested_explicitly

    def explicit(self) -> bool:
        return self._requested_explicitly

    def get_main_loadout(self) -> MainLoadout:
        return self._main_loadout

    def __repr__(self):
        return repr(self._main_loadout)

    def __str__(self):
        return str(self._main_loadout)


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
    _SINGLE_SHOT_FINDER = SingleTermFinder(SINGLE_SHOT)

    def __init__(self, weapon_archetypes: Tuple[WeaponArchetypes, ...], apex_config: ApexConfig):
        check_tuple(WeaponArchetypes, weapon_archetypes=weapon_archetypes)
        check_type(ApexConfig, apex_config=apex_config)

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

    def translate_main_loadout(self, words: Words) -> Optional[MainLoadout]:
        generator = self.translate_loadouts(words, main_loadouts_only=True)
        main_loadout: Optional[MainLoadout] = next(generator, None)
        if main_loadout is None:
            try:
                next(generator)
                self._LOGGER.warning('Only one loadout expected. Only keeping the first one.')
            except StopIteration:
                pass
        elif not isinstance(main_loadout, MainLoadout):
            raise RuntimeError(f'Expected a MainLoadout, not {main_loadout}.')

        return main_loadout

    def translate_loadouts(self, words: Words, main_loadouts_only: bool = False) -> \
            Generator[Loadout, None, None]:
        check_type(Words, words=words)
        check_bool(main_loadouts_only=main_loadouts_only)

        with_reloads = (not main_loadouts_only) and bool(self.is_asking_for_reloads(words))

        # The most recent weapon parsed.
        loadout: Loadout | None = None
        # Whether that most recent main weapon is intended to have a sidearm.
        translation = self._archetype_translator.translate_terms(words)
        loadout_requesting_sidearm: Optional[LoadoutRequestingSidearm] = None
        preceding_args = translation.get_preamble()

        for parsed_archetype in translation:
            archetype = parsed_archetype.get_value()
            following_args = parsed_archetype.get_following_words()
            loadout, loadout_requesting_sidearm, preceding_args = \
                self.process_archetype_args(archetype=archetype,
                                            preceding_args=preceding_args,
                                            following_args=following_args,
                                            preceding_loadout=loadout,
                                            loadout_requesting_sidearm=loadout_requesting_sidearm,
                                            with_reloads=with_reloads,
                                            main_loadouts_only=main_loadouts_only)

            if loadout is not None:
                yield loadout

        if loadout_requesting_sidearm is None:
            return

        if loadout_requesting_sidearm.explicit():
            self._LOGGER.warning(
                f'Requested sidearm for {loadout_requesting_sidearm}, but didn\'t find one. This '
                'incomplete loadout will be ignored.')
        else:
            yield loadout_requesting_sidearm.get_main_loadout()

    def process_archetype_args(self,
                               archetype: Optional[WeaponArchetypes],
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadout: Optional[Loadout],
                               loadout_requesting_sidearm: Optional[LoadoutRequestingSidearm],
                               with_reloads: bool,
                               main_loadouts_only: bool) -> \
            Tuple[Optional[Loadout], Optional[LoadoutRequestingSidearm], Words]:
        check_type(WeaponArchetypes, optional=True, archetype=archetype)
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutRequestingSidearm,
                   optional=True,
                   looking_for_sidearm=loadout_requesting_sidearm)
        check_type(Loadout, optional=True, preceding_loadout=preceding_loadout)
        check_bool(with_reloads=with_reloads,
                   main_loadouts_only=main_loadouts_only)

        overall_level = self.get_overall_level(preceding_args)
        if archetype is None:
            # Asking for previous main weapon.
            if main_loadouts_only:
                self._LOGGER.warning(
                    f'"{THE_SAME_MAIN_WEAPON}" term not allowed here. Ignoring it.')
                return None, None, following_args

            if loadout_requesting_sidearm is not None and loadout_requesting_sidearm.explicit():
                self._LOGGER.warning(
                    'Cannot have sidearm of a main weapon; that makes no sense. Ignoring sidearm '
                    'term.')

            if preceding_loadout is None:
                self._LOGGER.warning(
                    f'No previous loadout specified for "{THE_SAME_MAIN_WEAPON}" to make sense. '
                    'Term will be ignored.')
                return None, None, following_args

            requested_sidearm, following_args = self.is_asking_for_sidearm(following_args)
            # We assume that the only reason you would say "the same main weapon" is to specify a
            # different sidearm.
            loadout_requesting_sidearm = LoadoutRequestingSidearm(
                main_loadout=preceding_loadout.get_main_loadout(),
                requested_explicitly=requested_sidearm)
            return None, loadout_requesting_sidearm, following_args

        parsed_weapon = archetype.get_best_match(words=following_args, overall_level=overall_level)
        weapon: Weapon = parsed_weapon.get_value()
        following_args = parsed_weapon.get_untranslated_words()

        single_shot, following_args = self.is_asking_for_single_shot(following_args)
        if not single_shot:
            single_shot, _ = self.is_asking_for_single_shot(preceding_args)

        with_sidearm, following_args = self.is_asking_for_sidearm(following_args)
        if with_sidearm and main_loadouts_only:
            self._LOGGER.warning(f'"{WITH_SIDEARM}" term not allowed here. Ignoring it.')
            with_sidearm = False

        if loadout_requesting_sidearm is not None:
            if single_shot:
                self._LOGGER.warning(
                    'Sidearms cannot be single shot. Single shot term will be ignored.')

            if with_sidearm:
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                self._LOGGER.warning(
                    'Only one sidearm allowed. Extraneous sidearm term will be ignored.')

            loadout = loadout_requesting_sidearm.get_main_loadout().add_sidearm(weapon)
            loadout_requesting_sidearm = None
        else:
            main_loadout: MainLoadout = weapon.single_shot() if single_shot else weapon
            if with_sidearm:
                # We need to hold up and look for a sidearm.
                return None, LoadoutRequestingSidearm(main_loadout), following_args
            loadout = (main_loadout
                       if main_loadouts_only else
                       self.add_default_sidearm(main_loadout))

        if with_reloads:
            loadout = loadout.reload()

        return loadout, loadout_requesting_sidearm, following_args

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
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[bool, Words]:
        translation = LoadoutTranslator._SIDEARM_FINDER.find_all(archetype_args)
        return bool(translation), translation.get_untranslated_words()

    @staticmethod
    def is_asking_for_single_shot(archetype_args: Words) -> Tuple[bool, Words]:
        translation = LoadoutTranslator._SINGLE_SHOT_FINDER.find_all(archetype_args)
        return bool(translation), translation.get_untranslated_words()

    def add_default_sidearm(self, main_weapon: MainLoadout) -> NonReloadingLoadout:
        check_type(MainLoadout, main_weapon=main_weapon)

        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            weapon = main_weapon.add_sidearm(default_sidearm)
        else:
            weapon = main_weapon
        return weapon
