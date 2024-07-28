import abc
import logging
from typing import Generator, List, Optional, Tuple

from apex_assistant.checker import check_bool, check_tuple, check_type
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.apex_terms import (BASE,
                                              FULLY_KITTED,
                                              LEVEL_1,
                                              LEVEL_2,
                                              LEVEL_3,
                                              LEVEL_4,
                                              REVERSED,
                                              THE_SAME_MAIN_WEAPON,
                                              WITHOUT_RELOAD,
                                              WITH_RELOAD,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import BoolTranslator, SingleTermFinder, Translator
from apex_assistant.weapon import (FullLoadout, Loadout,
                                   MainLoadout,
                                   NonReloadingLoadout,
                                   Weapon,
                                   WeaponArchetypes)


class ArchetypeParseResult:
    pass


class LoadoutsRequestingSidearm:
    def __init__(self,
                 main_loadouts: MainLoadout | Tuple[MainLoadout],
                 requested_explicitly: bool = True):
        if not isinstance(main_loadouts, tuple):
            main_loadouts = (main_loadouts,)
        check_bool(requested_explicitly=requested_explicitly)
        check_tuple(MainLoadout, main_loadouts=main_loadouts)
        self._main_loadouts = main_loadouts
        self._requested_explicitly = requested_explicitly

    def explicit(self) -> bool:
        return self._requested_explicitly

    def get_main_loadouts(self) -> Tuple[MainLoadout]:
        return self._main_loadouts

    def __repr__(self):
        return repr(self._main_loadouts)

    def __str__(self):
        return str(self._main_loadouts)


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

        self._fully_kitted_weapons: Tuple[Weapon, ...] = tuple(
            weapon_archetype.get_fully_kitted_weapon()
            for weapon_archetype in weapon_archetypes)
        self._direct_archetypes_translator = Translator[WeaponArchetypes]({
            weapon_archetype.get_base_term(): weapon_archetype
            for weapon_archetype in weapon_archetypes})

        parsers: List[_Parser] = [ArchetypeParser(self, weapon_archetype)
                                  for weapon_archetype in weapon_archetypes]
        parsers.append(SameMainParser(self))
        parsers.append(ReversedParser(self))
        self._loadout_parser_translator = Translator[_Parser]({
            parser.get_term(): parser
            for parser in parsers})

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
        translation = self._direct_archetypes_translator.translate_terms(words)
        words = translation.get_untranslated_words()
        if len(translation) > 1:
            self._LOGGER.warning('More than one value found. Only the last one will be used.')
        archetypes = translation.get_latest_value()
        if archetypes is None:
            return None
        return archetypes.get_best_match(words).get_value()

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

    def translate_any_loadouts(self, words: Words) -> Generator[Loadout, None, None]:
        check_type(Words, words=words)

        with_reloads = bool(self.is_asking_for_reloads(words))

        # The most recent weapon parsed.
        preceding_loadout: Optional[Loadout] = None
        # Whether that most recent main weapon is intended to have a sidearm.
        translation = self._loadout_parser_translator.translate_terms(words)
        loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm] = None
        preceding_args = translation.get_preamble()

        for parsed_archetype in translation:
            parser = parsed_archetype.get_value()
            following_args = parsed_archetype.get_following_words()

            with_sidearm, following_args = self.is_asking_for_sidearm(following_args)

            loadouts, loadouts_requesting_sidearm, preceding_args = \
                parser.process_archetype_args(
                    preceding_args=preceding_args,
                    following_args=following_args,
                    preceding_loadout=preceding_loadout,
                    loadout_requesting_sidearm=loadouts_requesting_sidearm,
                    with_sidearm=with_sidearm,
                    with_reloads=with_reloads)
            if not isinstance(loadouts, tuple):
                loadouts = (loadouts,)

            check_tuple(Loadout, loadouts=loadouts)
            check_type(LoadoutsRequestingSidearm,
                       optional=True,
                       loadouts_requesting_sidearm=loadouts_requesting_sidearm)
            check_type(Words, preceding_args=preceding_args)

            for loadout in loadouts:
                yield loadout

            if len(loadouts) != 0:
                preceding_loadout = loadouts[-1]

        if loadouts_requesting_sidearm is None:
            return

        if loadouts_requesting_sidearm.explicit():
            self._LOGGER.warning(
                f'Requested sidearm for {loadouts_requesting_sidearm}, but didn\'t find one. This '
                'incomplete loadout will be ignored.')
        else:
            for loadout in loadouts_requesting_sidearm.get_main_loadouts():
                yield loadout

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[bool, Words]:
        translation = LoadoutTranslator._SIDEARM_FINDER.find_all(archetype_args)
        return bool(translation), translation.get_untranslated_words()

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

    def add_default_sidearm(self, main_weapon: MainLoadout) -> NonReloadingLoadout:
        check_type(MainLoadout, main_weapon=main_weapon)

        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            weapon = main_weapon.add_sidearm(default_sidearm)
        else:
            weapon = main_weapon
        return weapon


class _Parser(abc.ABC):
    _LOGGER = logging.getLogger()

    def __init__(self, loadout_translator: LoadoutTranslator, term: RequiredTerm):
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        check_type(RequiredTerm, term=term)
        self._loadout_translator = loadout_translator
        self._term = term

    def get_term(self) -> RequiredTerm:
        return self._term

    @abc.abstractmethod
    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadout: Optional[Loadout],
                               loadout_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: bool,
                               with_reloads: bool) -> \
            Tuple[Loadout | Tuple[Loadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        raise NotImplementedError()

    def get_overall_level(self, words: Words):
        return self._loadout_translator.get_overall_level(words)

    def add_default_sidearm(self, main_loadout: MainLoadout):
        return self._loadout_translator.add_default_sidearm(main_loadout)

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[bool, Words]:
        return LoadoutTranslator.is_asking_for_sidearm(archetype_args)

    @staticmethod
    def get_single_shot_variants(main_weapon: Weapon) -> Tuple[MainLoadout, ...]:
        check_type(Weapon, main_weapon=main_weapon)
        return ((main_weapon, main_weapon.single_shot())
                if main_weapon.is_single_shot_advisable() else
                (main_weapon,))

    @staticmethod
    def warn(message: str):
        _Parser._LOGGER.warning(message)


class ArchetypeParser(_Parser):
    def __init__(self, loadout_translator: LoadoutTranslator, archetypes: WeaponArchetypes):
        check_type(WeaponArchetypes, archetypes=archetypes)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        super().__init__(loadout_translator=loadout_translator,
                         term=archetypes.get_base_term())
        self._archetype = archetypes

    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadout: Optional[Loadout],
                               loadout_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: bool,
                               with_reloads: bool) -> \
            Tuple[Loadout | Tuple[Loadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   looking_for_sidearm=loadout_requesting_sidearm)
        check_type(Loadout, optional=True, preceding_loadout=preceding_loadout)
        check_bool(with_reloads=with_reloads)

        overall_level = self.get_overall_level(preceding_args)
        parsed_weapon = self._archetype.get_best_match(words=following_args,
                                                       overall_level=overall_level)
        weapon: Weapon = parsed_weapon.get_value()
        following_args = parsed_weapon.get_untranslated_words()

        if loadout_requesting_sidearm is not None:
            if with_sidearm:
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                self.warn('Only one sidearm allowed. Extraneous sidearm term will be ignored.')

            loadouts = tuple(main_loadout.add_sidearm(weapon)
                             for main_loadout in loadout_requesting_sidearm.get_main_loadouts())
        else:
            main_loadouts = self.get_single_shot_variants(weapon)
            if with_sidearm:
                # We need to hold up and look for a sidearm.
                return tuple(), LoadoutsRequestingSidearm(main_loadouts), following_args
            loadouts = tuple(self.add_default_sidearm(main_loadout)
                             for main_loadout in main_loadouts)

        if with_reloads:
            loadouts = tuple(loadout.reload() for loadout in loadouts)

        return loadouts, None, following_args


class SameMainParser(_Parser):
    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(loadout_translator, term=THE_SAME_MAIN_WEAPON)

    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadout: Optional[Loadout],
                               loadout_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: bool,
                               with_reloads: bool) -> \
            Tuple[Loadout | Tuple[Loadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   looking_for_sidearm=loadout_requesting_sidearm)
        check_type(Loadout, optional=True, preceding_loadout=preceding_loadout)

        if loadout_requesting_sidearm is not None and loadout_requesting_sidearm.explicit():
            self.warn(
                'Cannot have sidearm of a main weapon; that makes no sense. Ignoring sidearm term.')

        if preceding_loadout is None:
            self.warn(f'No previous loadout specified for "{THE_SAME_MAIN_WEAPON}" to make sense. '
                      'Term will be ignored.')
            return tuple(), None, following_args

        requested_sidearm, following_args = self.is_asking_for_sidearm(following_args)

        main_loadouts = self.get_single_shot_variants(preceding_loadout.get_main_weapon())

        if with_sidearm:
            loadout_requesting_sidearm = LoadoutsRequestingSidearm(
                main_loadouts,
                requested_explicitly=requested_sidearm)
            return tuple(), loadout_requesting_sidearm, following_args

        return main_loadouts, None, following_args


class ReversedParser(_Parser):
    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(loadout_translator=loadout_translator, term=REVERSED)

    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadout: Optional[Loadout],
                               loadout_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: bool,
                               with_reloads: bool) -> \
            Tuple[Loadout | Tuple[Loadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   looking_for_sidearm=loadout_requesting_sidearm)
        check_type(Loadout, optional=True, preceding_loadout=preceding_loadout)
        if loadout_requesting_sidearm is not None:
            self.warn('Sidearm term was not followed by a sidearm. Ignoring it.')
        if with_sidearm:
            self.warn('Sidearm of reverse term is implied. Ignoring sidearm term.')

        reversed_loadouts = self._reverse_loadout(preceding_loadout=preceding_loadout,
                                                  with_reloads=with_reloads)
        return reversed_loadouts, None, following_args

    def _reverse_loadout(self,
                         preceding_loadout: Optional[Loadout],
                         with_reloads: bool) -> \
            Tuple[Loadout]:
        if preceding_loadout is None:
            self.warn(f'No previous loadout to reverse. Ignoring "{self.get_term()}" term.')
            return tuple()

        main_weapon = preceding_loadout.get_sidearm()
        if main_weapon is None:
            self.warn(f'Previous loadout had no sidearm; can\'t reverse it.')
            return tuple()

        main_loadouts = self.get_single_shot_variants(main_weapon)
        sidearm = preceding_loadout.get_main_loadout().get_main_weapon()

        return tuple(self._get_loadout(main_loadout=main_loadout,
                                       sidearm=sidearm,
                                       with_reloads=with_reloads)
                     for main_loadout in main_loadouts)

    @staticmethod
    def _get_loadout(main_loadout: MainLoadout,
                     sidearm: Weapon,
                     with_reloads: bool) -> Loadout:
        loadout = FullLoadout(main_loadout, sidearm)
        if with_reloads:
            loadout = loadout.reload()
        return loadout
