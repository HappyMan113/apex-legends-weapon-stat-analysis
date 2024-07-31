import abc
import logging
from enum import IntEnum
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
                                              THE_SAME_SIDEARM,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import Translator
from apex_assistant.weapon import (FullLoadout, Loadout,
                                   SingleWeaponLoadout,
                                   Weapon,
                                   WeaponArchetypes)


class ArchetypeParseResult:
    pass


class LoadoutsRequestingSidearm:
    def __init__(self,
                 main_loadouts: SingleWeaponLoadout | Tuple[SingleWeaponLoadout],
                 requested_explicitly: bool = True):
        if not isinstance(main_loadouts, tuple):
            main_loadouts = (main_loadouts,)
        check_bool(requested_explicitly=requested_explicitly)
        check_tuple(SingleWeaponLoadout, main_loadouts=main_loadouts)
        self._main_loadouts = main_loadouts
        self._requested_explicitly = requested_explicitly

    def explicit(self) -> bool:
        return self._requested_explicitly

    def get_main_loadouts(self) -> Tuple[SingleWeaponLoadout]:
        return self._main_loadouts

    def __repr__(self):
        return repr(self._main_loadouts)

    def __str__(self):
        return str(self._main_loadouts)


class WithSidearm(IntEnum):
    FIGURE_OUT_LATER = 0
    SAME_SIDEARM = 1
    WITHOUT_SIDEARM = 2


class LoadoutTranslator:
    _LOGGER = logging.getLogger()
    _OVERALL_LEVEL_TRANSLATOR = Translator({
        BASE: OverallLevel.BASE,
        LEVEL_1: OverallLevel.LEVEL_1,
        LEVEL_2: OverallLevel.LEVEL_2,
        LEVEL_3: OverallLevel.LEVEL_3,
        (FULLY_KITTED | LEVEL_4): OverallLevel.FULLY_KITTED
    })
    _SIDEARM_KEYWORD_TRANSLATOR = Translator[WithSidearm]({
        WITH_SIDEARM: WithSidearm.FIGURE_OUT_LATER,
        THE_SAME_SIDEARM: WithSidearm.SAME_SIDEARM
    })

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

    def translate_weapons(self, words: Words) -> Generator[Weapon, None, None]:
        check_type(Words, words=words)
        translation = self._direct_archetypes_translator.translate_terms(words)

        overall_level = self.get_overall_level(translation.get_preamble())
        for term in translation:
            archetypes = term.get_value()
            weapon_args = term.get_following_words()
            translated_value = archetypes.get_best_match(words=weapon_args,
                                                         overall_level=overall_level)

            yield translated_value.get_value()

            overall_level = self.get_overall_level(translated_value.get_untranslated_words())

    def get_fully_kitted_weapons(self) -> Tuple[Weapon, ...]:
        return self._fully_kitted_weapons

    def translate_loadouts(self, words: Words) -> Generator[FullLoadout, None, None]:
        check_type(Words, words=words)

        # The most recent weapon parsed.
        preceding_loadouts: Tuple[FullLoadout, ...] = tuple()
        # Whether that most recent main weapon is intended to have a sidearm.
        translation = self._loadout_parser_translator.translate_terms(words)
        loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm] = None
        preceding_args = translation.get_preamble()

        for parsed_archetype in translation:
            parser = parsed_archetype.get_value()
            following_args = parsed_archetype.get_following_words()

            with_sidearm, following_args = self.is_asking_for_sidearm(following_args)

            loadouts, loadouts_requesting_sidearm, preceding_args = parser.process_archetype_args(
                preceding_args=preceding_args,
                following_args=following_args,
                preceding_loadouts=preceding_loadouts,
                loadouts_requesting_sidearm=loadouts_requesting_sidearm,
                with_sidearm=with_sidearm)

            check_tuple(FullLoadout, loadouts=loadouts)
            check_type(LoadoutsRequestingSidearm,
                       optional=True,
                       loadouts_requesting_sidearm=loadouts_requesting_sidearm)
            check_type(Words, preceding_args=preceding_args)

            for loadout in loadouts:
                yield loadout

            if len(loadouts) != 0:
                preceding_loadouts = loadouts

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
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[WithSidearm, Words]:
        translation = LoadoutTranslator._SIDEARM_KEYWORD_TRANSLATOR.translate_terms(archetype_args)
        with_sidearm = translation.get_latest_value(default_value=WithSidearm.WITHOUT_SIDEARM)
        return with_sidearm, translation.get_untranslated_words()

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
                               preceding_loadouts: Tuple[FullLoadout, ...],
                               loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: WithSidearm) -> \
            Tuple[Tuple[FullLoadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        raise NotImplementedError()

    def get_overall_level(self, words: Words):
        return self._loadout_translator.get_overall_level(words)

    def get_fully_kitted_weapons(self) -> Tuple[Weapon, ...]:
        return self._loadout_translator.get_fully_kitted_weapons()

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[WithSidearm, Words]:
        return LoadoutTranslator.is_asking_for_sidearm(archetype_args)

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
                               preceding_loadouts: Tuple[FullLoadout, ...],
                               loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: WithSidearm) -> \
            Tuple[Tuple[FullLoadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   loadouts_requesting_sidearm=loadouts_requesting_sidearm)
        check_tuple(FullLoadout, preceding_loadouts=preceding_loadouts)
        check_type(WithSidearm, with_sidearm=with_sidearm)

        overall_level = self.get_overall_level(preceding_args)
        parsed_weapon = self._archetype.get_best_match(words=following_args,
                                                       overall_level=overall_level)
        weapon: Weapon = parsed_weapon.get_value()
        following_args = parsed_weapon.get_untranslated_words()

        if loadouts_requesting_sidearm is not None:
            if with_sidearm in (WithSidearm.SAME_SIDEARM, WithSidearm.FIGURE_OUT_LATER):
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                self.warn('Only one sidearm allowed. Extraneous sidearm term will be ignored.')

            loadouts = tuple(main_loadout.add_sidearm(weapon)
                             for main_loadout in loadouts_requesting_sidearm.get_main_loadouts())
        else:
            main_loadouts = tuple(weapon.get_main_loadout_variants())
            if with_sidearm is WithSidearm.FIGURE_OUT_LATER:
                # We need to hold up and look for a sidearm.
                return tuple(), LoadoutsRequestingSidearm(main_loadouts), following_args
            elif with_sidearm is WithSidearm.SAME_SIDEARM:
                sidearms = set(preceding_loadout.get_sidearm()
                               for preceding_loadout in preceding_loadouts)
            elif with_sidearm is WithSidearm.WITHOUT_SIDEARM:
                sidearms = self.get_fully_kitted_weapons()
            else:
                raise RuntimeError(f'Unsupported with_sidearm: {with_sidearm}.')

            loadouts = tuple(FullLoadout(main_loadout, sidearm)
                             for main_loadout in main_loadouts
                             for sidearm in sidearms)

        return loadouts, None, following_args


class SameMainParser(_Parser):
    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(loadout_translator, term=THE_SAME_MAIN_WEAPON)

    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadouts: Tuple[FullLoadout, ...],
                               loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: WithSidearm) -> \
            Tuple[Tuple[FullLoadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   loadouts_requesting_sidearm=loadouts_requesting_sidearm)
        check_tuple(FullLoadout, preceding_loadouts=preceding_loadouts)
        check_type(WithSidearm, with_sidearm=with_sidearm)

        if loadouts_requesting_sidearm is not None and loadouts_requesting_sidearm.explicit():
            self.warn(
                'Cannot have sidearm of a main weapon; that makes no sense. Ignoring sidearm term.')

        if len(preceding_loadouts) == 0:
            self.warn(f'No previous loadout specified for "{THE_SAME_MAIN_WEAPON}" to make sense. '
                      'Term will be ignored.')
            return tuple(), None, following_args

        main_loadouts = tuple(set(
            main_loadout
            for preceding_loadout in preceding_loadouts
            for main_loadout in preceding_loadout.get_main_weapon().get_main_loadout_variants()))

        if with_sidearm is WithSidearm.FIGURE_OUT_LATER:
            requested_explicitly = True
        elif with_sidearm is WithSidearm.WITHOUT_SIDEARM:
            requested_explicitly = False
        elif with_sidearm is WithSidearm.SAME_SIDEARM:
            self.warn('Same main weapon with same sidearm would result in a duplicate loadout. '
                      'Skipping.')
            return tuple(), None, following_args
        else:
            raise RuntimeError(f'Unsupported with_sidearm: {with_sidearm}')

        loadouts_requesting_sidearm = LoadoutsRequestingSidearm(main_loadouts, requested_explicitly)
        return tuple(), loadouts_requesting_sidearm, following_args


class ReversedParser(_Parser):
    def __init__(self, loadout_translator: LoadoutTranslator):
        super().__init__(loadout_translator=loadout_translator, term=REVERSED)

    def process_archetype_args(self,
                               preceding_args: Words,
                               following_args: Words,
                               preceding_loadouts: Tuple[FullLoadout, ...],
                               loadouts_requesting_sidearm: Optional[LoadoutsRequestingSidearm],
                               with_sidearm: WithSidearm) -> \
            Tuple[Tuple[Loadout, ...], Optional[LoadoutsRequestingSidearm], Words]:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        check_type(LoadoutsRequestingSidearm,
                   optional=True,
                   loadouts_requesting_sidearm=loadouts_requesting_sidearm)
        check_tuple(FullLoadout, preceding_loadouts=preceding_loadouts)
        check_type(WithSidearm, with_sidearm=with_sidearm)

        if loadouts_requesting_sidearm is not None:
            self.warn('Sidearm term was not followed by a sidearm. Ignoring it.')
        if with_sidearm in (WithSidearm.SAME_SIDEARM, WithSidearm.FIGURE_OUT_LATER):
            self.warn('Sidearm of reverse term is implied. Ignoring sidearm term.')
        if len(preceding_loadouts) == 0:
            self.warn(f'No previous loadout to reverse.')

        reversed_loadouts = tuple(
            FullLoadout(main_loadout=main_loadout, sidearm=preceding_loadout.get_main_weapon())
            for preceding_loadout in preceding_loadouts
            for main_loadout in preceding_loadout.get_sidearm().get_main_loadout_variants())
        return reversed_loadouts, None, following_args
