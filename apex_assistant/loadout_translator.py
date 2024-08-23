import abc
import logging
from enum import IntEnum
from typing import Collection, Generator, Iterable, List, Optional, Tuple, TypeVar

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.legend import Legend
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.apex_terms import (BASE,
                                              FULLY_KITTED,
                                              LEVEL_1,
                                              LEVEL_2,
                                              LEVEL_3,
                                              LEVEL_4,
                                              THE_SAME_MAIN_WEAPON,
                                              THE_SAME_SIDEARM,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import Translator
from apex_assistant.weapon import FullLoadout, SingleWeaponLoadout, Weapon, WeaponArchetypes
from apex_assistant.weapon_class import WeaponClass


class ArchetypeParseResult:
    pass


class WithSidearm(IntEnum):
    FIGURE_OUT_LATER = 0
    SAME_SIDEARM = 1
    WITHOUT_SIDEARM = 2


T = TypeVar('T')


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
        self._weapon_archetypes = weapon_archetypes
        self._apex_config = apex_config
        self._direct_archetypes_translator = Translator[WeaponArchetypes]({
            weapon_archetype.get_base_term(): weapon_archetype
            for weapon_archetype in weapon_archetypes})

        parsers: List[_WeaponParser] = [ArchetypeParser(self, weapon_archetype)
                                        for weapon_archetype in weapon_archetypes]
        parsers.append(SameMainParser())
        # noinspection PyTypeChecker
        parsers.extend(WeaponClassParser(self, weapon_class) for weapon_class in WeaponClass)

        self._loadout_parser_translator = Translator[_WeaponParser]({
            parser.get_term(): parser
            for parser in parsers})

    def set_legend(self, legend: Optional[Legend]) -> Optional[Legend]:
        return self._apex_config.set_legend(legend)

    def get_legend(self) -> Optional[Legend]:
        return self._apex_config.get_legend()

    def _legend_fits_archetypes(self, archetypes: WeaponArchetypes) -> bool:
        return self._legend_fits(archetypes.get_associated_legend())

    def _legend_fits_loadout(self, loadout: FullLoadout) -> bool:
        return all(self.legend_fits_weapon(weapon) for weapon in loadout.get_weapons())

    def legend_fits_weapon(self, weapon: SingleWeaponLoadout) -> bool:
        return self._legend_fits(weapon.get_archetype().get_associated_legend())

    def _legend_fits(self, associated_legend: Optional[Legend]) -> bool:
        return associated_legend is None or self.get_legend() is associated_legend

    def translate_weapons(self, words: Words) -> Generator[Weapon, None, None]:
        check_type(Words, words=words)
        translation = self._direct_archetypes_translator.translate_terms(words)

        overall_level = self.get_overall_level(translation.get_preamble())
        legend = self.get_legend()

        for term in translation:
            archetypes = term.get_value()

            weapon_args = term.get_following_words()
            translated_value = archetypes.get_best_match(words=weapon_args,
                                                         overall_level=overall_level,
                                                         legend=legend)

            if self._legend_fits_archetypes(archetypes):
                yield translated_value
            else:
                self._LOGGER.warning(f'Weapon {archetypes.get_base_term()} is associated with '
                                     f'{archetypes.get_associated_legend()}, and current legend '
                                     f'is configured as {self.get_legend()}. Skipping weapon.')

            overall_level = self.get_overall_level(translated_value.get_untranslated_words())

    def iget_fully_kitted_weapons(self,
                                  include_care_package_weapons: bool = True,
                                  include_non_hopped_up_weapons: bool = True) -> \
            Generator[Weapon, None, None]:
        legend = self.get_legend()
        return (fully_kitted_weapon
                for weapon_archetypes in self._weapon_archetypes
                if (self._legend_fits_archetypes(weapon_archetypes) and
                    (include_care_package_weapons or not weapon_archetypes.is_care_package()))
                for fully_kitted_weapon in
                weapon_archetypes.get_fully_kitted_weapons(
                    legend=legend,
                    include_non_hopped_up=include_non_hopped_up_weapons))

    def get_fully_kitted_weapons(self,
                                 include_care_package_weapons: bool = True,
                                 include_non_hopped_up_weapons: bool = True) -> \
            Tuple[Weapon, ...]:
        return tuple(self.iget_fully_kitted_weapons(
            include_care_package_weapons=include_care_package_weapons,
            include_non_hopped_up_weapons=include_non_hopped_up_weapons))

    def translate_full_loadouts(self, words: Words) -> Generator[FullLoadout, None, None]:
        check_type(Words, words=words)

        explicit_loadouts: set[FullLoadout] = set()
        either_slot_weapons: list[Weapon] = []

        found_a_full_loadout = False
        for result in self._get_mains_and_sidearms(words):
            if not result.has_sidearms():
                either_slot_weapons.extend(result.get_main_weapons())

            found_a_full_loadout = found_a_full_loadout or result.has_full_loadouts()
            for full_loadout in self._filter_loadouts(result.get_full_loadouts(),
                                                      explicit_loadouts=explicit_loadouts):
                yield full_loadout
                explicit_loadouts.add(full_loadout)

        if len(either_slot_weapons) > 1:
            for full_loadout in self._filter_loadouts(FullLoadout.get_loadouts(either_slot_weapons),
                                                      explicit_loadouts=explicit_loadouts):
                yield full_loadout
        elif len(either_slot_weapons) == 1:
            for full_loadout in self._filter_loadouts(
                    self.get_loadouts_with_other_fully_kitted(either_slot_weapons[0]),
                    explicit_loadouts=explicit_loadouts):
                yield full_loadout

    def _filter_loadouts(self,
                         loadouts: Iterable[FullLoadout],
                         explicit_loadouts: Collection[FullLoadout]) -> \
            Generator[FullLoadout, None, None]:
        for loadout in loadouts:
            if not self._legend_fits_loadout(loadout):
                self._LOGGER.warning(
                    f'Loadout {loadout} contains one or more weapons which are not associated with '
                    f'{self.get_legend()}. Skipping.')
                continue

            if loadout in explicit_loadouts:
                self._LOGGER.warning(f'Duplicate loadout {loadout} will be skipped.')
                continue

            yield loadout

    def get_loadouts_with_other_fully_kitted(self, either_slot_weapon: Weapon) -> \
            Generator[FullLoadout, None, None]:
        fully_kitted_weapons = self.get_fully_kitted_weapons()
        for sidearm in fully_kitted_weapons:
            yield FullLoadout(either_slot_weapon, sidearm)

        for main_weapon in fully_kitted_weapons:
            yield FullLoadout(main_weapon, either_slot_weapon)

    def _get_mains_and_sidearms(self, words: Words) -> \
            Generator['MainWeaponsAndSidearms', None, None]:
        check_type(Words, words=words)

        translation = self._loadout_parser_translator.translate_terms(words)
        prev_result = MainWeaponsAndSidearms()
        cur_result: Optional[MainWeaponsAndSidearms] = None
        preceding_args = translation.get_preamble()
        looking_for_sidearm = False

        for parsed_parser in translation:
            following_args = parsed_parser.get_following_words()

            with_sidearm, following_args = self.is_asking_for_sidearm(following_args)
            result = parsed_parser.get_value().process_archetype_args(
                prev_result=prev_result,
                preceding_args=preceding_args,
                following_args=following_args)
            preceding_args = result.get_untranslated_following_args()

            mains_and_sidearms, looking_for_sidearm = self._get_mains_and_sidearms2(
                prev_result=prev_result,
                weapons=result.get_weapons(),
                looking_for_sidearm=looking_for_sidearm,
                with_sidearm=with_sidearm)

            if not looking_for_sidearm:
                yield mains_and_sidearms

            prev_result = mains_and_sidearms.with_fallback(prev_result)
            cur_result = mains_and_sidearms

        if looking_for_sidearm:
            self._LOGGER.warning(
                f'Requested sidearm for {cur_result.get_main_weapons()}, but didn\'t find one. '
                'This incomplete loadout will be ignored.')

    @staticmethod
    def _get_mains_and_sidearms2(prev_result: 'MainWeaponsAndSidearms',
                                 weapons: Tuple[Weapon, ...],
                                 looking_for_sidearm: bool,
                                 with_sidearm: WithSidearm) -> \
            Tuple['MainWeaponsAndSidearms', bool]:
        if looking_for_sidearm:
            if with_sidearm is not WithSidearm.WITHOUT_SIDEARM:
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                LoadoutTranslator._LOGGER.warning(
                    'Only one sidearm allowed. Extraneous sidearm term will be ignored.')

            mains_and_sidearms = MainWeaponsAndSidearms(main_weapons=prev_result.get_main_weapons(),
                                                        sidearms=weapons)
            looking_for_sidearm = False

        elif with_sidearm is WithSidearm.FIGURE_OUT_LATER:
            # We need to hold up and look for a sidearm.
            mains_and_sidearms = MainWeaponsAndSidearms(main_weapons=weapons)
            looking_for_sidearm = True

        elif with_sidearm is WithSidearm.SAME_SIDEARM:
            sidearms = prev_result.get_sidearms()
            if len(sidearms) == 0:
                LoadoutTranslator._LOGGER.warning(
                    'No sidearm was specified previously. Skipping "same sidearm" loadout.')
                mains_and_sidearms = MainWeaponsAndSidearms()
            else:
                mains_and_sidearms = MainWeaponsAndSidearms(main_weapons=weapons,
                                                            sidearms=sidearms)
            looking_for_sidearm = False

        elif with_sidearm is WithSidearm.WITHOUT_SIDEARM:
            mains_and_sidearms = MainWeaponsAndSidearms(main_weapons=weapons)
            looking_for_sidearm = False

        else:
            raise RuntimeError(f'Unsupported with_sidearm: {with_sidearm}.')

        return mains_and_sidearms, looking_for_sidearm

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> Tuple[WithSidearm, Words]:
        translation = LoadoutTranslator._SIDEARM_KEYWORD_TRANSLATOR.translate_terms(archetype_args)
        with_sidearm = translation.get_latest_value(default_value=WithSidearm.WITHOUT_SIDEARM)
        return with_sidearm, translation.get_untranslated_words()

    @staticmethod
    def get_overall_level(preceding_args: Words) -> OverallLevel:
        overall_level_translation = \
            LoadoutTranslator._OVERALL_LEVEL_TRANSLATOR.translate_terms(preceding_args)
        overall_level_translations = overall_level_translation.values()
        if len(set(overall_level_translations)) > 1:
            LoadoutTranslator._LOGGER.warning('More than one overall level specified. Only the '
                                              'last one will be used.')

        return (overall_level_translations[-1]
                if len(overall_level_translations) != 0
                else OverallLevel.PARSE_WORDS)


class MainWeaponsAndSidearms:
    def __init__(self,
                 main_weapons: Weapon | Tuple[Weapon, ...] = tuple[Weapon, ...](),
                 sidearms: Weapon | Tuple[Weapon, ...] = tuple[Weapon, ...]()):
        if not isinstance(main_weapons, Tuple):
            check_type(Weapon, main_weapons=main_weapons)
            main_weapons: Tuple[Weapon, ...] = (main_weapons,)
        check_tuple(Weapon, main_weapons=main_weapons, sidearm=sidearms)
        if len(sidearms) > 0 and len(main_weapons) == 0:
            raise ValueError('Cannot specify a sidearm without a main weapon.')

        self._main_weapons = main_weapons
        self._sidearms = sidearms

    def with_fallback(self, main_weapons_and_sidearms: 'MainWeaponsAndSidearms') -> \
            'MainWeaponsAndSidearms':
        main_weapons = (self._main_weapons if len(self._main_weapons) > 0 else
                        main_weapons_and_sidearms._main_weapons)
        sidearms = (self._sidearms if len(self._sidearms) > 0 else
                    main_weapons_and_sidearms._sidearms)
        return MainWeaponsAndSidearms(main_weapons=main_weapons,
                                      sidearms=sidearms)

    def get_full_loadouts(self) -> Generator[FullLoadout, None, None]:
        for main_weapon in self._main_weapons:
            for sidearm in self._sidearms:
                yield FullLoadout(main_weapon, sidearm)

    def get_main_weapons(self) -> Tuple[Weapon, ...]:
        return self._main_weapons

    def has_main_weapons(self) -> bool:
        return len(self._main_weapons) > 0

    def has_sidearms(self) -> bool:
        return len(self._sidearms) > 0

    def get_sidearms(self) -> Tuple[Weapon, ...]:
        return self._sidearms

    def has_full_loadouts(self):
        return self.has_main_weapons() and self.has_sidearms()


class ParserResult:
    def __init__(self,
                 untranslated_following_args: Words,
                 main_weapons: Weapon | Tuple[Weapon, ...] = tuple[Weapon, ...]()):
        check_type(Words, untranslated_following_args=untranslated_following_args)
        if not isinstance(main_weapons, Tuple):
            check_type(Weapon, main_weapons=main_weapons)
            main_weapons: Tuple[Weapon, ...] = (main_weapons,)
        check_tuple(Weapon, main_weapons=main_weapons)

        self._untranslated_following_args = untranslated_following_args
        self._main_weapons = main_weapons

    def get_untranslated_following_args(self) -> Words:
        return self._untranslated_following_args

    def get_weapons(self) -> Tuple[Weapon, ...]:
        return self._main_weapons


class _WeaponParser(abc.ABC):
    def __init__(self, term: RequiredTerm):
        check_type(RequiredTerm, term=term)
        self._term = term

    def get_term(self) -> RequiredTerm:
        return self._term

    @abc.abstractmethod
    def process_archetype_args(self,
                               prev_result: MainWeaponsAndSidearms,
                               preceding_args: Words,
                               following_args: Words) -> ParserResult:
        raise NotImplementedError()


class ArchetypeParser(_WeaponParser):
    def __init__(self, loadout_translator: LoadoutTranslator, archetypes: WeaponArchetypes):
        check_type(WeaponArchetypes, archetypes=archetypes)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        super().__init__(term=archetypes.get_base_term())
        self._archetype = archetypes
        self._loadout_translator = loadout_translator

    def get_overall_level(self, words: Words) -> OverallLevel:
        return self._loadout_translator.get_overall_level(words)

    def get_legend(self) -> Optional[Legend]:
        return self._loadout_translator.get_legend()

    def get_best_match(self, preceding_args: Words, following_args: Words):
        overall_level = self.get_overall_level(preceding_args)
        parsed_weapon = self._archetype.get_best_match(words=following_args,
                                                       overall_level=overall_level,
                                                       legend=self.get_legend())
        return parsed_weapon

    def process_archetype_args(self,
                               prev_result: MainWeaponsAndSidearms,
                               preceding_args: Words,
                               following_args: Words) -> ParserResult:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)

        parsed_weapon = self.get_best_match(preceding_args=preceding_args,
                                            following_args=following_args)
        weapon = parsed_weapon.get_value()
        following_args = parsed_weapon.get_untranslated_words()
        return ParserResult(untranslated_following_args=following_args, main_weapons=weapon)


class SameMainParser(_WeaponParser):
    def __init__(self):
        super().__init__(THE_SAME_MAIN_WEAPON)

    def process_archetype_args(self,
                               prev_result: MainWeaponsAndSidearms,
                               preceding_args: Words,
                               following_args: Words) -> ParserResult:
        check_type(Words, preceding_args=preceding_args, following_args=following_args)
        return ParserResult(untranslated_following_args=following_args,
                            main_weapons=prev_result.get_main_weapons())


class WeaponClassParser(_WeaponParser):
    def __init__(self, loadout_translator: LoadoutTranslator, weapon_class: WeaponClass):
        check_type(WeaponClass, weapon_class=weapon_class)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        super().__init__(weapon_class.get_term())

        weapons_of_class = tuple(weapon
                                 for weapon in loadout_translator.iget_fully_kitted_weapons()
                                 if weapon.get_weapon_class() is weapon_class)
        weapon_set = frozenset(weapons_of_class)
        if len(weapon_set) < len(weapons_of_class):
            raise RuntimeError(f'Weapons of {weapon_class} class were not all unique.')
        self._weapons_of_class = weapons_of_class
        self._weapon_class = weapon_class
        self._loadout_translator = loadout_translator

    def process_archetype_args(self,
                               prev_result: MainWeaponsAndSidearms,
                               preceding_args: Words,
                               following_args: Words) -> ParserResult:
        check_type(MainWeaponsAndSidearms, prev_result=prev_result)
        check_type(Words, preceding_args=preceding_args, following_args=following_args)

        weapons_of_class = tuple(weapon
                                 for weapon in self._weapons_of_class
                                 if self._loadout_translator.legend_fits_weapon(weapon))
        return ParserResult(untranslated_following_args=following_args,
                            main_weapons=weapons_of_class)
