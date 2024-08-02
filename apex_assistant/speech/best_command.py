import logging
from typing import Generator, Iterable, Tuple, final

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand, Uniqueness
from apex_assistant.speech.apex_terms import BEST, LOADOUT, LOADOUTS
from apex_assistant.speech.term import Term, Words
from apex_assistant.speech.term_translator import BoolTranslator, IntTranslator, Translator
from apex_assistant.weapon import FullLoadout, Weapon
from apex_assistant.weapon_class import WeaponClass


LOGGER = logging.getLogger()


class BestLoadoutsCommand(ApexCommand):
    SINGULAR_TERM = LOADOUT
    PLURAL_TERM = LOADOUTS

    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        super().__init__(term=BEST,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)
        self._plural_translator = BoolTranslator(self.SINGULAR_TERM, self.PLURAL_TERM)
        self._number_translator = IntTranslator()
        weapon_classes: Tuple[WeaponClass, ...] = tuple(WeaponClass)
        self._weapon_class_translator = Translator[WeaponClass]({
            Term(weapon_class): weapon_class
            for weapon_class in weapon_classes
        })

    def get_weapons_of_class(self, weapon_class: WeaponClass):
        loadout_translator = self.get_translator()
        weapons = tuple(weapon
                        for weapon in loadout_translator.get_fully_kitted_weapons()
                        if weapon.get_weapon_class() is weapon_class)
        weapon_set = frozenset(weapons)
        if len(weapon_set) < len(weapons):
            raise RuntimeError(f'Weapons of {weapon_class} class were not all unique.')
        return weapon_set

    @staticmethod
    def get_loadouts(required_weapons: Iterable[Weapon],
                     optional_weapons: Iterable[Weapon]) -> Tuple[FullLoadout, ...]:
        required_weapons = set(required_weapons)
        optional_weapons = set(optional_weapons)
        loadouts = tuple(
            FullLoadout(main_loadout, sidearm)
            for main_weapon in required_weapons
            for main_loadout in main_weapon.get_main_loadout_variants()
            for sidearm in optional_weapons
        ) + tuple(FullLoadout(main_loadout, sidearm)
                  for main_weapon in optional_weapons - required_weapons
                  for main_loadout in main_weapon.get_main_loadout_variants()
                  for sidearm in required_weapons
                  )
        return loadouts

    @final
    def _execute(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)

        plural = self._plural_translator.translate_term(arguments)
        if plural is None:
            # Didn't say "loadouts". Must've misheard I guess.
            return ''

        number_term = self._number_translator.translate_at_start(arguments)

        if number_term is None:
            number = 3 if plural else 1
        elif number_term.get_value() <= 0:
            return 'Number must be positive.'
        else:
            number = number_term.get_value()

        optional_weapons = self.get_translator().get_fully_kitted_weapons()
        required_weapons = tuple(self.get_weapons(arguments))
        comparator = self.get_comparator()
        if len(required_weapons) == 0:
            loadouts = comparator.get_best_loadouts(weapons=optional_weapons,
                                                    max_num_loadouts=number)
        else:
            loadouts = self.get_loadouts(required_weapons, optional_weapons)

        comparison_result = comparator.compare_loadouts(loadouts).limit_to_best_num(number)
        loadouts = tuple(comparison_result.get_sorted_loadouts().keys())
        uniqueness_func = (self._get_uniqueness if len(required_weapons) == 1 else
                           self._get_uniqueness_full)
        uniqueness = uniqueness_func(loadouts)

        number = len(loadouts)
        term = self.SINGULAR_TERM if number == 1 else self.PLURAL_TERM
        best_weapons_str = (f'{number} best' if number != 1 else 'Best') + f' {term}'
        LOGGER.info(f'{best_weapons_str}: {comparison_result}')
        prefix = f'Best {term}, '

        return prefix + ', '.join(self._make_audible(loadout, uniqueness) for loadout in loadouts)

    def get_weapons(self, arguments: Words) -> Generator[Weapon, None, None]:
        class_translation = self._weapon_class_translator.translate_terms(arguments)
        loadout_translator = self.get_translator()

        for weapon in loadout_translator.translate_weapons(class_translation.get_preamble()):
            yield weapon

        for class_term in class_translation:
            for weapon in self.get_weapons_of_class(class_term.get_value()):
                yield weapon

            for weapon in loadout_translator.translate_weapons(class_term.get_following_words()):
                yield weapon

    @staticmethod
    def _get_uniqueness_full(loadouts: Tuple[FullLoadout, ...]) -> Uniqueness:
        check_tuple(FullLoadout, loadouts=loadouts)

        main_attachments_all_same = ApexCommand._attachments_all_same(loadout.get_main_weapon()
                                                                      for loadout in loadouts)
        sidearm_attachments_all_same = ApexCommand._attachments_all_same(loadout.get_sidearm()
                                                                         for loadout in loadouts)
        if main_attachments_all_same and sidearm_attachments_all_same:
            return Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
        return Uniqueness.SAY_FULL_LOADOUT_NAMES
