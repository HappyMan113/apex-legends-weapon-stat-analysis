import logging
import re
import typing

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.terms import (ApexTerms,
                                             ApexTermBase,
                                             ConcreteApexTerm,
                                             extract_terms)
from apex_stat_analysis.weapon import WeaponArchetype, WeaponBase
from apex_stat_analysis.weapon_database import ApexDatabase


LOGGER = logging.getLogger()

class CompareCommand(Command):
    def __init__(self):
        super().__init__(ApexTerms.COMPARE)
        self._term_to_weapon_dict: dict[tuple[ConcreteApexTerm], tuple[WeaponBase]] = {
            extract_terms(weapon.get_term()): (weapon,)
            for weapon in ApexDatabase.get_instance().get_base_weapons()
            if weapon.get_term() is not None
        }
        self._max_term_len = max(map(len, self._term_to_weapon_dict.keys()))

    def _execute(self, arguments: typing.Iterable[ApexTermBase]) -> str:
        arguments = extract_terms(arguments)
        weapons, arguments = self.get_base_weapons(arguments)
        if len(weapons) < 2:
            return 'Must specify two or more weapons to compare.'
        archetypes = [weapon.get_archetype() for weapon in weapons]
        if len(archetypes) != len(set(archetypes)):
            return 'Cannot compare weapons of same type! That\'s pointless!'
        with_reloads = (ApexTerms.NO_RELOAD_TERM not in arguments)
        if with_reloads:
            weapons = [weapon.reload() for weapon in weapons]

        LOGGER.debug(f'Comparing: {weapons}')
        comparison_result = ApexDatabase.get_instance().compare_weapons(weapons)
        best_weapon, score = comparison_result.get_best_weapon()
        LOGGER.debug(f'Best: {best_weapon}')
        audible_name = self.make_audible(best_weapon.get_archetype())

        if len(weapons) == 2:
            second_best_weapon, second_best_score = comparison_result.get_nth_best_weapon(2)
            second_audible_name = self.make_audible(second_best_weapon.get_archetype())
            better_percentage = round(((score / second_best_score) - 1) * 100)
            return (f'{audible_name} is {better_percentage:.0f} percent better than'
                    f' {second_audible_name}.')

        return f'{audible_name} is best.'

    @staticmethod
    def make_audible(weapon: WeaponArchetype | WeaponBase):
        audible_name = re.sub('[()]', '', weapon.get_name())
        return audible_name

    def get_base_weapons(self, weapon_terms: tuple[ConcreteApexTerm]) -> \
            tuple[tuple[WeaponBase], tuple[ConcreteApexTerm]]:
        weapons: list[WeaponBase] = []
        skipped_terms: list[ConcreteApexTerm] = []

        cur_term_idx = 0
        while cur_term_idx < len(weapon_terms):
            result = self.translate(weapon_terms[cur_term_idx:])
            if result is not None:
                weapons1, num_terms = result
                weapons.extend(weapons1)
                cur_term_idx += num_terms
            else:
                skipped_terms.append(weapon_terms[cur_term_idx])
                cur_term_idx += 1

        return tuple(weapons), tuple(skipped_terms)

    def translate(self, weapon_terms: tuple[ConcreteApexTerm]) -> \
            tuple[tuple[WeaponBase], int] | None:
        max_num_terms = min(self._max_term_len, len(weapon_terms))
        for num_terms in range(max_num_terms, 0, -1):
            terms_slice = weapon_terms[:num_terms]
            weapons = self._term_to_weapon_dict.get(terms_slice, None)
            if weapons is not None:
                return weapons, num_terms
        return None
