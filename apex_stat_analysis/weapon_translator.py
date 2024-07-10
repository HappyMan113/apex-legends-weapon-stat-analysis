from typing import Generator

from apex_stat_analysis.checker import check_tuple, check_type
from apex_stat_analysis.speech.apex_terms import SWITCHING_TO_SIDEARM, WITHOUT_RELOAD
from apex_stat_analysis.speech.term import Words
from apex_stat_analysis.speech.term_translator import (SingleTermFinder,
                                                       Translator)
from apex_stat_analysis.weapon import ConcreteWeapon, WeaponArchetype


class WeaponTranslator:
    _WITH_SIDEARM_FINDER = SingleTermFinder(SWITCHING_TO_SIDEARM)
    _NO_RELOAD_FINDER = SingleTermFinder(WITHOUT_RELOAD)

    def __init__(self, weapon_archetypes: tuple[WeaponArchetype, ...]):
        check_tuple(WeaponArchetype, weapon_archetypes=weapon_archetypes)
        self._base_weapons: tuple[ConcreteWeapon, ...] = tuple(
            base_weapon
            for weapon_archetype in weapon_archetypes
            for base_weapon in weapon_archetype.get_base_weapons())
        self._archetype_translator = Translator[WeaponArchetype]({
            weapon_archetype.get_term(): weapon_archetype
            for weapon_archetype in weapon_archetypes
            if weapon_archetype.get_term() is not None})

    def get_all_weapons(self) -> tuple[ConcreteWeapon, ...]:
        return self._base_weapons

    def translate_weapon_terms(self, words: Words) -> Generator[ConcreteWeapon, None, None]:
        check_type(Words, words=words)

        one_looking_for_sidearm: ConcreteWeapon | None = None
        with_reloads = not self._NO_RELOAD_FINDER.find_term(words)

        # TODO: Consider automatically adding hop-ups at levels 3 and 4 with no attachment
        #  specification.
        for parsed_archetype in self._archetype_translator.translate_terms(words):
            archetype = parsed_archetype.get_value()
            archetype_args = parsed_archetype.get_following_words()

            best_match = archetype.get_best_match(archetype_args)
            if one_looking_for_sidearm is not None:
                best_match = one_looking_for_sidearm.combine_with_sidearm(best_match)

            # TODO: This assumes that you can only have one sidearm, but maybe we should consider
            #  Ballistic's third weapon and Rampart's "Sheila".
            elif self._WITH_SIDEARM_FINDER.find_term(archetype_args):
                one_looking_for_sidearm = best_match
                continue

            one_looking_for_sidearm = None
            if with_reloads:
                best_match = best_match.reload()

            yield best_match
