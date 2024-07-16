from typing import Generator

from apex_assistant.checker import check_bool, check_tuple, check_type
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.apex_terms import SWITCHING_TO_SIDEARM, WITHOUT_RELOAD
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import SingleTermFinder, Translator
from apex_assistant.weapon import ConcreteWeapon, WeaponArchetype, WeaponBase


class WeaponTranslator:
    _WITH_SIDEARM_FINDER = SingleTermFinder(SWITCHING_TO_SIDEARM)
    _NO_RELOAD_FINDER = SingleTermFinder(WITHOUT_RELOAD)

    def __init__(self, weapon_archetypes: tuple[WeaponArchetype, ...], apex_config: ApexConfig):
        check_tuple(WeaponArchetype, weapon_archetypes=weapon_archetypes)
        self._base_weapons: tuple[ConcreteWeapon, ...] = tuple(
            base_weapon
            for weapon_archetype in weapon_archetypes
            for base_weapon in weapon_archetype.get_base_weapons())
        self._archetype_translator = Translator[WeaponArchetype]({
            weapon_archetype.get_term(): weapon_archetype
            for weapon_archetype in weapon_archetypes
            if weapon_archetype.get_term() is not None})
        default_sidearm_name_prop = apex_config.get_default_sidearm_name()
        self._default_sidearm_name_prop = default_sidearm_name_prop
        self._default_sidearm_prop = default_sidearm_name_prop.map(
            lambda name: (self.translate_concrete_weapon(Words(name)) if name is not None else
                          None))
        self._reload_by_default_prop = apex_config.get_reload_default()

    def set_reload_by_default(self, reload_by_default: bool) -> bool:
        check_bool(reload_by_default=reload_by_default)
        return self._reload_by_default_prop.set_value(reload_by_default)

    def set_default_sidearm(self, default_sidearm: ConcreteWeapon | None) -> ConcreteWeapon | None:
        check_type(ConcreteWeapon, optional=True, default_sidearm=default_sidearm)
        old_value = self._default_sidearm_prop.get_value()
        self._default_sidearm_name_prop.set_value(default_sidearm.get_name())
        return old_value

    def translate_concrete_weapon(self, words: Words) -> ConcreteWeapon | None:
        check_type(Words, words=words)
        return next((parsed_archetype.get_value().get_best_match(words)
                     for parsed_archetype in self._archetype_translator.translate_terms(words)),
                    None)

    def get_default_sidearm(self) -> ConcreteWeapon | None:
        return self._default_sidearm_prop.get_value()

    def get_reload_by_default(self) -> bool:
        return self._reload_by_default_prop.get_value()

    def get_base_weapons(self) -> tuple[ConcreteWeapon, ...]:
        return self._base_weapons

    def get_default_weapons(self) -> tuple[WeaponBase, ...]:
        weapons = self._base_weapons
        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            weapons = tuple(weapon.combine_with_sidearm(default_sidearm) for weapon in weapons)
        if self.get_reload_by_default():
            weapons = tuple(weapon.reload() for weapon in weapons)
        return weapons

    def translate_weapon_terms(self, words: Words) -> Generator[WeaponBase, None, None]:
        check_type(Words, words=words)

        one_looking_for_sidearm: ConcreteWeapon | None = None
        if self._NO_RELOAD_FINDER.find_term(words):
            with_reloads = False
        else:
            with_reloads = self.get_reload_by_default()
        default_sidearm: ConcreteWeapon | None = self.get_default_sidearm()

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

            elif default_sidearm is not None:
                best_match = best_match.combine_with_sidearm(default_sidearm)

            one_looking_for_sidearm = None
            if with_reloads:
                best_match = best_match.reload()

            yield best_match
