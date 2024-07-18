import logging
from typing import Generator, Tuple

from apex_assistant.checker import check_bool, check_tuple, check_type
from apex_assistant.speech.apex_config import ApexConfig
from apex_assistant.speech.apex_terms import (SWITCHING_TO_SIDEARM,
                                              THE_SAME_MAIN_WEAPON,
                                              WITHOUT_RELOAD)
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import Translator
from apex_assistant.weapon import ConcreteWeapon, WeaponArchetypes, Loadout


class WeaponTranslator:
    _LOGGER = logging.getLogger()

    def __init__(self, weapon_archetypes: tuple[WeaponArchetypes, ...], apex_config: ApexConfig):
        check_tuple(WeaponArchetypes, weapon_archetypes=weapon_archetypes)
        check_type(ApexConfig, apex_config=apex_config)

        self._base_weapons: Tuple[ConcreteWeapon, ...] = tuple(
            base_weapon
            for weapon_archetype in weapon_archetypes
            for base_weapon in weapon_archetype.get_base_weapons())
        self._fully_kitted_weapons: Tuple[ConcreteWeapon, ...] = tuple(
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

    def _to_default_loadout(self, weapon: ConcreteWeapon) -> Loadout:
        loadout = weapon
        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            loadout = loadout.add_sidearm(default_sidearm)
        if self.get_reload_by_default():
            loadout = loadout.reload()
        return loadout

    def get_fully_kitted_loadouts(self) -> Tuple[Loadout, ...]:
        return tuple(map(self._to_default_loadout, self._fully_kitted_weapons))

    def translate_weapon_terms(self, words: Words) -> Generator[Loadout, None, None]:
        check_type(Words, words=words)

        if WITHOUT_RELOAD in words:
            with_reloads = False
        else:
            with_reloads = self.get_reload_by_default()

        # The most recent concrete weapon parsed.
        main_weapon: ConcreteWeapon | None = None
        # Whether that most recent main weapon is intended to have a sidearm.
        looking_for_sidearm: bool = False

        # TODO: Consider automatically adding hop-ups at levels 3 and 4 with no attachment
        #  specification.
        for parsed_archetype in self._archetype_translator.translate_terms(words):
            archetype = parsed_archetype.get_value()
            archetype_args = parsed_archetype.get_following_words()

            if archetype is None:
                if looking_for_sidearm:
                    self._LOGGER.warning(
                        'Cannot have sidearm of a main weapon, that makes no sense. Ignoring '
                        'sidearm term.')

                if main_weapon is None:
                    self._LOGGER.warning(
                        f'No previous main weapon specified for "{parsed_archetype.get_term()}" to '
                        'make sense. Term will be ignored.')
                    continue
                if self.is_asking_for_sidearm(archetype_args):
                    looking_for_sidearm = True
                    continue

                weapon = self.add_default_sidearm(main_weapon)
            elif looking_for_sidearm:
                # TODO: This assumes that you can only have one sidearm, but maybe we should
                #  consider Ballistic's third weapon and Rampart's "Sheila".
                looking_for_sidearm = False
                sidearm = archetype.get_best_match(archetype_args)
                weapon = main_weapon.add_sidearm(sidearm)
            elif self.is_asking_for_sidearm(archetype_args):
                main_weapon = archetype.get_best_match(archetype_args)
                looking_for_sidearm = True
                continue
            else:
                main_weapon = archetype.get_best_match(archetype_args)
                weapon = self.add_default_sidearm(main_weapon)

            if with_reloads:
                weapon = weapon.reload()

            yield weapon

    @staticmethod
    def is_asking_for_sidearm(archetype_args: Words) -> bool:
        return SWITCHING_TO_SIDEARM in archetype_args

    def add_default_sidearm(self, main_weapon: ConcreteWeapon) -> Loadout:
        check_type(ConcreteWeapon, main_weapon=main_weapon)

        default_sidearm = self.get_default_sidearm()
        if default_sidearm is not None:
            weapon = main_weapon.add_sidearm(default_sidearm)
        else:
            weapon = main_weapon
        return weapon
