import abc
from enum import IntEnum
from typing import Dict, Iterable, Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_terms import MAIN, SIDEARM, SINGLE_SHOT
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import RequiredTerm
from apex_assistant.weapon import FullLoadout, Weapon, WeaponArchetype


class Uniqueness(IntEnum):
    SAY_MAIN_ARCHETYPE_NAMES = 0
    SAY_SIDEARM_ARCHETYPE_NAMES = 1
    SAY_FULL_LOADOUT_ARCHETYPE_NAMES = 2
    SAY_MAIN_LOADOUT_NAMES = 3
    SAY_SIDEARM_WEAPON_NAMES = 4
    SAY_FULL_LOADOUT_NAMES = 5


class ApexCommand(Command, abc.ABC):
    def __init__(self,
                 term: RequiredTerm,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        check_type(RequiredTerm, term=term)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        check_type(LoadoutComparator, loadout_comparator=loadout_comparator)
        super().__init__(term)
        self._translator = loadout_translator
        self._comparator = loadout_comparator

    def get_translator(self):
        return self._translator

    def get_comparator(self):
        return self._comparator

    @staticmethod
    def _attachments_all_same(weapons: Iterable[Weapon]) -> bool:
        archetype_to_loadout_dict: Dict[WeaponArchetype, Weapon] = {}
        for weapon in weapons:
            archetype = weapon.get_archetype()
            if (archetype in archetype_to_loadout_dict and
                    archetype_to_loadout_dict[archetype] != weapon):
                return False
            archetype_to_loadout_dict[archetype] = weapon
        return True

    @staticmethod
    def _get_uniqueness(loadouts: Tuple[FullLoadout, ...]) -> Uniqueness:
        check_tuple(FullLoadout, loadouts=loadouts)

        main_weapon_archetypes = set(loadout.get_archetype() for loadout in loadouts)
        main_weapon_archetypes_unique = len(main_weapon_archetypes) == len(loadouts)
        sidearms = set(weapon.get_sidearm() for weapon in loadouts)

        main_attachments_all_same = ApexCommand._attachments_all_same(loadout.get_main_weapon()
                                                                      for loadout in loadouts)

        if main_weapon_archetypes_unique:
            return Uniqueness.SAY_MAIN_ARCHETYPE_NAMES

        sidearms_all_same = len(sidearms) == 1
        if sidearms_all_same:
            if main_attachments_all_same:
                return Uniqueness.SAY_MAIN_ARCHETYPE_NAMES
            else:
                return Uniqueness.SAY_MAIN_LOADOUT_NAMES

        main_loadouts = set(weapon.get_main_loadout() for weapon in loadouts)
        main_loadouts_all_same = len(main_loadouts) == 1
        sidearm_archetypes = set(
            loadout.get_sidearm().get_archetype() if loadout.get_sidearm() is not None else None
            for loadout in loadouts)
        sidearm_archetypes_unique = len(sidearm_archetypes) == len(loadouts)

        if main_loadouts_all_same and sidearm_archetypes_unique:
            return Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES

        sidearms_unique = len(sidearms) == len(loadouts)
        sidearm_attachments_all_same = ApexCommand._attachments_all_same(loadout.get_sidearm()
                                                                         for loadout in loadouts)

        if main_loadouts_all_same and sidearms_unique:
            if sidearm_attachments_all_same:
                return Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES
            else:
                return Uniqueness.SAY_SIDEARM_WEAPON_NAMES

        if main_attachments_all_same and sidearm_attachments_all_same:
            return Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES
        return Uniqueness.SAY_FULL_LOADOUT_NAMES

    @staticmethod
    def _make_audible(loadout: FullLoadout,
                      uniqueness: Uniqueness = Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES) -> str:
        check_type(FullLoadout, loadout=loadout)
        check_type(Uniqueness, uniqueness=uniqueness)

        if uniqueness is Uniqueness.SAY_MAIN_ARCHETYPE_NAMES:
            main_loadout = loadout.get_main_loadout()
            loadout_term = MAIN.append(main_loadout.get_archetype().get_term())
            if main_loadout.is_single_shot():
                loadout_term = loadout_term.append(SINGLE_SHOT)
        elif uniqueness is Uniqueness.SAY_SIDEARM_ARCHETYPE_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_archetype().get_term()
        elif uniqueness is Uniqueness.SAY_MAIN_LOADOUT_NAMES:
            loadout_term = MAIN.append(loadout.get_main_loadout().get_term())
        elif uniqueness is Uniqueness.SAY_FULL_LOADOUT_ARCHETYPE_NAMES:
            main_loadout = loadout.get_main_loadout()
            main_term = main_loadout.get_archetype().get_term()
            if main_loadout.is_single_shot():
                main_term = main_term.append(SINGLE_SHOT)
            sidearm_term = loadout.get_sidearm().get_archetype().get_term()
            loadout_term = main_term.append(SIDEARM, sidearm_term)
        elif uniqueness is Uniqueness.SAY_SIDEARM_WEAPON_NAMES:
            loadout_term = SIDEARM + loadout.get_sidearm().get_term()
        else:
            loadout_term = loadout.get_term()

        return loadout_term.to_audible_str()
