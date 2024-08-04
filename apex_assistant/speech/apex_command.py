import abc
from enum import IntEnum

from apex_assistant.checker import check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import RequiredTerm


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

    def get_translator(self) -> LoadoutTranslator:
        return self._translator

    def get_comparator(self) -> LoadoutComparator:
        return self._comparator
