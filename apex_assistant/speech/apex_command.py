import abc

from apex_assistant.checker import check_type
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import RequiredTerm
from apex_assistant.weapon_comparer import WeaponComparer
from apex_assistant.weapon_translator import WeaponTranslator


class ApexCommand(Command, abc.ABC):
    def __init__(self,
                 term: RequiredTerm,
                 weapon_translator: WeaponTranslator,
                 weapon_comparer: WeaponComparer):
        check_type(RequiredTerm, term=term)
        check_type(WeaponTranslator, weapon_translator=weapon_translator)
        check_type(WeaponComparer, weapon_comparer=weapon_comparer)
        super().__init__(term)
        self._translator = weapon_translator
        self._comparer = weapon_comparer

    def get_translator(self):
        return self._translator

    def get_comparer(self):
        return self._comparer
