import abc

from apex_assistant.checker import check_type
from apex_assistant.loadout_comparer import LoadoutComparer
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import RequiredTerm


class ApexCommand(Command, abc.ABC):
    def __init__(self,
                 term: RequiredTerm,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparer: LoadoutComparer):
        check_type(RequiredTerm, term=term)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        check_type(LoadoutComparer, loadout_comparer=loadout_comparer)
        super().__init__(term)
        self._translator = loadout_translator
        self._comparer = loadout_comparer

    def get_translator(self):
        return self._translator

    def get_comparer(self):
        return self._comparer
