import abc

from apex_assistant.checker import check_type
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import RequiredTerm


class ApexCommand(Command, abc.ABC):
    def __init__(self, term: RequiredTerm, loadout_translator: LoadoutTranslator):
        check_type(RequiredTerm, term=term)
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        super().__init__(term)
        self._translator = loadout_translator

    def get_translator(self) -> LoadoutTranslator:
        return self._translator
