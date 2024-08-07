from enum import StrEnum

from apex_assistant.checker import check_type
from apex_assistant.speech.term import TermBase


class SuffixedArchetypeType(StrEnum):
    HOPPED_UP = 'hopped-up'
    REVVED_UP = 'revved'
    SLOW = 'slow'


class Suffix:
    def __init__(self, suffix: TermBase, suffix_type: SuffixedArchetypeType):
        check_type(TermBase, suffix=suffix)
        check_type(SuffixedArchetypeType, suffix_type=suffix_type)
        self._suffix_type = suffix_type
        self._suffix = suffix

    def get_type(self) -> SuffixedArchetypeType:
        return self._suffix_type

    def get_term(self) -> TermBase:
        return self._suffix
