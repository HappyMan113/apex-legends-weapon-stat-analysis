from enum import StrEnum
from typing import Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.speech.term import TermBase


class SuffixedArchetypeType(StrEnum):
    HOPPED_UP = 'hopped-up'
    REVVED_UP = 'revved'
    SLOW = 'slow'
    AKIMBO = 'akimbo'


class Suffix:
    def __init__(self,
                 suffixes: TermBase | Tuple[TermBase, ...],
                 suffix_types: SuffixedArchetypeType | Tuple[SuffixedArchetypeType, ...]):
        if not isinstance(suffixes, tuple):
            check_type(TermBase, suffix=suffixes)
            suffixes = (suffixes,)
        else:
            check_tuple(TermBase, suffixes=suffixes)

        if not isinstance(suffix_types, tuple):
            check_type(SuffixedArchetypeType, suffix_type=suffix_types)
            suffix_types = (suffix_types,)
        else:
            check_tuple(SuffixedArchetypeType, suffix_type=suffix_types)

        if len(suffixes) != len(suffix_types):
            raise ValueError('Must have the same number of suffix types as suffixes.')

        self._suffix_types = suffix_types
        self._suffixes = suffixes

    def __len__(self):
        return len(self._suffixes)

    def get_types(self) -> Tuple[SuffixedArchetypeType, ...]:
        return self._suffix_types

    def get_terms(self) -> Tuple[TermBase, ...]:
        return self._suffixes
