import logging
from typing import Generic, Mapping, Optional, Tuple, TypeVar

from apex_assistant.checker import check_mapping, check_type
from apex_assistant.speech.term import RequiredTerm, TermBase, Word, Words
from apex_assistant.speech.translations import (FindResult,
                                                FoundTerm,
                                                TranslatedTerm,
                                                TranslatedTermBuilder,
                                                Translation)


_LOGGER = logging.getLogger()
T = TypeVar('T')


class Translator(Generic[T]):
    def __init__(self, terms: Mapping[RequiredTerm, T] | None = None):
        self._max_term_variation_len = 0
        self._terms_by_first_words: dict[Word, dict[RequiredTerm, T]] = {}
        self._values = set()
        if terms is not None:
            self.add_terms(terms)

    def add_term(self, term: RequiredTerm, val: T):
        self.add_terms({term: val})

    def add_terms(self, terms: Mapping[RequiredTerm, T]):
        check_mapping(RequiredTerm, terms=terms)
        self._max_term_variation_len = max(self._max_term_variation_len,
                                           max((term.get_max_variation_len() for term in terms),
                                               default=0))
        for term, val in terms.items():
            self._values.add(val)

            for first_word in term.get_possible_first_words():
                assert isinstance(first_word, Word)
                terms_elt = self._terms_by_first_words.get(first_word, {})
                if len(terms_elt) == 0:
                    self._terms_by_first_words[first_word] = terms_elt
                terms_elt[term] = val

    def values(self):
        return self._values

    def __contains__(self, term: RequiredTerm):
        first_first_word = next(term.get_possible_first_words())
        return term in self._terms_by_first_words.get(first_first_word, {})

    def translate_terms(self, words: Words) -> Translation[T]:
        check_type(Words, words=words)
        if self._max_term_variation_len == 0:
            return Translation(words, tuple())

        prev_hit_idx: int = -1
        hit_idx: int = 0
        builder: TranslatedTermBuilder[T] | None = None
        translated_terms: list[TranslatedTerm[T]] = []
        while hit_idx < len(words):
            res = self._translate_term(words[hit_idx:hit_idx + self._max_term_variation_len])
            if res is not None:
                if builder is not None:
                    translated_terms.append(builder.build())
                term, val, words_inc = res
                builder = TranslatedTermBuilder(term=term,
                                                value=val,
                                                preceding_words=words[prev_hit_idx + 1:hit_idx])
                prev_hit_idx = hit_idx
            else:
                if builder is not None:
                    word = words[hit_idx]
                    builder.add_word(word)
                words_inc = 1

            hit_idx += words_inc

        if builder is not None:
            translated_terms.append(builder.build())
        return Translation(words, tuple(translated_terms))

    def _translate_term(self, words: Words) -> Tuple[RequiredTerm, T, int] | None:
        check_type(Words, words=words)
        if len(words) == 0:
            return None

        first_word = words.get_first_word()
        if first_word is None:
            return None

        term_to_val_dict = self._terms_by_first_words.get(first_word, None)
        if term_to_val_dict is None:
            return None

        for num_words in range(self._max_term_variation_len, 0, -1):
            term_to_test = words[:num_words]
            term_and_val: Tuple[RequiredTerm, T] | None = self._translate_term_directly(
                term_to_test=term_to_test,
                term_to_val_dict=term_to_val_dict)
            if term_and_val is not None:
                term, val = term_and_val
                return term, val, num_words

        return None

    @staticmethod
    def _translate_term_directly(term_to_test: Words, term_to_val_dict: dict[RequiredTerm, T]) -> \
            tuple[RequiredTerm, T] | None:
        assert isinstance(term_to_test, Words)
        for term, val in term_to_val_dict.items():
            if term.has_variation(term_to_test):
                return term, val
        return None


class BoolTranslator:
    def __init__(self, true_term: RequiredTerm, false_term: RequiredTerm):
        self._translator = Translator({true_term: True,
                                       false_term: False})

    def translate(self, words: Words, default_value: bool) -> FindResult:
        translation = self._translator.translate_terms(words)
        if len(translation) > 1:
            _LOGGER.warning('More than one value found. Only the last one will be used.')
            value = translation.get_latest_value()
        else:
            value = next(iter(translation), default_value)
        return FindResult(found_any=value,
                          found_terms=translation.terms(),
                          untranslated_words=translation.get_untranslated_words())


class SingleTermFinder:
    def __init__(self, term: TermBase):
        check_type(TermBase, term=term)
        req_term, is_opt = term.unwrap()

        self._translator = Translator({req_term: None})
        self._is_opt = is_opt

    def find_all(self, said_words: Words) -> FindResult:
        check_type(Words, said_words=said_words)
        translation = self._translator.translate_terms(said_words)
        return FindResult(found_any=(len(translation) > 0 or self._is_opt),
                          found_terms=translation.terms(),
                          untranslated_words=translation.get_untranslated_words())
