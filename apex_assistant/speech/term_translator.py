import logging
from typing import Generic, Mapping, Optional, Tuple, TypeVar, final

from apex_assistant.checker import check_mapping, check_type
from apex_assistant.speech.apex_terms import NUMBER_TERMS
from apex_assistant.speech.term import IntTerm, RequiredTerm, TermBase, Word, Words
from apex_assistant.speech.translations import (FindResult,
                                                FoundAtStartTerm,
                                                TranslatedTerm,
                                                TranslatedTermBuilder,
                                                Translation)


_LOGGER = logging.getLogger()
T = TypeVar('T')


class Translator(Generic[T]):
    def __init__(self, terms: Mapping[TermBase, T] | None = None):
        self._max_term_variation_len = 0
        self._terms_by_first_words: dict[Word, dict[RequiredTerm, T]] = {}
        self._values: set[T] = set()
        self._opt_term_and_val: Optional[Tuple[RequiredTerm, T]] = None
        if terms is not None:
            self.add_terms(terms)

    @final
    def add_term(self, term: TermBase, val: T):
        self.add_terms({term: val})

    @final
    def add_terms(self, terms: Mapping[TermBase, T]):
        check_mapping(TermBase, terms=terms)
        self._max_term_variation_len = max(self._max_term_variation_len,
                                           max((term.get_max_variation_len() for term in terms),
                                               default=0))
        for term, val in terms.items():
            term, is_opt = term.unwrap()
            self._values.add(val)
            if is_opt:
                if self._opt_term_and_val is not None:
                    raise ValueError('Cannot have more than one optional term in a translator.')
                self._opt_term_and_val = term, val

            for first_word in term.get_possible_first_words():
                assert isinstance(first_word, Word)
                terms_elt = self._terms_by_first_words.get(first_word, {})
                if len(terms_elt) == 0:
                    self._terms_by_first_words[first_word] = terms_elt
                terms_elt[term] = val

    @final
    def values(self):
        return self._values

    @final
    def __contains__(self, term: RequiredTerm):
        first_first_word = next(term.get_possible_first_words())
        return term in self._terms_by_first_words.get(first_first_word, {})

    @final
    def translate_terms(self, words: Words) -> Translation[T]:
        check_type(Words, words=words)
        if self._max_term_variation_len == 0:
            return self._get_translation(words, tuple())

        prev_stop_idx: int = 0
        hit_idx: int = 0
        builder: TranslatedTermBuilder[T] | None = None
        translated_terms: list[TranslatedTerm[T]] = []
        while hit_idx < len(words):
            res = self._translate_term(words[hit_idx:hit_idx + self._max_term_variation_len])
            if res is not None:
                preceding_words = words[prev_stop_idx:hit_idx]
                if builder is not None:
                    translated_terms.append(builder.build(preceding_words, hit_idx))
                term, val, words_inc = res
                builder = TranslatedTermBuilder(term=term,
                                                value=val,
                                                word_start_idx=hit_idx,
                                                preceding_words=preceding_words)
                prev_stop_idx = hit_idx + words_inc
            else:
                words_inc = 1

            hit_idx += words_inc

        if builder is not None:
            translated_terms.append(builder.build(following_words=words[prev_stop_idx:],
                                                  word_stop_idx=prev_stop_idx))
        return self._get_translation(words, tuple(translated_terms))

    def translate_at_start(self, words: Words) -> Tuple[Optional[T], Words]:
        translation = self.translate_terms(words)
        first_term = translation.get_first_term()
        untranslated_words = translation.get_untranslated_words()
        if first_term is None or first_term.get_word_start_idx() != 0:
            return None, untranslated_words
        return first_term.get_value(), untranslated_words

    @final
    def _get_translation(self,
                         original_words: Words,
                         translated_terms: Tuple[TranslatedTerm[T]]) -> Translation[T]:
        if len(translated_terms) == 0 and self._opt_term_and_val is not None:
            term, val = self._opt_term_and_val
            start_stop_idx = 0
            default_term = TranslatedTerm(term=term,
                                          value=val,
                                          word_start_idx=start_stop_idx,
                                          word_stop_idx=start_stop_idx,
                                          preceding_words=original_words[:start_stop_idx],
                                          following_words=original_words[start_stop_idx:])
            translated_terms = (default_term,)

        return Translation(original_words=original_words, translated_terms=translated_terms)

    def _translate_term(self, words: Words) -> Tuple[RequiredTerm, T, int] | None:
        check_type(Words, words=words)
        if len(words) == 0:
            raise ValueError('words cannot be empty.')

        first_word = words.get_first_word()
        if first_word is None:
            return None

        term_to_val_dict = self._terms_by_first_words.get(first_word, None)
        if term_to_val_dict is None:
            return None

        for num_words in range(min(len(words), self._max_term_variation_len), 0, -1):
            term_to_test = words[:num_words]
            term_and_val: Optional[Tuple[RequiredTerm, T]] = self._translate_term_directly(
                term_to_test=term_to_test,
                term_to_val_dict=term_to_val_dict)
            if term_and_val is not None:
                term, val = term_and_val
                return term, val, num_words

        return None

    @staticmethod
    @final
    def _translate_term_directly(term_to_test: Words, term_to_val_dict: dict[RequiredTerm, T]) -> \
            tuple[RequiredTerm, T] | None:
        assert isinstance(term_to_test, Words)
        for term, val in term_to_val_dict.items():
            if term.has_variation(term_to_test):
                return term, val
        return None


class SingleTermFinder:
    def __init__(self, term: TermBase):
        check_type(TermBase, term=term)
        req_term, is_opt = term.unwrap()

        self._translator = Translator({req_term: req_term})
        self._is_opt = is_opt
        self._req_term = req_term
        self._term = term

    def find_all(self, said_words: Words) -> FindResult:
        check_type(Words, said_words=said_words)
        translation = self._translator.translate_terms(said_words)
        return FindResult(found_any=(len(translation) > 0 or self._is_opt),
                          found_terms=translation.terms(),
                          untranslated_words=translation.get_untranslated_words())

    def find_at_start(self, said_words: Words) -> Optional[FoundAtStartTerm]:
        find_result = self.find_all(said_words[:self._req_term.get_max_variation_len()])
        if len(find_result) == 0:
            return None
        first_result = find_result[0]
        if first_result.get_word_start_idx() != 0:
            return None

        num_words = len(first_result)
        following_words = said_words[num_words:]
        return FoundAtStartTerm(length=num_words, following_words=following_words)

    def get_term(self) -> TermBase:
        return self._term


class BoolTranslator:
    def __init__(self, false_term: RequiredTerm, true_term: RequiredTerm):
        self._translator = Translator[bool]({true_term: True, false_term: False})

    def translate_term(self, words: Words, default: Optional[bool] = None) -> \
            Tuple[Optional[bool], Words]:
        translation = self._translator.translate_terms(words)
        if len(translation) > 1:
            _LOGGER.warning('More than one bool value found. Only the last one will be used.')
        return translation.get_latest_value(default), translation.get_untranslated_words()


class IntTranslator(Translator[int]):
    _NEGATIVE_FINDER = SingleTermFinder(IntTerm.NEGATIVE)

    def __init__(self):
        super().__init__({
            ((IntTerm.NEGATIVE + number_term) if negative else number_term): (
                    (-1 if negative else 1) * idx)
            for idx, number_term in enumerate(NUMBER_TERMS)
            for negative in (False, True)})

    def _translate_term(self, words: Words) -> Optional[Tuple[RequiredTerm, int, int]]:
        result = super()._translate_term(words)
        if result is not None:
            return result

        negative_term = IntTranslator._NEGATIVE_FINDER.find_at_start(words)
        if negative_term is None:
            int_val = IntTranslator._translate_abs_int(words)
            if int_val is None:
                return None

            return IntTerm(int_val), int_val, 1

        abs_int = IntTranslator._translate_abs_int(negative_term.get_following_words())
        if abs_int is None:
            return None

        int_val = -abs_int
        num_words = len(negative_term) + 1
        return IntTerm(int_val), int_val, num_words

    @staticmethod
    def _translate_abs_int(words: Words) -> Optional[int]:
        try:
            int_value = int(words.get_first_word().get_stripped_word())
        except ValueError:
            return None

        return int_value
