import logging
from typing import Generator, Generic, Mapping, Tuple, TypeVar

from apex_stat_analysis.checker import check_type
from apex_stat_analysis.speech.term import RequiredTerm, Word, Words


logger = logging.getLogger()
T = TypeVar('T')

class ParsedAndFollower(Generic[T]):
    def __init__(self, term: RequiredTerm, value: T, following_words: Words):
        check_type(RequiredTerm, term=term)
        check_type(Words, following_words=following_words)
        self.term = term
        self.value = value
        self.following_words = following_words

    def get_term(self) -> RequiredTerm:
        return self.term

    def get_parsed(self) -> T:
        return self.value

    def get_following_words(self) -> Words:
        return self.following_words


class ParsedAndFollowerBuilder(Generic[T]):
    def __init__(self, term: RequiredTerm, value: T):
        check_type(RequiredTerm, term=term)
        self.term = term
        self.value = value
        self.follower_words: list[Word] = []

    def add_word(self, word: Word):
        check_type(Word, word=word)
        self.follower_words.append(word)

    def build(self) -> ParsedAndFollower[T]:
        return ParsedAndFollower(term=self.term,
                                 value=self.value,
                                 following_words=Words(self.follower_words))


class Translator(Generic[T]):
    def __init__(self, terms: Mapping[RequiredTerm, T] | None = None):
        self._term_word_lim = 0
        self._terms_by_first_words: dict[Word, dict[RequiredTerm, T]] = {}
        self._values = set()
        if terms is not None:
            self.add_terms(terms)

    def add_term(self, term: RequiredTerm, val: T):
        self.add_terms({term: val})

    def add_terms(self, terms: Mapping[RequiredTerm, T]):
        self._term_word_lim = max(self._term_word_lim,
                                  max((term.get_max_variation_len() for term in terms), default=0))
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

    def translate_terms(self, words: Words) -> Generator[ParsedAndFollower[T], None, None]:
        check_type(Words, words=words)
        if self._term_word_lim == 0:
            return

        idx = 0
        builder: ParsedAndFollowerBuilder[T] | None = None
        while idx < len(words):
            res = self._translate_term(words[idx:idx + self._term_word_lim])
            if res is not None:
                if builder is not None:
                    yield builder.build()
                term, val, words_inc = res
                builder = ParsedAndFollowerBuilder(term=term, value=val)
            else:
                if builder is not None:
                    word = words[idx]
                    builder.add_word(word)
                words_inc = 1

            idx += words_inc

        if builder is not None:
            yield builder.build()

    def _translate_term(self, words: Words) -> tuple[RequiredTerm, T, int] | None:
        check_type(Words, words=words)
        if len(words) == 0:
            return None

        first_word = words.get_first_word()
        if first_word is None:
            return None

        term_to_val_dict = self._terms_by_first_words.get(first_word, None)
        if term_to_val_dict is None:
            return None

        for num_words in range(self._term_word_lim, 0, -1):
            term_to_test = words[:num_words]
            term_and_val: tuple[RequiredTerm, T] | None = self._translate_term2(
                term_to_test=term_to_test,
                term_to_val_dict=term_to_val_dict)
            if term_and_val is not None:
                term, val = term_and_val
                return term, val, num_words

        return None

    @staticmethod
    def _translate_term2(term_to_test: Words, term_to_val_dict: dict[RequiredTerm, T]) -> \
            tuple[RequiredTerm, T] | None:
        assert isinstance(term_to_test, Words)
        for term, val in term_to_val_dict.items():
            if term.has_variation(term_to_test):
                return term, val
        return None


class TermLookerUpper:
    def __init__(self, terms: Tuple[RequiredTerm, ...]):
        check_type(tuple, terms=terms)
        self._translator = Translator[RequiredTerm]({term: term for term in terms})

    def look_up_term(self, words: Words) -> tuple[RequiredTerm, ...]:
        return tuple(parsed.get_parsed() for parsed in self._translator.translate_terms(words))


class SingleTermFinder:
    def __init__(self, term: RequiredTerm):
        self._looker_upper = Translator[RequiredTerm]({term: term})

    def find_term(self, words: Words) -> bool:
        return next(self._looker_upper.translate_terms(words), None) is not None
