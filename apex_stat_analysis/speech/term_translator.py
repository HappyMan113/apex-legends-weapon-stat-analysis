import logging
import typing
from typing import Generator, Generic, TypeVar

from apex_stat_analysis.speech.terms import ApexTermBase, Words, Word


logger = logging.getLogger()
T = TypeVar('T')

class ParsedAndFollower(Generic[T]):
    def __init__(self,
                 term: ApexTermBase,
                 value: T,
                 preceding_words: Words | None,
                 following_words: Words | None):
        self.term = term
        self.value = value
        self.preceding_words = preceding_words
        self.following_words = following_words

    def get_term(self) -> ApexTermBase:
        return self.term

    def get_parsed(self) -> T:
        return self.value

    def get_preceding_words(self) -> Words | None:
        return self.preceding_words

    def get_following_words(self) -> Words | None:
        return self.following_words


class ParsedAndFollowerBuilder(Generic[T]):
    def __init__(self, term: ApexTermBase, value: T, preceding_words: Words | None):
        assert isinstance(term, ApexTermBase)
        assert isinstance(preceding_words, (Words, type(None)))
        self.term = term
        self.value = value
        self.preceding_words = preceding_words
        self.follower_words: list[Word] = []

    def add_word(self, word: Word):
        assert isinstance(word, Word)
        self.follower_words.append(word)

    def build(self) -> ParsedAndFollower[T]:
        return ParsedAndFollower(term=self.term,
                                 value=self.value,
                                 preceding_words=self.preceding_words,
                                 following_words=(Words(self.follower_words)
                                                  if len(self.follower_words) > 0
                                                  else None))


class ApexTranslator(Generic[T]):
    def __init__(self, terms: typing.Mapping[ApexTermBase, T] | None = None):
        self._apex_term_word_lim = 0
        self._apex_terms_by_first_words: dict[Word, dict[ApexTermBase, T]] = {}
        self._values = set()
        if terms is not None:
            self.add_terms(terms)

    def add_term(self, term: ApexTermBase, val: T):
        self.add_terms({term: val})

    def add_terms(self, terms: typing.Mapping[ApexTermBase, T]):
        self._apex_term_word_lim = max(self._apex_term_word_lim,
                                       max((term.get_max_variation_len() for term in terms),
                                           default=0))
        for term, val in terms.items():
            self._values.add(val)

            for first_word in term.get_possible_first_words():
                terms_elt = self._apex_terms_by_first_words.get(first_word, {})
                if len(terms_elt) == 0:
                    self._apex_terms_by_first_words[first_word] = terms_elt
                terms_elt[term] = val

    def values(self):
        return self._values

    def __contains__(self, term: ApexTermBase):
        first_first_word = next(term.get_possible_first_words())
        return term in self._apex_terms_by_first_words.get(first_first_word, [])

    def translate_terms(self, words: Words) -> Generator[ParsedAndFollower[T], None, None]:
        assert isinstance(words, Words)
        if self._apex_term_word_lim == 0:
            return

        idx = 0
        builder: ParsedAndFollowerBuilder[T] | None = None
        while idx < len(words):
            res = self._translate_term(words[idx:idx + self._apex_term_word_lim])
            if res is not None:
                if builder is not None:
                    yield builder.build()
                    preceding_words: Words | None = None
                else:
                    preceding_words: Words | None = words[:idx] if idx != 0 else None
                apex_term, val, words_inc = res
                builder = ParsedAndFollowerBuilder(term=apex_term,
                                                   value=val,
                                                   preceding_words=preceding_words)
            else:
                if builder is not None:
                    word = words[idx]
                    builder.add_word(word)
                words_inc = 1

            idx += words_inc

        if builder is not None:
            yield builder.build()

    def _translate_term(self, words: Words) -> tuple[ApexTermBase, T, int] | None:
        assert isinstance(words, Words)
        for num_words in range(self._apex_term_word_lim, 0, -1):
            term_to_test = words[:num_words]
            term_and_val: tuple[ApexTermBase, T] | None = self._translate_term2(term_to_test)
            if term_and_val is not None:
                term, val = term_and_val
                return term, val, num_words

        return None

    def _translate_term2(self, term_to_test: Words) -> tuple[ApexTermBase, T] | None:
        assert isinstance(term_to_test, Words)
        first_word = term_to_test.get_first_word()
        term_to_val_dict = self._apex_terms_by_first_words.get(first_word, None)
        if term_to_val_dict is None:
            return None

        for term, val in term_to_val_dict.items():
            if term.has_variation(term_to_test):
                return term, val
        return None
