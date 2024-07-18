from typing import Generic, Iterable, Iterator, Optional, Tuple, TypeVar

from apex_assistant.checker import check_bool, check_tuple, check_type
from apex_assistant.speech.term import RequiredTerm, Word, Words

T = TypeVar('T')


class FoundTerm:
    def __init__(self, preceding_words: Words, following_words: Words):
        check_type(Words,
                   preceding_words=preceding_words,
                   following_words=following_words)
        self._preceding_words = preceding_words
        self._following_words = following_words

    def get_preceding_words(self) -> Words:
        return self._preceding_words

    def get_following_words(self) -> Words:
        return self._following_words


class FindResult:
    def __init__(self,
                 found_any: bool,
                 found_terms: Tuple[FoundTerm, ...],
                 untranslated_words: Words):
        check_bool(found_any=found_any)
        check_tuple(FoundTerm, found_terms=found_terms)
        check_type(Words, untranslated_words=untranslated_words)
        self._found_any = found_any
        self._found_terms = found_terms
        self._untranslated_words = untranslated_words

    def __bool__(self) -> bool:
        return self._found_any

    def __iter__(self) -> Iterator[FoundTerm]:
        return self._found_terms.__iter__()

    def __len__(self):
        return len(self._found_terms)

    def found_any(self) -> bool:
        return self._found_any

    def get_untranslated_words(self) -> Words:
        return self._untranslated_words

    def get_found_terms(self) -> Tuple[FoundTerm, ...]:
        return self._found_terms


class TranslatedTerm(FoundTerm, Generic[T]):
    def __init__(self,
                 term: RequiredTerm,
                 value: T,
                 preceding_words: Words,
                 following_words: Words):
        super().__init__(preceding_words=preceding_words,
                         following_words=following_words)
        check_type(RequiredTerm, term=term)
        check_type(Words,
                   preceding_words=preceding_words,
                   following_words=following_words)
        self.term = term
        self.value = value

    def get_term(self) -> RequiredTerm:
        return self.term

    def get_value(self) -> T:
        return self.value


class TranslatedTermBuilder(Generic[T]):
    def __init__(self, term: RequiredTerm, value: T, preceding_words: Words):
        check_type(RequiredTerm, term=term)
        check_type(Words, preceding_words=preceding_words)
        self.term = term
        self.value = value
        self.preceding_words: Words = preceding_words
        self.follower_words: list[Word] = []

    def add_word(self, word: Word):
        check_type(Word, word=word)
        self.follower_words.append(word)

    def build(self) -> TranslatedTerm[T]:
        return TranslatedTerm(term=self.term,
                              value=self.value,
                              preceding_words=self.preceding_words,
                              following_words=Words(self.follower_words))


class Translation(Generic[T], Iterable[TranslatedTerm[T]]):
    def __init__(self,
                 original_words: Words,
                 translated_terms: Tuple[TranslatedTerm[T], ...]):
        check_type(Words, original_words=original_words)
        check_tuple(allowed_element_types=TranslatedTerm, translated_terms=translated_terms)
        self._original_words = original_words
        self._translated_terms = translated_terms

    def __len__(self) -> int:
        return len(self._translated_terms)

    def __iter__(self) -> Iterator[TranslatedTerm[T]]:
        return iter(self._translated_terms)

    def terms(self) -> Tuple[TranslatedTerm[T]]:
        return self._translated_terms

    def values(self) -> Tuple[T, ...]:
        return tuple(term.get_value() for term in self._translated_terms)

    def get_untranslated_words(self) -> Words:
        if len(self._translated_terms) == 0:
            # Can't recover from what was parsed if no terms were parsed.
            return self._original_words

        words: list[Word] = list(self._translated_terms[0].get_preceding_words())
        for translated_term in self._translated_terms:
            words.extend(translated_term.get_following_words())
        return Words(words)

    def get_latest_value(self) -> Optional[T]:
        return self._translated_terms[-1].get_value()
