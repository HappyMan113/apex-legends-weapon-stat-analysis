import abc
import string
from enum import StrEnum
from types import MappingProxyType
from typing import Generator, Iterable, Iterator, TypeAlias, Union

from apex_assistant.checker import check_bool, check_int, check_str, check_tuple, check_type


class Word:
    _TRANSLATE_TABLE = str.maketrans('', '', string.punctuation)

    def __init__(self, original_word: str):
        check_str(original_word=original_word, allow_blank=False)
        stripped_word = original_word.translate(Word._TRANSLATE_TABLE).lower()
        check_str(stripped_word=stripped_word, allow_blank=False)
        self.original_word = original_word
        self.stripped_word = stripped_word

    def __repr__(self):
        return self.stripped_word

    def __str__(self):
        return self.original_word

    def __hash__(self):
        return hash(self.stripped_word)

    def __eq__(self, other: 'Word'):
        return isinstance(other, Word) and self.stripped_word == other.stripped_word


class Words:
    """ More or less just a glorified tuple of Word. """

    def __init__(self, words: str | Iterable[Word]):
        if isinstance(words, str):
            words = tuple(map(Word, words.split(' ')))
        elif not isinstance(words, tuple):
            words = tuple(words)
            check_tuple(Word, words=words)
        self.words = words

    def __repr__(self):
        return ' '.join(map(repr, self.words))

    def __str__(self):
        return ' '.join(map(str, self.words))

    def get_words(self) -> tuple[Word]:
        return self.words

    def get_stripped_words(self) -> tuple[str]:
        return tuple(word.stripped_word for word in self.words)

    def __hash__(self):
        return hash(self.words)

    def __eq__(self, other):
        return isinstance(other, Words) and self.words == other.words

    def __getitem__(self, sl: int | slice):
        assert isinstance(sl, (int, slice))
        if isinstance(sl, int):
            return self.words[sl]
        return Words(self.words[sl])

    def __add__(self, other: 'Words'):
        return Words(self.words + other.words)

    def __len__(self):
        return len(self.words)

    def get_first_word(self) -> Word:
        if len(self.words) == 0:
            raise RuntimeError('There is no first word!')
        return self.words[0]

    def __iter__(self) -> Iterator[Word]:
        return self.words.__iter__()

    def __contains__(self, term: Union['TermBase', Word]):
        if isinstance(term, Word):
            return term in self.words
        if isinstance(term, TermBase):
            # noinspection PyProtectedMember
            return term._is_contained_in_words(self)

        raise TypeError(
            f'__contains__ only accepts objects of type Word or TermBase, which {term} is not')


class _WordsAreForWhat(StrEnum):
    FOR_SPEECH = 'for_speech'
    FOR_TEXT = 'for_text'


REQ_TERM: TypeAlias = 'RequiredTerm'
OPT_TERM: TypeAlias = 'OptTerm'
TERM: TypeAlias = Union[REQ_TERM, OPT_TERM, 'TermBase']


class TermBase(abc.ABC):
    @abc.abstractmethod
    def get_max_variation_len(self) -> int:
        """Gets upper bound of the maximum number of words that could translate into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_min_variation_len(self) -> int:
        """Gets lower bound of the minimum number of words that could translate into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def has_variation(self, said_words: Words) -> bool:
        """Determines if the given words translate directly into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_variation_lens(self, said_words: Words) -> Generator[int, None, None]:
        """Gets each number, n, where, said_words[:n] translates into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def has_variation_len(self, n: int) -> bool:
        """Determines if a number of words, n, can translate into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_human_readable_words(self, for_what: _WordsAreForWhat) -> Generator[Words, None, None]:
        """Gets the words in this term that would be good for speaking or printing."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Gets a technically detailed representation of this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def __str__(self) -> str:
        """Gets a string representation of this term that would be good for speaking or printing."""
        return ' '.join(map(str, self.get_human_readable_words(_WordsAreForWhat.FOR_TEXT)))

    def to_audible_str(self) -> str:
        """Gets a string representation of this term that would be good for speaking or printing."""
        return ' '.join(map(str, self.get_human_readable_words(_WordsAreForWhat.FOR_SPEECH)))

    def __add__(self, next_term: TERM) -> TERM:
        """Shorthand for self.append(next_term)."""
        check_type(TermBase, next_term=next_term)
        return self.append(next_term)

    @abc.abstractmethod
    def append(self, *next_terms: TERM) -> TERM:
        """
        For indices n_1, n_2, n_3,...n_m in ascending order, a given Words object, "words",
        translates into a combined term self.append(next_term_1, next_term_2,... next_term_m) if
        words[:n_1] translates into self; words[n_1:n_2] translates into next_term_1; words[n_2:n_3]
        translates into next_term_2;... and words[n_m:] translates into next_term_m.
        """
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def _insert_opt(self, term_to_insert: OPT_TERM) -> TERM:
        """
        Gets a term that has the given optional term inserted before self, such that a Word, "word"
        will translate into the resulting term if words translates into the wrapped term
        contained in self, "wrapped"; or if words translates into (term_to_insert.wrapped +
        wrapped).
        """
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def _is_contained_in_words(self, said_words: Words) -> bool:
        """Determines if any slice of the given words translates into this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def _insert_order_agnostic(self, other_term: TERM) -> REQ_TERM:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')


class RequiredTerm(TermBase):
    @abc.abstractmethod
    def get_possible_first_words(self) -> Generator[Word, None, None]:
        """Gets the first word in each Words object that translates to this term."""
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def __add__(self, next_term: TERM) -> REQ_TERM:
        check_type(TermBase, next_term=next_term)
        return self.append(next_term)

    def append(self, *next_terms: TERM) -> REQ_TERM:
        check_tuple(TermBase, allow_empty=False, next_terms=next_terms)
        return CombinedTerm(self, *next_terms)

    def opt(self, include_in_speech: bool = False) -> OPT_TERM:
        """
        A word translates into the resulting optional term if the word translates into self or if
        the word is blank.
        """
        check_bool(include_in_speech=include_in_speech)
        return OptTerm(self, include_in_speech=include_in_speech)

    def __or__(self, other: REQ_TERM) -> REQ_TERM:
        """Shorthand for self.or_any(other)."""
        check_type(RequiredTerm, other=other)
        return self.or_any(other)

    def or_any(self, *or_operands: REQ_TERM) -> REQ_TERM:
        """
        A given Words object, "words", translates into self.or_any(*or_operands) if words translates
        into self, or if words translates into any of the operands in or_operands.
        """
        check_tuple(allowed_element_types=RequiredTerm, allow_empty=False, or_operands=or_operands)
        return OrTerm(*((self,) + or_operands))

    def _insert_opt(self, term_to_insert: OPT_TERM) -> REQ_TERM:
        without_opt_term = self
        with_opt_term = CombinedTerm(term_to_insert.get_wrapped_term(), without_opt_term)
        if term_to_insert.included_by_default():
            return with_opt_term | without_opt_term
        return without_opt_term | with_opt_term

    def _is_contained_in_words(self, said_words: Words) -> bool:
        check_type(Words, said_words=said_words)
        return any(next(self.get_variation_lens(said_words[start_idx:]), None) is not None
                   for start_idx in range(len(said_words)))

    def _insert_order_agnostic(self, other_term: REQ_TERM) -> REQ_TERM:
        return (self + other_term) | (self + other_term)

    def append_order_agnostic(self, other_term: TERM) -> REQ_TERM:
        return other_term._insert_order_agnostic(self)


class Term(RequiredTerm):
    @staticmethod
    def _map_term(words: str | Iterable[Word] | Words) -> Words:
        """
        Converts the given string or Iterable of "Word"s into a Words object, ensuring that there is
        at least one resulting Word in the Words object to ensure that a first word can be extracted
        from it.
        """
        if not isinstance(words, Words):
            words = Words(words)
        if len(words) == 0:
            raise ValueError('Each "words" object should contain at least one word.')
        return words

    def __init__(self,
                 known_term: str | Iterable[Word] | Words,
                 *variations: str | Iterable[Word] | Words):
        known_term = self._map_term(known_term)
        variations: set[Words, ...] = set(map(self._map_term, variations)) - {known_term}
        variations_dict: dict[Word, set[Words]] = {}
        for variation in variations:
            first_word = variation.get_first_word()
            if first_word not in variations_dict:
                variations_n: set[Words] = set()
                variations_dict[first_word] = variations_n
            else:
                variations_n = variations_dict[first_word]
            variations_n.add(variation)
        self._variations_dict = MappingProxyType(variations_dict)
        self._min_variation_len = min(len(known_term), min(map(len, variations),
                                                           default=len(known_term)))
        self._max_variation_len = max(len(known_term), max(map(len, variations),
                                                           default=len(known_term)))
        self._known_words = known_term
        self._first_words: set[Word] = {known_term.get_first_word()} | set(self._variations_dict)

    def get_variation_lens(self, said_words: Words) -> Generator[int, None, None]:
        check_type(Words, said_words=said_words)
        max_num_words = min(len(said_words), self.get_max_variation_len())
        min_num_words = max(self.get_min_variation_len(), 1)
        for num_words in filter(lambda nw: self.has_variation(said_words[:nw]),
                                range(max_num_words, min_num_words - 1, -1)):
            yield num_words

    def get_max_variation_len(self) -> int:
        return self._max_variation_len

    def get_min_variation_len(self) -> int:
        return self._min_variation_len

    def has_variation(self, said_words: Words) -> bool:
        check_type(Words, said_words=said_words)
        if len(said_words) == 0:
            return False

        if said_words == self._known_words:
            return True

        first_word = said_words.get_first_word()
        variations_n = self._variations_dict.get(first_word, None)
        return variations_n is not None and said_words in variations_n

    def has_variation_len(self, n) -> bool:
        return n == len(self._known_words) or n in self._variations_dict

    def get_human_readable_words(self, for_what: _WordsAreForWhat) -> Generator[Words, None, None]:
        yield self._known_words

    def __repr__(self) -> str:
        delim = '|'
        words: list[Words] = [self._known_words]
        for values in self._variations_dict.values():
            words.extend(values)
        return '{' + delim.join(map(repr, words)) + '}'

    def __hash__(self) -> int:
        return self._known_words.__hash__()

    def __eq__(self, other) -> bool:
        return self._known_words.__eq__(other)

    def get_possible_first_words(self) -> Generator[Word | None, None, None]:
        for first_word in self._first_words:
            yield first_word


class OptTerm(TermBase):
    def __init__(self, opt_word: RequiredTerm, include_in_speech: bool):
        check_type(RequiredTerm, opt_word=opt_word)
        check_bool(include_in_speech=include_in_speech)

        self._wrapped = opt_word
        self._include_in_speech = include_in_speech

    def get_max_variation_len(self) -> int:
        return self._wrapped.get_max_variation_len()

    def get_min_variation_len(self) -> int:
        return 0

    def get_human_readable_words(self, for_what: _WordsAreForWhat) -> Generator[Words, None, None]:
        check_type(_WordsAreForWhat, for_what=for_what)
        if self._include_in_speech or (for_what is not _WordsAreForWhat.FOR_SPEECH):
            for words in self._wrapped.get_human_readable_words(for_what):
                yield words

    def __repr__(self):
        return f'[{repr(self._wrapped)}]'

    def get_variation_lens(self, said_words: Words) -> Generator[int, None, None]:
        check_type(Words, said_words=said_words)
        for num_words in self._wrapped.get_variation_lens(said_words):
            yield num_words
        yield 0

    def has_variation(self, said_words: Words) -> bool:
        check_type(Words, said_words=said_words)
        return self._wrapped.has_variation(said_words)

    def has_variation_len(self, n: int) -> bool:
        check_int(min_value=0, n=n)
        return n == 0 or self._wrapped.has_variation_len(n)

    def append(self, *next_terms: TERM) -> TERM:
        check_tuple(TermBase, allow_empty=False, next_terms=next_terms)
        # noinspection PyProtectedMember
        resulting_term: TERM = next_terms[0]._insert_opt(self)
        if len(next_terms) > 1:
            resulting_term = resulting_term.append(*next_terms[1:])
        return resulting_term

    def _insert_opt(self, term_to_insert: OPT_TERM) -> OPT_TERM:
        check_type(OptTerm, term_to_insert=term_to_insert)
        # We're going to have to account for all 4 possible permutations.
        term1: RequiredTerm = term_to_insert.get_wrapped_term()
        term2: RequiredTerm = self.get_wrapped_term()
        incl_term1 = term_to_insert.included_by_default()
        incl_term2 = self.included_by_default()
        both_terms = term1 + term2
        if incl_term1 and incl_term2:
            req_terms = (both_terms, term1, term2)
            include_in_speech = True
        elif incl_term2:
            req_terms = (term2, term1, both_terms)
            include_in_speech = True
        else:
            req_terms = (term1, term2, both_terms)
            include_in_speech = incl_term1
        return OrTerm(*req_terms).opt(include_in_speech=include_in_speech)

    def get_wrapped_term(self) -> RequiredTerm:
        """Gets the required term that is wrapped by this optional term."""
        return self._wrapped

    def included_by_default(self) -> bool:
        """Gets whether the wrapped required term is included by default in speakable strings."""
        return self._include_in_speech

    def _is_contained_in_words(self, said_words: Words) -> bool:
        return True

    def _insert_order_agnostic(self, other_term: REQ_TERM) -> REQ_TERM:
        wrapped = self.get_wrapped_term()
        return (((wrapped + other_term) | (other_term + wrapped) | other_term)
                if self.included_by_default() else
                (other_term | (wrapped + other_term) | (other_term + wrapped)))


class CombinedTerm(RequiredTerm):
    def __init__(self, first_sub_term: RequiredTerm, *sub_terms: TermBase):
        check_type(RequiredTerm, first_sub_term=first_sub_term)
        check_tuple(TermBase, allow_empty=False, sub_terms=sub_terms)

        self.first_sub_term = first_sub_term
        sub_terms: tuple[TermBase, ...] = (first_sub_term,) + sub_terms
        self.max_variation_len = sum(term.get_max_variation_len() for term in sub_terms)
        self.min_variation_len = sum(term.get_min_variation_len() for term in sub_terms)
        self.sub_terms = sub_terms

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def has_variation(self, said_words: Words) -> bool:
        check_type(Words, said_words=said_words)
        return any(len(said_words) == num_words
                   for num_words in self.get_variation_lens(said_words))

    def get_variation_lens(self, said_words: Words) -> Generator[int, None, None]:
        check_type(Words, said_words=said_words)
        return self.__get_variation_lens(0, self.sub_terms, said_words)

    def __get_variation_lens(self,
                             tot_num_words: int,
                             sub_terms: tuple[TermBase, ...],
                             said_words: Words | None,
                             already_yielded: set[int] = set[int]()) -> Generator[int, None, None]:
        cur_term = sub_terms[0]
        if len(sub_terms) == 1:
            for num_words in cur_term.get_variation_lens(said_words):
                result = tot_num_words + num_words
                if result not in already_yielded:
                    already_yielded.add(result)
                yield result
            return

        for num_words in cur_term.get_variation_lens(said_words):
            if num_words > len(said_words):
                continue

            for result in self.__get_variation_lens(tot_num_words=tot_num_words + num_words,
                                                    sub_terms=sub_terms[1:],
                                                    said_words=said_words[num_words:],
                                                    already_yielded=already_yielded):
                yield result

    def has_variation_len(self, n: int) -> bool:
        check_int(min_value=0, n=n)
        # Not totally accurate, but good enough for now.
        return self.max_variation_len >= n >= self.min_variation_len

    def get_human_readable_words(self, for_what: _WordsAreForWhat) -> Generator[Words, None, None]:
        for sub_term in self.sub_terms:
            for words in sub_term.get_human_readable_words(for_what):
                yield words

    def __repr__(self) -> str:
        return '(' + ' '.join(map(repr, self.sub_terms)) + ')'

    def get_possible_first_words(self) -> Generator[Word, None, None]:
        for word in self.first_sub_term.get_possible_first_words():
            yield word


class OrTerm(RequiredTerm):
    def __init__(self, *or_operands: RequiredTerm):
        check_tuple(RequiredTerm, allow_empty=False, or_operands=or_operands)
        if len(or_operands) < 2:
            raise ValueError('Must specify at least two OR operands.')
        self.or_operands = or_operands
        self.max_variation_len = max(operand.get_max_variation_len()
                                     for operand in self.or_operands)
        self.min_variation_len = min(operand.get_max_variation_len()
                                     for operand in self.or_operands)

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def get_variation_lens(self, said_words: Words) -> Generator[int, None, None]:
        check_type(Words, said_words=said_words)
        accum = set[int]()
        for operand in self.or_operands:
            for variation_len in operand.get_variation_lens(said_words):
                if variation_len not in accum:
                    accum.add(variation_len)
                    yield variation_len

    def has_variation_len(self, n: int) -> bool:
        check_int(min_value=0, n=n)
        return any(operand.has_variation_len(n)
                   for operand in self.or_operands)

    def has_variation(self, said_words: Words) -> bool:
        check_type(Words, said_words=said_words)
        return any(operand.has_variation(said_words)
                   for operand in self.or_operands)

    def get_possible_first_words(self) -> Generator[Word, None, None]:
        for operand in self.or_operands:
            for possible_first_word in operand.get_possible_first_words():
                yield possible_first_word

    def get_human_readable_words(self, for_what: _WordsAreForWhat) -> Generator[Words, None, None]:
        for words in self.or_operands[0].get_human_readable_words(for_what):
            yield words

    def __repr__(self):
        return '(' + ' | '.join(map(repr, self.or_operands)) + ')'


class IntTerm(Term):
    def __init__(self, value: int, *variations: str):
        check_int(min_value=0, value=value)
        super().__init__(str(value), *variations)
        self.value = value

    def __int__(self):
        return self.value

    def append_int(self, *next_terms: 'IntTerm'):
        ints_str = ''.join(map(str, map(int, next_terms)))
        return super().append(*next_terms) | Term(ints_str)
