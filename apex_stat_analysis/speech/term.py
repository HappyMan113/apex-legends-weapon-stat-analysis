import abc
import string
from types import MappingProxyType
from typing import Generator, Iterable, Iterator

from apex_stat_analysis.checker import check_bool, check_str, check_tuple, check_type


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

    def get_first_word(self) -> Word | None:
        return self.words[0] if len(self.words) > 0 else None

    def __iter__(self) -> Iterator[Word]:
        return self.words.__iter__()


class _TermBase(abc.ABC):
    @abc.abstractmethod
    def get_max_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_min_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def has_variation(self, said_term: Words) -> bool:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_variation_lens(self, said_term: Words) -> Generator[int, None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def has_variation_len(self, n) -> bool:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_human_readable_words(self) -> Generator[Words, None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def __str__(self) -> str:
        return ' '.join(map(str, self.get_human_readable_words()))


class RequiredTerm(_TermBase):
    @abc.abstractmethod
    def get_possible_first_words(self) -> Generator[Word, None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def combine(self, *next_terms: _TermBase) -> 'CombinedTerm':
        check_tuple(_TermBase, allow_empty=False, next_terms=next_terms)
        return CombinedTerm(self, *next_terms)

    def opt(self, include_by_default: bool = False):
        return OptTerm(self, include_by_default=include_by_default)

    def __or__(self, other: 'RequiredTerm') -> 'OrTerm':
        assert isinstance(other, RequiredTerm)
        return self.or_(other)

    def or_(self, *or_operands: 'RequiredTerm') -> 'OrTerm':
        assert all(isinstance(operand, _TermBase) for operand in or_operands)
        return OrTerm(*((self,) + or_operands))

    def __add__(self, other: '_TermBase') -> 'RequiredTerm':
        check_type(_TermBase, other=other)
        return self.combine(other)


class Term(RequiredTerm):
    @staticmethod
    def _map_term(words: str | Iterable[str] | Words) -> Words:
        if not isinstance(words, Words):
            words = Words(words)
        if len(words) == 0:
            raise ValueError('Each "words" object should contain at least one word.')
        return words

    def __init__(self, known_term: str | list[str] | Words, *variations: str | list[str] | Words):
        known_term = self._map_term(known_term)
        variations: tuple[Words, ...] = tuple(map(self._map_term, variations))
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
        self._known_term = known_term

    def get_variation_lens(self, said_term: Words) -> Generator[int, None, None]:
        max_num_words = min(len(said_term), self.get_max_variation_len())
        min_num_words = max(self.get_min_variation_len(), 1)
        for num_words in filter(lambda nw: self.has_variation(said_term[:nw]),
                                range(max_num_words, min_num_words - 1, -1)):
            yield num_words

    def get_max_variation_len(self) -> int:
        return self._max_variation_len

    def get_min_variation_len(self) -> int:
        return self._min_variation_len

    def has_variation(self, said_term: Words) -> bool:
        assert isinstance(said_term, Words)

        if said_term == self._known_term:
            return True

        first_word = said_term.get_first_word()
        variations_n = self._variations_dict.get(first_word, None)
        return variations_n is not None and said_term in variations_n

    def has_variation_len(self, n) -> bool:
        return n == len(self._known_term) or n in self._variations_dict

    def get_human_readable_words(self) -> Generator[Words, None, None]:
        yield self._known_term

    def __repr__(self):
        delim = '|'
        words: list[Words] = [self._known_term]
        for values in self._variations_dict.values():
            words.extend(values)
        return '(' + delim.join(map(repr, words)) + ')'

    def __hash__(self):
        return self._known_term.__hash__()

    def __eq__(self, other):
        return self._known_term.__eq__(other)

    def get_possible_first_words(self) -> Generator[Word | None, None, None]:
        yield self._known_term.get_first_word()
        for first_word in self._variations_dict:
            yield first_word


class OptTerm(_TermBase):
    def __init__(self, opt_word: RequiredTerm, include_by_default: bool):
        check_type(RequiredTerm, opt_word=opt_word)
        check_bool(include_by_default=include_by_default)

        self.wrapped = opt_word
        self.include_by_default = include_by_default

    def get_max_variation_len(self) -> int:
        return self.wrapped.get_max_variation_len()

    def get_min_variation_len(self) -> int:
        return 0

    def get_human_readable_words(self) -> Generator[Words, None, None]:
        if self.include_by_default:
            for words in self.wrapped.get_human_readable_words():
                yield words

    def __repr__(self):
        return f'[{repr(self.wrapped)}]'

    def get_variation_lens(self, said_term: Words) -> Generator[int, None, None]:
        for num_words in self.wrapped.get_variation_lens(said_term):
            yield num_words
        yield 0

    def has_variation(self, said_term: Words) -> bool:
        return self.wrapped.has_variation(said_term)

    def has_variation_len(self, n) -> bool:
        return n == 0 or self.wrapped.has_variation_len(n)

    def __add__(self, other: 'RequiredTerm | OptTerm') -> '_TermBase':
        return self.combine(other)

    def combine(self, next_term: 'RequiredTerm | OptTerm') -> 'RequiredTerm | OptTerm':
        check_type((RequiredTerm, OptTerm), next_term=next_term)
        if isinstance(next_term, RequiredTerm):
            with_opt_term = CombinedTerm(self.wrapped, next_term)
            if self.include_by_default:
                return with_opt_term | next_term
            return next_term | with_opt_term
        elif isinstance(next_term, OptTerm):
            # We're gonna have to account for all 4 possible permutations.
            self_wrapped = self.wrapped
            next_wrapped = next_term.wrapped
            self_and_next_wrapped = self_wrapped + next_wrapped
            if self.include_by_default and next_term.include_by_default:
                req_terms = (self_and_next_wrapped, self_wrapped, next_wrapped)
                include_by_default = True
            elif self.include_by_default:
                req_terms = (self_wrapped, next_wrapped, self_and_next_wrapped)
                include_by_default = True
            elif next_term.include_by_default:
                req_terms = (next_wrapped, self_wrapped, self_and_next_wrapped)
                include_by_default = True
            else:
                req_terms = (next_wrapped, self_wrapped, self_and_next_wrapped)
                include_by_default = False
            return OrTerm(*req_terms).opt(include_by_default=include_by_default)
        else:
            raise TypeError('Can only combine optional or required terms.')


class CombinedTerm(RequiredTerm):
    def __init__(self, first_sub_term: RequiredTerm, *sub_terms: _TermBase):
        check_type(RequiredTerm, first_sub_term=first_sub_term)
        check_tuple(_TermBase, sub_terms=sub_terms)
        if len(sub_terms) < 1:
            raise ValueError('Must provide at required term and at least one additional term '
                             f'to combine into the {self.__class__.__name__} initializer.')
        self.first_sub_term = first_sub_term
        sub_terms: tuple[_TermBase, ...] = (first_sub_term,) + sub_terms
        self.max_variation_len = sum(term.get_max_variation_len() for term in sub_terms)
        self.min_variation_len = sum(term.get_min_variation_len() for term in sub_terms)
        self.sub_terms = sub_terms

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def has_variation(self, said_term: Words) -> bool:
        return any(len(said_term) == num_words
                   for num_words in self.get_variation_lens(said_term))

    def get_variation_lens(self, said_term: Words) -> Generator[int, None, None]:
        return self.__get_variation_lens(0, self.sub_terms, said_term)

    def __get_variation_lens(self,
                             tot_num_words: int,
                             sub_terms: tuple[_TermBase, ...],
                             said_term: Words | None,
                             already_yielded: set[int] = set[int]()) -> Generator[int, None, None]:
        if len(sub_terms) == 0:
            raise RuntimeError('Must call this method with at least one term. Makes no sense '
                               'otherwise.')

        cur_term = sub_terms[0]
        if len(sub_terms) == 1:
            for num_words in cur_term.get_variation_lens(said_term):
                result = tot_num_words + num_words
                if result not in already_yielded:
                    already_yielded.add(result)
                yield result
            return

        for num_words in cur_term.get_variation_lens(said_term):
            if num_words > len(said_term):
                continue

            for result in self.__get_variation_lens(tot_num_words=tot_num_words + num_words,
                                                    sub_terms=sub_terms[1:],
                                                    said_term=said_term[num_words:],
                                                    already_yielded=already_yielded):
                yield result

    def has_variation_len(self, n) -> bool:
        # Not totally accurate, but good enough for now.
        return self.max_variation_len >= n >= self.min_variation_len

    def get_human_readable_words(self) -> Generator[Words, None, None]:
        for sub_term in self.sub_terms:
            for words in sub_term.get_human_readable_words():
                yield words

    def __repr__(self) -> str:
        return '(' + ' '.join(map(repr, self.sub_terms)) + ')'

    def get_possible_first_words(self) -> Generator[Word, None, None]:
        for word in self.first_sub_term.get_possible_first_words():
            yield word


class OrTerm(RequiredTerm):
    def __init__(self, *or_operands: RequiredTerm):
        assert all(isinstance(operand, RequiredTerm) for operand in or_operands)
        assert len(or_operands) > 1
        self.or_operands = or_operands
        self.max_variation_len = max(operand.get_max_variation_len()
                                     for operand in self.or_operands)
        self.min_variation_len = min(operand.get_max_variation_len()
                                     for operand in self.or_operands)

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def get_variation_lens(self, said_term: Words) -> Generator[int, None, None]:
        accum = set[int]()
        for operand in self.or_operands:
            for variation_len in operand.get_variation_lens(said_term):
                if variation_len not in accum:
                    accum.add(variation_len)
                    yield variation_len

    def has_variation_len(self, n) -> bool:
        return any(operand.has_variation_len(n)
                   for operand in self.or_operands)

    def has_variation(self, said_term: Words) -> bool:
        return any(operand.has_variation(said_term)
                   for operand in self.or_operands)

    def get_possible_first_words(self) -> Generator[Word, None, None]:
        for operand in self.or_operands:
            for possible_first_word in operand.get_possible_first_words():
                yield possible_first_word

    def get_human_readable_words(self) -> Generator[Words, None, None]:
        for words in self.or_operands[0].get_human_readable_words():
            yield words

    def __repr__(self):
        return '(' + ' | '.join(map(repr, self.or_operands)) + ')'


class IntTerm(Term):
    def __init__(self, value: int, *variations: str):
        super().__init__(str(value), *variations)
        self.value = value

    def __int__(self):
        return self.value

    def combine_int(self, *next_terms: 'IntTerm'):
        ints_str = ''.join(map(str, map(int, next_terms)))
        return super().combine(*next_terms) | Term(ints_str)
