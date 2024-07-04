import abc
import re
from types import MappingProxyType
import typing

def strip_punctuation(text: str):
    return re.sub(r'[.,!?()\]\[{}]', '', text)


class Word:
    def __init__(self, original_word: str):
        if len(original_word) == 0:
            raise ValueError('Original word is empty.')
        stripped_word = strip_punctuation(original_word).lower().replace('-', '')
        if len(stripped_word) == 0:
            raise ValueError('Stripped word is empty.')
        self.original_word = original_word
        self.stripped_word = stripped_word

    def __repr__(self):
        return self.original_word

    def __hash__(self):
        return hash(self.stripped_word)

    def __eq__(self, other: 'Word'):
        return isinstance(other, Word) and self.stripped_word == other.stripped_word


class Words:
    """ More or less just a glorified tuple of Word. """

    def __init__(self, words: str | typing.Iterable[Word]):
        if isinstance(words, str):
            words = tuple(map(Word, words.split(' ')))
        elif not isinstance(words, tuple):
            words = tuple(words)
        if len(words) == 0:
            raise ValueError('Must have at least one word.')
        self.words = words

    def __repr__(self):
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

    def __iter__(self) -> typing.Iterator[Word]:
        return self.words.__iter__()


class ApexTermBase(abc.ABC):
    @abc.abstractmethod
    def get_max_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_min_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def has_variation(self, said_term: Words) -> bool:
        return any(len(said_term) == num_words
                   for num_words in self.get_variation_lens(said_term))

    @abc.abstractmethod
    def get_variation_lens(self, said_term: Words) -> typing.Generator[int, None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def has_variation_len(self, n) -> bool:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def __add__(self, other):
        return self.combine(other)

    def combine(self, *next_terms: 'ApexTermBase'):
        return CombinedTerm(self, *next_terms)

    def combine_order_agnostic(self, *next_terms: 'ApexTermBase'):
        return OrderAgnosticCombinedTerm(self, *next_terms)

    def opt(self):
        return OptTerm(self)

    @abc.abstractmethod
    def get_possible_first_words(self) -> typing.Generator[Word | None, None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')


class ConcreteApexTerm(ApexTermBase):
    @staticmethod
    def _map_term(term: str | typing.Iterable[str] | Words) -> Words:
        return term if isinstance(term, Words) else Words(term)

    def __init__(self, known_term: str | list[str] | Words, *variations: str | list[str] | Words):
        known_term = self._map_term(known_term)
        variations: tuple[Words] = tuple(map(self._map_term, variations))
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

    def get_variation_lens(self, said_term: Words) -> typing.Generator[int, None, None]:
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

    def __repr__(self):
        return self._known_term.__repr__()

    def __str__(self):
        return self._known_term.__str__()

    def __hash__(self):
        return self._known_term.__hash__()

    def __eq__(self, other):
        return self._known_term.__eq__(other)

    def with_variations(self, *variations: str | list[str] | Words):
        new_variations: tuple[Words] = (
                tuple(term
                      for term_set in self._variations_dict.values()
                      for term in term_set) +
                tuple(map(self._map_term, variations)))
        return ConcreteApexTerm(self._known_term, *new_variations)

    def get_possible_first_words(self) -> typing.Generator[Word | None, None, None]:
        yield self._known_term.get_first_word()
        for first_word in self._variations_dict:
            yield first_word


class OptTerm(ApexTermBase):
    def __init__(self, opt_word: ApexTermBase):
        self.wrapped = opt_word

    def get_max_variation_len(self) -> int:
        return self.wrapped.get_max_variation_len()

    def get_min_variation_len(self) -> int:
        return 0

    def __repr__(self):
        return f'[{self.wrapped}]'

    def get_variation_lens(self, said_term: Words) -> typing.Generator[int, None, None]:
        for num_words in self.wrapped.get_variation_lens(said_term):
            yield num_words
        yield 0

    def has_variation(self, said_term: Words) -> bool:
        return self.wrapped.has_variation(said_term)

    def has_variation_len(self, n) -> bool:
        return n == 0 or self.wrapped.has_variation_len(n)

    def get_possible_first_words(self) -> typing.Generator[Word | None, None, None]:
        yield None
        for word in self.wrapped.get_possible_first_words():
            yield word


class CombinedTerm(ApexTermBase):
    def __init__(self, *sub_terms: ApexTermBase):
        assert len(sub_terms) > 0
        self.sub_terms = sub_terms
        self.max_variation_len = sum(term.get_max_variation_len() for term in sub_terms)
        self.min_variation_len = sum(term.get_min_variation_len() for term in sub_terms)

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def get_variation_lens(self, said_term: Words) -> typing.Generator[int, None, None]:
        return self.__get_variation_lens(0, self.sub_terms, said_term)

    def __get_variation_lens(self,
                             tot_num_words: int,
                             sub_terms: tuple[ApexTermBase],
                             said_term: Words) -> \
            typing.Generator[int, None, None]:
        assert len(sub_terms) > 0
        if len(sub_terms) == 1:
            for num_words in sub_terms[0].get_variation_lens(said_term):
                yield tot_num_words + num_words
            return

        for num_words in sub_terms[0].get_variation_lens(said_term):
            if num_words >= len(said_term):
                continue

            for result in self.__get_variation_lens(tot_num_words=tot_num_words + num_words,
                                                    sub_terms=sub_terms[1:],
                                                    said_term=said_term[num_words:]):
                yield result

    def has_variation_len(self, n) -> bool:
        # Not totally accurate, but good enough for now.
        return self.max_variation_len >= n >= self.min_variation_len

    def __repr__(self) -> str:
        return ' '.join([(str(term)
                          if not term.has_variation_len(0)
                          else f'[{term}]')
                         for term in self.sub_terms])

    def get_possible_first_words(self) -> typing.Generator[Word | None, None, None]:
        for term in self.sub_terms[0].get_possible_first_words():
            yield term


class OrderAgnosticCombinedTerm(CombinedTerm):
    def get_variation_lens(self, said_term: Words) -> typing.Generator[int, None, None]:
        return self.__get_variation_lens2(0, set(self.sub_terms), said_term)

    def __get_variation_lens2(self,
                             tot_num_words: int,
                             sub_terms: set[ApexTermBase],
                             said_term: Words) -> \
            typing.Generator[int, None, None]:
        assert len(sub_terms) > 0
        if len(sub_terms) == 1:
            sub_term = next(iter(sub_terms))
            for num_words in sub_term.get_variation_lens(said_term):
                yield tot_num_words + num_words
            return

        for sub_term in sub_terms:
            for num_words in sub_term.get_variation_lens(said_term):
                if num_words >= len(said_term):
                    continue

                for result in self.__get_variation_lens2(
                        tot_num_words=tot_num_words + num_words,
                        sub_terms=sub_terms - {sub_term},
                        said_term=said_term[num_words:]):
                    yield result

    def __repr__(self):
        return '{' + CombinedTerm.__repr__(self) + '}'

    def get_possible_first_words(self) -> typing.Generator[Word | None, None, None]:
        for sub_term in self.sub_terms:
            for first_word in sub_term.get_possible_first_words():
                yield first_word


class IntTerm(ConcreteApexTerm):
    def __init__(self, value: int, *variations: str):
        super().__init__(str(value), *variations)
        self.value = value

    def __int__(self):
        return self.value


class ApexTerms:
    COMPARE = ConcreteApexTerm('compare', 'which is better', 'which weapon is better',
                               'which one is better', 'what\'s better')
    BEST = ConcreteApexTerm('best', 'best weapons', 'that\'s')
    STOP = ConcreteApexTerm('stop')
    COMMANDS = (COMPARE, BEST, STOP)

    WITH_SIDEARM = ConcreteApexTerm('with sidearm', 'and sidearm', 'switching to sidearm',
                                    'and then switching to', 'with sidearm of', 'sidearm',
                                    'with secondary', 'with secondary weapon',
                                    'with secondary weapon of')

    NO_MAG_TERM = ConcreteApexTerm('no mag', 'no magazine', 'nomag')
    ALL_MAG_TERMS: typing.Tuple[ConcreteApexTerm] = (
        NO_MAG_TERM,
        ConcreteApexTerm('level 1 mag',
                         'level 1 magazine',
                         'white mag',
                         'white magazine',
                         'lv. 1 mag',
                         'lv. 1 magazine'),
        ConcreteApexTerm('level 2 mag',
                         'level 2 magazine',
                         'blue mag',
                         'blue magazine',
                         'lv. 2 mag',
                         'lv. 2 magazine'),
        ConcreteApexTerm('level 3 or 4 mag',
                         'level 3 or 4 magazine',
                         'lv. 3 or 4 magazine',
                         'level 3 magazine',
                         'purple mag',
                         'purple magazine',
                         'level 4 mag',
                         'level 4 magazine',
                         'lv. 3 mag',
                         'lv. 3 magazine',
                         'lv. 4 mag',
                         'lv. 4 magazine'),
    )

    NO_STOCK_TERM = ConcreteApexTerm('no stock', 'no standard stock', 'no sniper stock')
    ALL_STOCK_TERMS: typing.Tuple[ConcreteApexTerm] = (
        NO_STOCK_TERM,
        ConcreteApexTerm('level 1 stock',
                         'level 1 standard stock',
                         'level 1 stock stock',
                         'white stock'
                         'white standard stock'
                         'white sniper stock',
                         'lv. 1 stock',
                         'lv. 1 standard stock',
                         'lv. 1 sniper stock'),
        ConcreteApexTerm('level 2 stock',
                         'level 2 standard stock',
                         'level 2 stock stock',
                         'blue stock'
                         'blue standard stock'
                         'blue sniper stock',
                         'lv. 2 stock',
                         'lv. 2 standard stock',
                         'lv. 2 sniper stock'),
        ConcreteApexTerm('level 3 or 4 stock',
                         'level 3 stock',
                         'level 3 standard stock',
                         'level 3 sniper stock',
                         'purple stock',
                         'purple standard stock',
                         'purple sniper stock',
                         'level 4 stock',
                         'level 4 standard stock',
                         'level 4 sniper stock',
                         'golden stock',
                         'golden standard stock',
                         'golden sniper stock',
                         'lv. 3 stock',
                         'lv. 3 standard stock',
                         'lv. 3 sniper stock',
                         'lv. 4 stock',
                         'lv. 4 standard stock',
                         'lv. 4 sniper stock'),
    )

    NO_RELOAD_TERM = ConcreteApexTerm('no reload',
                                      'no reloads',
                                      'no reloading',
                                      'without reload',
                                      'without reloads',
                                      'without reloading',
                                      'with no reload',
                                      'with no reloads',
                                      'with no reloading')
    WITH_RELOAD_TERM = ConcreteApexTerm('with reloads',
                                        'with reload',
                                        'with reloading',
                                        'reloading',
                                        'reloaded')

    NO_BOLT_TERM = ConcreteApexTerm('no bolt')
    ALL_BOLT_TERMS = (NO_BOLT_TERM,
                      ConcreteApexTerm('level 1 bolt',
                                       'lv. 1 bolt',
                                       'white bolt',
                                       'level 1 shotgun bolt',
                                       'shotgun bolt level 1',
                                       'shotgun bolt lv. 1',
                                       'lv. 1 shotgun bolt'),
                      ConcreteApexTerm('level 2 bolt',
                                       'lv. 2 bolt',
                                       'blue bolt',
                                       'blue shotgun bolt'
                                       'level 2 shotgun bolt',
                                       'lv. 2 shotgun bolt',
                                       'shotgun bolt level 2',
                                       'shotgun bolt lv. 2',
                                       'lv. 2 shotgun bolt'),
                      ConcreteApexTerm('level 3 or 4 bolt',
                                       'lv. 3 or 4 bolt',
                                       'level 3 bolt',
                                       'lv. 3 bolt',
                                       'level 3 shotgun bolt'
                                       'lv. 3 shotgun bolt'
                                       'purple bolt',
                                       'purple shotgun bolt',
                                       'level 4 bolt',
                                       'lv. 4 bolt',
                                       'level 4 shotgun bolt',
                                       'lv. 4 shotgun bolt',
                                       'golden bolt',
                                       'golden shotgun bolt')
                      )

    THIRTY_THIRTY_REPEATER = ConcreteApexTerm(
        '30-30', '33', 'there are three', '30 seconds ready', 'very very', '3030')
    BOCEK = ConcreteApexTerm('Bocek', 'bow check')
    CAR = ConcreteApexTerm(
        'C.A.R.', 'car', 'TARG', 'cut', 'tar', 'sorry', 'tower', 'tart', 'CAR-SMG')
    CARE_PACKAGE = ConcreteApexTerm('care package')
    ALTERNATOR = ConcreteApexTerm('Alternator', 'I don\'t need her', 'I\'ll do neither')
    CHARGE_RIFLE = ConcreteApexTerm(
        'Charge Rifle', 'charger full', 'charged rifle', 'Ciao Dreffel', 'charger')
    FAR = ConcreteApexTerm('far', 'bar', 'bark')
    RAMPAGE = ConcreteApexTerm('Rampage', 'my page', 'webpage')
    DEVOTION = ConcreteApexTerm(
        'Devotion', 'version', 'the ocean', 'lotion', 'the motion', 'motion', 'ocean')
    SHATTER_CAPS = ConcreteApexTerm('shatter caps', 'shattercaps', 'set our caps', 'share our caps',
                                    'sorry caps', 'still our gaps', 'shower caps', 'shutter caps',
                                    'shadowcaps')
    MINIMAL = ConcreteApexTerm('minimal', 'no more', 'I\'m in a moment', 'maybe more',
                               'minimal draw')
    SLOW = ConcreteApexTerm('slow', 'so', 'hello')
    QUICK = ConcreteApexTerm('quick', 'quit', 'pick')
    DRAWN = ConcreteApexTerm('drawn', 'drawing', 'John', 'drone', 'you\'re on', 'fully drawn',
                             'full draw')
    TURBOCHARGER = ConcreteApexTerm(
        'Turbocharger', 'dipper charger', 'gerber charger', 'to a charger', 'turbo charger',
        'temperature', 'timber charger', 'supercharger', 'derpacharger', 'double charger',
        'There we are Roger', 'terroir charger', 'to butch rogers', 'gibber-chatter')
    EVA_8 = ConcreteApexTerm(
        'EVA-8', 'either eat', 'e-rate', 'eve at 8', 'eva 8', 'ebay', 'you\'re right', 'you bait',
        'ev8', 'eva', 'you ate', 'evade', 'eva8', 'E-V-A-T-E', 'ebate', 'you wait', 'ewait',
        'evate', 'eat the eat', 'you may', 'if a', 'every', 'elley')
    FLATLINE = ConcreteApexTerm(
        'Flatline', 'fly line', 'da-von', 'well then', 'batman', 'that one', 'it\'s fine',
        'that line', 'it\'s not mine', 'flatlined'
        # Also sounds like "Prowler": 'bye-bye'
    )
    G7_SCOUT = ConcreteApexTerm(
        'G7 Scout', 'do 7th scout', 'd7 scout', 'houston scout', 'g7 let\'s go',
        'she\'s having a scalp', 'do 7scout', 'u7s go', 'do you see him on the scale',
        'u7scout', 'use 7th scout', 'he\'s from scout', 'G7 scalp', '7 scout', '7scout', 'do 7',
        'she\'s having a scowl', 'd7', 'you seven', 'This is XanathScout', 'G-Zone Scout',
        'let\'s go', 'ciao')
    HAVOC = ConcreteApexTerm(
        'HAVOC', 'have it', 'add it', 'have a', 'evoke', 'have a look', 'avok', 'havok'
         # Also sounds like "Hammerpoint": 'have a good one',
         )
    HEMLOCK = ConcreteApexTerm(
        'Hemlock', 'm-lok', 'and look', 'good luck', 'hemba', 'I\'m not', 'have a lot',
        'M.L.A.', 'M-LOT', 'mwah', 'ma')
    KRABER = ConcreteApexTerm(
        'Kraber', 'credit', 'KBIR', 'paper', 'kripper', 'grayer', 'Creepers', 'Taylor',
        'Creeper',
        'covered', 'Khyber', 'Kramer', 'Krabber',
        # Sounds like 30-30: 'thank you'
    )
    LONGBOW = ConcreteApexTerm(
        'Longbow', 'Wombo', 'I\'m well', 'Buh-bye', 'Bumbo', 'Number', 'Lambo', 'Lombo')
    L_STAR = ConcreteApexTerm(
        'L-STAR', 'It is done', 'I\'ll star', 'L star', 'that\'ll start', 'I\'ll start')
    MASTIFF = ConcreteApexTerm('Mastiff', 'Matthew', 'massive', 'bastard', 'next'
                               # Also sounds like "Nemesis": 'that\'s it',
                               )
    MOZAMBIQUE = ConcreteApexTerm(
        'Mozambique', 'mojambique', 'or jim beak', 'what would it be', 'what does it mean',
        'how\'s the beat', 'what does it beat', 'what does it be', 'well that\'s gonna be it',
        'that was in B', 'buzz-a-bee', 'that\'s a beat', 'was me', 'musts\'t be', 'let\'s meet',
        'wasn\'t me', 'was me', 'that\'s me', 'it wasn\'t me', 'must be', 'well it\'ll be',
        'that was a beat', 'listen to the beat', 'how\'s it be', 'BuzzMe', 'Flows in B',
        'more than me', 'was it me', 'it wasn\'t me', 'wasn\'t beat', 'Buzzard B',
        'Wasn\'t big',
        'BuzzBee', 'what does that mean', 'that was a bit', 'must be', 'with me', 'let\'s beat',
        'bows and beats', 'Nozambique', 'oh it\'s a beak', 'that was a meek', 'was it weak',
        'mosebeek', 'lithium beak', 'that was an Amiga', 'it was a Mieko', 'Lozenbeak',
        'Rosenbeek',
        'It doesn\'t make')
    HAMMERPOINT = ConcreteApexTerm(
        'Hammerpoint', 'hemmorhoid', 'end report', 'Ember Hoyt', 'never mind', 'nevermind',
        'error point', 'airpoint', 'camera point', 'here I\'ll play it', 'him right point',
        'fair point', 'him our point', 'hear my point', 'your own point', 'hammer point',
        'And we\'re playing', 'her point')
    NEMESIS = ConcreteApexTerm(
        'Nemesis', 'and this is', 'now what\'s this', 'Namaste', 'messes', 'nervousness', 'yes',
        'gracias', 'there it is', 'no messes', 'and that\'s this', 'he misses',
        'and that\'s just')
    P2020 = ConcreteApexTerm('P2020', 'be 2020', 'B-2020', 'P-220')
    PEACEKEEPER = ConcreteApexTerm('Peacekeeper', 'today', '2k', 'BK', 'P.K.')
    DISRUPTOR = ConcreteApexTerm('Disruptor', 'it\'s Raptor', 'the softer', 'stopping', 'disrupted')
    PROWLER = ConcreteApexTerm(
        'Prowler', 'power', 'browler', 'howdy', 'probably', 'brawler', 'powler', 'howler',
        'fowler', 'brother', 'totaler', 'teller', 'proudly')
    R301_CARBINE = ConcreteApexTerm(
        'R-301 Carbine', 'R301', 'R3-1', 'I 3-1', 'bye 301', 'after they weren\'t covering',
        'R3-1 covering', 'R3 when in carbine', 'R31 carbine', 'oh I see everyone caught mine',
        'I threw out a carbine', 'I should learn carbon', 'R3-1 cop mine',
        'Wash your own car by then', 'R3-A1 cop mine', 'I\'ll stay one comment',
        'R3-O1 covering',
        'thanks everyone for coming', 'I\'ll also go into copying', 'I throw in carbine',
        'R3-1 copying', 'I forgot to cut me in', 'I\'ve ruined the carpet',
        'I\'ll see you on Carbine', 'I should run to carbine', 'R31Carbine',
        'I have to uncover him', 'I should run Karmine', 'R3-1 coming',
        'I think we\'re on carbine',
        'I threw on carbine', 'R31 carbene', 'R31 covering', 'I threw in carbon',
        'That\'s a lot of carbon', 'I threw in a carbine', 'I\'ll show you when to come in',
        'I\'ll see you in comment', 'I\'ll see you in the car bye', 'I\'ll throw in a carbine',
        'R31cabine', 'I threw in one carbine', 'I actually want a carbine', 'R3-O-1 Copying',
        'I\'ll see you all in Carbine', 'R3-1 copy', 'R3-1 copy me', 'I\'ll see you one',
        'R3-A1',
        'R3-O-1', 'or should I go one', 'I\'ll see you at one', 'or a second one', 'R2-D2 won',
        'or I\'ll show you one', 'that\'s day one', 'I\'ll show you one', 'I\'ll take one',
        'I fear one', 'or 3-1', 'I say one', 'or is there one', 'I see one coming',
        'R3-1 to Carbine', 'R301 Combine')
    R99 = ConcreteApexTerm(
        'R-99', 'R99', '$5.99', 'or nine nine', 'or nine-to-nine', 'or ninety-nine',
        'I don\'t know', 'R9', 'all done', 'I had a dead eye', 'hard on your nine', 'hard 99',
        'all right any line', 'I don\'t need nine', 'irony 9', 'I already know',
        'I already need a 9', '$1.99', 'R-89', 'iron 9', 'oh I don\'t even know')
    REVVED = ConcreteApexTerm(
        'revved up', 'wrapped up', 'rev it up', 'ribbed up', 'revved it', 'rev\'d', 'revved',
        'R.I.P.', 'round')
    RE_45 = ConcreteApexTerm(
        'RE-45', 'RA-45', 'R45', 'RD-45', 'are we 45', 'RU45', 'are you 45', 'R8-45')
    SENTINEL = ConcreteApexTerm(
        'sentinel', 'what\'s that now', 'is that not', 'setting\' off', 'that\'s it now',
        'techno', 'is that no', 'said no', 'such an old', '7-0')
    AMPED = ConcreteApexTerm('amped', 'ant', 'it', 'end', 'yipped')
    SPITFIRE = ConcreteApexTerm(
        'Spitfire', 'step out of the car', 'is that her', 'it\'s a bit better', 'fitzpire',
        'skip her', 'zip fire', 'stay fire', 'set fire')
    TRIPLE_TAKE = ConcreteApexTerm(
        'Triple Take', 'triple-tick', 'triple T', 'chipotle', 'sure thing', 'chilti', 'Chanté',
        'triple-click', 'it\'s real thick')
    VOLT = ConcreteApexTerm('Volt', 'oh', 'bull', 'boop', 'what', 'well', 'vote')
    WINGMAN = ConcreteApexTerm('Wingman', 'we\'ll be back', 'wing then', 'wing men', 'wingmen')
    BOOSTED_LOADER = CombinedTerm(ConcreteApexTerm('boosted', 'who\'s dead', 'that\'s it'),
                                  ConcreteApexTerm('loader', 'loaded', 'love you', 'odor'))

    OPT_WITH = ConcreteApexTerm('with').opt()

    WEAPON_ARCHETYPE_TERMS: typing.Tuple[ApexTermBase] = (
        THIRTY_THIRTY_REPEATER.combine(SLOW),
        THIRTY_THIRTY_REPEATER,
        ALTERNATOR,
        ALTERNATOR.combine(OPT_WITH, DISRUPTOR),
        BOCEK,
        BOCEK.combine(DRAWN),
        BOCEK.combine(OPT_WITH, SHATTER_CAPS, MINIMAL),
        BOCEK.combine(OPT_WITH, SHATTER_CAPS, DRAWN),
        CAR,
        CHARGE_RIFLE,
        DEVOTION,
        DEVOTION.combine(CARE_PACKAGE),
        DEVOTION.combine(OPT_WITH, TURBOCHARGER),
        EVA_8,
        EVA_8.combine(CARE_PACKAGE),
        FLATLINE,
        G7_SCOUT,
        HAVOC,
        HAVOC.combine(OPT_WITH, TURBOCHARGER),
        HEMLOCK,
        KRABER,
        LONGBOW,
        L_STAR,
        MASTIFF,
        MOZAMBIQUE,
        MOZAMBIQUE.combine(OPT_WITH, HAMMERPOINT),
        NEMESIS,
        P2020,
        P2020.combine(OPT_WITH, HAMMERPOINT),
        PEACEKEEPER,
        PEACEKEEPER.combine(OPT_WITH, DISRUPTOR),
        PROWLER,
        PROWLER.combine(CARE_PACKAGE),
        R301_CARBINE,
        R99,
        RAMPAGE,
        RAMPAGE.combine(REVVED),
        RE_45,
        RE_45.combine(OPT_WITH, HAMMERPOINT),
        SENTINEL,
        SENTINEL.combine(AMPED),
        SPITFIRE,
        TRIPLE_TAKE,
        VOLT,
        WINGMAN,
        WINGMAN.combine(OPT_WITH, BOOSTED_LOADER)
    )
