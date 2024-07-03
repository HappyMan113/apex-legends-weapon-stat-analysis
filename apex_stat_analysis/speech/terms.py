import abc
import re
from types import MappingProxyType
import typing

def strip_punctuation(text: str):
    return re.sub(r'[.,!?()\]\[{}]', '', text)


class Term:
    def __init__(self, words: str | typing.Iterable[str]):
        if isinstance(words, str):
            words = words.split(' ')
        original_words = tuple(filter(len, words))
        self.words = tuple([strip_punctuation(original_word).lower().replace('-', '')
                            for original_word in original_words])
        self.original_words = original_words

    def __repr__(self):
        return ' '.join(self.original_words)

    def get_words(self):
        return self.words

    def __hash__(self):
        return hash(self.words)

    def __eq__(self, other):
        return isinstance(other, Term) and self.words == other.words

    def __getitem__(self, sl):
        return Term(self.original_words[sl])

    def __len__(self):
        return len(self.words)


class ApexTermBase(abc.ABC):
    @abc.abstractmethod
    def get_max_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def get_min_variation_len(self) -> int:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    @abc.abstractmethod
    def _has_variation(self, said_term: Term) -> bool:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def has_variation(self, said_term: Term) -> bool:
        assert isinstance(said_term, Term)
        return self._has_variation(said_term)

    @abc.abstractmethod
    def has_variation_len(self, n) -> bool:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')

    def __add__(self, other):
        return self.combine(other)

    def combine(self, *next_terms: 'ApexTermBase'):
        return _CombinedTerm(self, *next_terms)

    def combine_opt_term(self, optional_term: 'ConcreteApexTerm'):
        return _CombinedTerm(self, optional_term.with_variations(''))

    def get_base_terms(self) -> typing.Generator['ConcreteApexTerm', None, None]:
        raise NotImplementedError(f'Must implement in {self.__class__.__name__}')


class ConcreteApexTerm(ApexTermBase):
    @staticmethod
    def _map_term(term: str | typing.Iterable[str] | Term) -> Term:
        return term if isinstance(term, Term) else Term(term)

    def __init__(self, known_term: str | list[str] | Term, *variations: str | list[str] | Term):
        known_term = self._map_term(known_term)
        all_variations: tuple[Term] = (known_term,) + tuple(map(self._map_term, variations))
        variations_dict: dict[int, set[Term]] = {}
        for variation in all_variations:
            n = len(variation)
            if n not in variations_dict:
                variations_n: set[Term] = set()
                variations_dict[n] = variations_n
            else:
                variations_n = variations_dict[n]
            variations_n.add(variation)
        self._variations_dict = MappingProxyType(variations_dict)
        self._min_variation_len = min(map(len, all_variations))
        self._max_variation_len = max(map(len, all_variations))
        self._known_term = known_term

    def get_max_variation_len(self) -> int:
        return self._max_variation_len

    def get_min_variation_len(self) -> int:
        return self._min_variation_len

    def _has_variation(self, said_term: Term) -> bool:
        n = len(said_term)
        variations_n = self._variations_dict.get(n, None)
        return variations_n is not None and said_term in variations_n

    def has_variation_len(self, n) -> bool:
        return n in self._variations_dict

    def get_base_terms(self) -> typing.Generator['ConcreteApexTerm', None, None]:
        if not self.has_variation_len(0):
            yield self

    def __len__(self):
        return self._known_term.__len__()

    def __repr__(self):
        return self._known_term.__repr__()

    def __str__(self):
        return self._known_term.__str__()

    def __hash__(self):
        return self._known_term.__hash__()

    def __eq__(self, other):
        return self._known_term.__eq__(other)

    def with_variations(self, *variations: str | list[str] | Term):
        new_variations: tuple[Term] = (
                tuple(term
                      for term_set in self._variations_dict.values()
                      for term in term_set) +
                tuple(map(self._map_term, variations)))
        return ConcreteApexTerm(*new_variations)


class _CombinedTerm(ApexTermBase):
    def __init__(self, *sub_terms: ApexTermBase):
        self.sub_terms = sub_terms
        self.max_variation_len = sum(term.get_max_variation_len() for term in self.sub_terms)
        self.min_variation_len = sum(term.get_min_variation_len() for term in self.sub_terms)

    def get_max_variation_len(self) -> int:
        return self.max_variation_len

    def get_min_variation_len(self) -> int:
        return self.min_variation_len

    def _has_variation(self, said_term: Term) -> bool:
        sub_terms = self.sub_terms
        if len(sub_terms) == 0:
            return False

        cur_word_idx = 0
        for sub_term in sub_terms[:-1]:
            max_num_words = min(len(said_term) - cur_word_idx, sub_term.get_max_variation_len())
            num_words: int | None = next(
                    filter(lambda nw: sub_term.has_variation(
                            said_term[cur_word_idx:cur_word_idx + nw]),
                           range(max_num_words, 0, -1)),
                    None)
            if num_words is None:
                return False

            cur_word_idx += num_words

        # The remaining number of words in the said term should match the number of words in one of
        # the variations of the last term; otherwise, it's not a match.
        last_sub_term = sub_terms[-1]
        return last_sub_term.has_variation(said_term[cur_word_idx:])

    def has_variation_len(self, n) -> bool:
        # Not totally accurate, but good enough for now.
        return self.max_variation_len >= n >= self.min_variation_len

    def __len__(self):
        return sum(map(len, self.sub_terms))

    def __repr__(self) -> str:
        return ' '.join(map(str, self.sub_terms))

    def get_base_terms(self) -> typing.Generator[ConcreteApexTerm, None, None]:
        for sub_term in self.sub_terms:
            for base_term in sub_term.get_base_terms():
                yield base_term


def extract_terms(terms: ApexTermBase | typing.Iterable[ApexTermBase]) -> tuple[ConcreteApexTerm]:
    if isinstance(terms, ApexTermBase):
        terms = (terms,)
    return tuple(base_term
                 for term in terms
                 for base_term in term.get_base_terms())


def terms_match(terms1: typing.Iterable[ApexTermBase], terms2: typing.Iterable[ApexTermBase]) -> \
        bool:
    base_terms1 = extract_terms(terms1)
    base_terms2 = extract_terms(terms2)
    return base_terms1 == base_terms2


class ApexTerms:
    COMPARE = ConcreteApexTerm('compare', 'which is better', 'which weapon is better',
                               'which one is better', 'what\'s better')
    BEST = ConcreteApexTerm('best', 'best weapons')
    NUMBERS = (ConcreteApexTerm('1', 'one'),
               ConcreteApexTerm('2', 'two'),
               ConcreteApexTerm('3', 'three'),
               ConcreteApexTerm('4', 'four'),
               ConcreteApexTerm('5', 'five'),
               ConcreteApexTerm('6', 'six'),
               ConcreteApexTerm('7', 'seven'),
               ConcreteApexTerm('8', 'eight'),
               ConcreteApexTerm('9', 'nine'),
               ConcreteApexTerm('10', 'ten'))
    COMMANDS = (COMPARE, BEST) + NUMBERS

    STOP = ConcreteApexTerm('stop')

    WITH_SIDEARM = ConcreteApexTerm('with sidearm', 'and sidearm', 'switching to sidearm',
                                    'and then switching to', 'with sidearm of', 'sidearm',
                                    'with secondary', 'with secondary weapon',
                                    'with secondary weapon of', 'and')

    NO_MAG_TERM = ConcreteApexTerm('no mag', 'no magazine')
    ALL_MAG_TERMS: typing.Tuple[ConcreteApexTerm] = (
        NO_MAG_TERM,
        ConcreteApexTerm('level 1 mag',
                         'level 1 magazine',
                         'white mag',
                         'white magazine'),
        ConcreteApexTerm('level 2 mag',
                         'level 2 magazine',
                         'blue mag',
                         'blue magazine'),
        ConcreteApexTerm('level 3 or 4 mag',
                         'level 3 magazine',
                         'purple mag',
                         'purple magazine',
                         'level 4 mag',
                         'level 4 magazine'),
    )

    NO_STOCK_TERM = ConcreteApexTerm('no stock', 'no standard stock', 'no sniper stock')
    ALL_STOCK_TERMS: typing.Tuple[ConcreteApexTerm] = (
        NO_STOCK_TERM,
        ConcreteApexTerm('level 1 stock',
                         'level 1 standard stock',
                         'level 1 stock stock',
                         'white stock'
                         'white standard stock'
                         'white sniper stock'),
        ConcreteApexTerm('level 2 stock',
                         'level 2 standard stock',
                         'level 2 stock stock',
                         'blue stock'
                         'blue standard stock'
                         'blue sniper stock'),
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
                         'golden sniper stock'),
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
                                       'white bolt',
                                       'level 1 shotgun bolt',
                                       'shotgun bolt level 1'),
                      ConcreteApexTerm('level 2 bolt',
                                       'blue bolt',
                                       'blue shotgun bolt'
                                       'level 2 shotgun bolt',
                                       'shotgun bolt level 2'),
                      ConcreteApexTerm('level 3 or 4 bolt',
                                       'level 3 bolt',
                                       'level 3 shotgun bolt'
                                       'purple bolt',
                                       'purple shotgun bolt',
                                       'level 4 bolt',
                                       'level 4 shotgun bolt',
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
            'EVA-8', 'either eat', 'e-rate', 'eve at 8', 'eva 8', 'ebay', 'you\'re right',
            'you bait',
            'ev8', 'eva', 'you ate', 'evade', 'eva8', 'E-V-A-T-E', 'ebate', 'you wait', 'ewait',
            'evate', 'eat the eat', 'you may', 'if a', 'every', 'elley')
    FLATLINE = ConcreteApexTerm(
            'Flatline', 'fly line', 'da-von', 'well then', 'batman', 'that one', 'it\'s fine',
            'that line', 'it\'s not mine',
            # Also sounds like "Prowler": 'bye-bye'
    )
    G7_SCOUT = ConcreteApexTerm(
            'G7 Scout', 'do 7th scout', 'd7 scout', 'houston scout', 'g7 let\'s go',
            'she\'s having a scalp', 'do 7scout', 'u7s go', 'do you see him on the scale',
            'u7scout', 'use 7th scout', 'he\'s from scout', 'G7 scalp', '7 scout', '7scout', 'do 7',
            'she\'s having a scowl', 'd7', 'you seven', 'This is XanathScout', 'G-Zone Scout',
            'let\'s go', 'ciao')
    HAVOC = ConcreteApexTerm('HAVOC', 'have it', 'add it', 'have a', 'evoke', 'have a look', 'avok',
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
    VOLT = ConcreteApexTerm('Volt', '4', 'oh', 'bull', 'boop', 'what', 'well', 'vote')
    WINGMAN = ConcreteApexTerm('Wingman', 'we\'ll be back', 'wing then', 'wing men', 'wingmen')
    BOOSTED_LOADER = _CombinedTerm(ConcreteApexTerm('boosted', 'who\'s dead', 'that\'s it'),
                                   ConcreteApexTerm('loader', 'loaded', 'love you', 'odor'))
    SMG = ConcreteApexTerm('SMG')
    SHOTGUN = ConcreteApexTerm('shotgun')
    RIFLE = ConcreteApexTerm('rifle')

    WITH = ConcreteApexTerm('with')

    WEAPON_ARCHETYPE_TERMS: typing.Tuple[ApexTermBase] = (
        THIRTY_THIRTY_REPEATER.combine(SLOW),
        THIRTY_THIRTY_REPEATER.combine_opt_term(QUICK),
        ALTERNATOR,
        ALTERNATOR.combine_opt_term(WITH).combine(DISRUPTOR),
        BOCEK.combine_opt_term(MINIMAL),
        BOCEK.combine(DRAWN),
        BOCEK.combine_opt_term(WITH).combine(SHATTER_CAPS, MINIMAL),
        BOCEK.combine_opt_term(WITH).combine(SHATTER_CAPS, DRAWN),
        CAR.combine_opt_term(SMG),
        CHARGE_RIFLE.combine_opt_term(FAR),
        DEVOTION,
        DEVOTION.combine(CARE_PACKAGE),
        DEVOTION.combine_opt_term(WITH).combine(TURBOCHARGER),
        EVA_8.combine_opt_term(SHOTGUN),
        EVA_8.combine_opt_term(SHOTGUN).combine(CARE_PACKAGE),
        FLATLINE,
        G7_SCOUT,
        HAVOC.combine_opt_term(RIFLE),
        HAVOC.combine_opt_term(RIFLE).combine_opt_term(WITH).combine(TURBOCHARGER),
        HEMLOCK,
        KRABER,
        LONGBOW,
        L_STAR,
        MASTIFF.combine_opt_term(SHOTGUN),
        MOZAMBIQUE.combine_opt_term(SHOTGUN),
        MOZAMBIQUE.combine_opt_term(SHOTGUN).combine_opt_term(WITH).combine(HAMMERPOINT),
        NEMESIS,
        P2020,
        P2020.combine_opt_term(WITH).combine(HAMMERPOINT),
        PEACEKEEPER.combine_opt_term(SHOTGUN),
        PEACEKEEPER.combine_opt_term(SHOTGUN).combine_opt_term(WITH).combine(DISRUPTOR),
        PROWLER,
        PROWLER.combine(CARE_PACKAGE),
        R301_CARBINE,
        R99,
        RAMPAGE,
        RAMPAGE.combine(REVVED),
        RE_45,
        RE_45.combine_opt_term(WITH).combine(HAMMERPOINT),
        SENTINEL,
        SENTINEL.combine(AMPED),
        SPITFIRE,
        TRIPLE_TAKE,
        VOLT.combine_opt_term(SMG),
        WINGMAN,
        WINGMAN.combine_opt_term(WITH).combine(BOOSTED_LOADER)
    )

    ALL_TERMS: typing.Tuple[ConcreteApexTerm] = (COMMANDS +
                                                 WEAPON_ARCHETYPE_TERMS +
                                                 ALL_STOCK_TERMS +
                                                 ALL_MAG_TERMS +
                                                 ALL_BOLT_TERMS +
                                                 (WITH_RELOAD_TERM, NO_RELOAD_TERM, STOP))
