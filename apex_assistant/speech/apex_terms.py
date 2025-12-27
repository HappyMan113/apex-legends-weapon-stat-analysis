from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from apex_assistant.checker import check_tuple, check_type
from apex_assistant.speech.term import IntTerm, OptTerm, RequiredTerm, Term, TermBase


def _create_level_terms(attachment_base_name: RequiredTerm,
                        alternate_none_term: RequiredTerm | None = None) -> \
        tuple[RequiredTerm, ...]:
    assert isinstance(attachment_base_name, RequiredTerm)
    none_term = WITHOUT + attachment_base_name
    if alternate_none_term is not None:
        assert isinstance(alternate_none_term, RequiredTerm)
        none_term = none_term | alternate_none_term
    all_levels_term: tuple[RequiredTerm, ...] = (
            (none_term,) +
            tuple((OPT_WITH_EXCL + ((level_term + attachment_base_name) |
                                    (attachment_base_name + level_term)))
                  for level_term in LEVEL_TERMS))
    for term in all_levels_term:
        assert isinstance(term, RequiredTerm)
    return all_levels_term


class SuffixedArchetypeType(StrEnum):
    HOPPED_UP = 'hopped-up'
    REVVED_UP = 'revved'
    SLOW = 'slow'
    AKIMBO = 'akimbo'
    MODDED = 'modded'


class Suffix:
    def __init__(self, *suffixes: TermBase | Tuple[TermBase, ...]):
        check_tuple(TermBase, suffixes=suffixes, allow_empty=False)
        self._suffix_types = tuple(_SUFFIX_TO_TYPE_DICT[suffix] for suffix in suffixes)
        self._suffixes = suffixes

    def __len__(self):
        return len(self._suffixes)

    def get_types(self) -> Tuple[SuffixedArchetypeType, ...]:
        return self._suffix_types

    def get_terms(self) -> Tuple[TermBase, ...]:
        return self._suffixes

    def has_term(self, term: TermBase) -> bool:
        if term not in _SUFFIX_TO_TYPE_DICT:
            raise ValueError(f'{term} is not a weapon suffix term.')
        return term in self._suffixes

    def __repr__(self):
        return f'Suffix{self._suffix_types}'


ZERO = IntTerm(0, 'zero')
ONE = IntTerm(1, 'one', 'a', 'the', 'in a')
TWO = IntTerm(2, 'two')
THREE = IntTerm(3, 'three')
FOUR = IntTerm(4, 'four')
EIGHT = IntTerm(8, 'eight')
NUMBER_TERMS = (ZERO,
                ONE,
                TWO,
                THREE,
                FOUR,
                IntTerm(5, 'five'),
                IntTerm(6, 'six'),
                IntTerm(7, 'seven'),
                EIGHT,
                IntTerm(9, 'nine'),
                IntTerm(10, 'ten'))
COMPARE = Term('compare')
_BEST = Term('best')
BEST = _BEST | _BEST + Term('the')
CREATE_SUMMARY_REPORT = Term('create summary report')
WEAPON: RequiredTerm = Term('weapon', 'gun')
WEAPONS_OPT = Term('weapons', 'guns').opt(include_in_output=True)
_SIDE = Term('side', 'secondary')
_ARM = Term('arm', 'weapon')
_ARMS = Term('arms', 'weapons')
_SIDEARM: RequiredTerm = Term('sidearm') | (_SIDE + _ARM)
_SIDEARMS: RequiredTerm = Term('sidearms') | (_SIDE + _ARMS)
LOADOUT: RequiredTerm = Term('loadout', 'load out')
LOADOUTS: RequiredTerm = Term('loadouts', 'load outs')
STOP = Term('stop')
CONFIGURE: RequiredTerm = Term('configure', 'set') + Term('default').opt()

TRUE = Term('true', 'on') | ONE
FALSE = Term('false', 'off') | ZERO

WITH = Term('with')
OPT_WITH_INCL: OptTerm = WITH.opt(include_in_output=True)
OPT_WITH_EXCL: OptTerm = WITH.opt(include_in_output=False)
AND = Term('and', 'an', 'any')
SWITCHING_TO = Term('switching to', 'and switching to', 'and then switching to')
SIDEARM_OPT_OF = _SIDEARM + Term('of').opt()
WITH_SIDEARM = AND | (AND | _SIDEARM) | ((WITH | SWITCHING_TO) + SIDEARM_OPT_OF) | SIDEARM_OPT_OF
MAIN = Term('main', 'primary')
_THE_SAME = Term('the').opt() + Term('same')
THE_SAME_SIDEARM: RequiredTerm = _THE_SAME + _SIDEARM
THE_SAME_MAIN_WEAPON: RequiredTerm = (_THE_SAME + (MAIN | WEAPON | (MAIN + WEAPON)))

OR = Term('or')

LEVEL = Term('level', 'lv.')
WHITE = Term('white')
BLUE = Term('blue')
PURPLE = Term('purple')
GOLDEN = Term('golden')
_LEVEL_0 = (LEVEL + ZERO)
LEVEL_1 = (LEVEL + ONE) | WHITE
LEVEL_2 = (LEVEL + TWO) | BLUE
LEVEL_3 = (LEVEL + THREE) | PURPLE
LEVEL_4 = (LEVEL + FOUR) | GOLDEN
LEVEL_3_OR_4 = LEVEL_3 | LEVEL_4 | LEVEL.append(THREE, OR, FOUR)
LEVEL_TERMS: tuple[RequiredTerm, ...] = (LEVEL_1, LEVEL_2, LEVEL_3_OR_4)
BASE = Term('base') | _LEVEL_0
WITHOUT = (Term('no', 'without', 'with no', 'without a', 'without any', 'without an') | BASE |
           (WITH + BASE))
NONE = Term('none', 'left out', 'excluded')

MAG: RequiredTerm = (Term('extended').opt() +
                     Term('sniper', 'light', 'energy', 'heavy').opt() +
                     Term('mag', 'magazine'))
ALL_MAG_TERMS = _create_level_terms(MAG, Term('nomag'))

STOCK = Term('stock', 'standard stock', 'sniper stock')
ALL_STOCK_TERMS = _create_level_terms(STOCK)

FIRE_MODE_SINGLE = Term('single', 'single fire')
CHARGED = Term('charged')

FULLY_KITTED: RequiredTerm = (Term('fully', 'full').opt() + Term('kitted') |
                              Term('fully-kitted'))

BOLT = Term('bolt')
SHOTGUN = Term('shotgun')
SHOTGUN_BOLT = SHOTGUN.opt() + BOLT
ALL_BOLT_TERMS = _create_level_terms(SHOTGUN_BOLT, Term('no-bolt'))

SMG = Term('SMG')
SMG_OPT = SMG.opt()

THIRTY_THIRTY_REPEATER = Term('30-30', '3030')
BOCEK = Term('Bocek', 'bow')
CAR = Term('car', 'C.A.R.', 'CAR SMG')
CARE_PACKAGE: RequiredTerm = Term('care package', 'supply drop')
CARE_PACKAGE_OPT: OptTerm = CARE_PACKAGE.opt()
ALTERNATOR = Term('Alternator')
CHARGE_RIFLE = Term('Charge Rifle')
FAR = Term('far', 'long distance')
RAMPAGE = Term('Rampage')
DEVOTION = Term('Devotion')
SHATTER_CAPS = Term('shatter caps', 'shattercaps')
MINIMAL_DRAW = Term('minimal draw', 'minimal')
SLOW = Term('slow')
QUICK_OPT: OptTerm = Term('quick').opt(include_in_output=True)
DRAWN = Term('full draw', 'drawn', 'fully drawn', 'max draw', 'maximum draw')
TURBOCHARGER = Term('Turbocharger', 'turbo charger')
TURBOCHARGED = Term('turbocharged', 'turbo charged')
GRAFFITI_MOD = Term('graffiti') + Term('mod').opt()
DOUBLE_TAP = Term('double tap', 'double-tap', 'doubletap')
_EVA = Term('eva')
EVA_8 = (Term('eva 8', 'eva', 'eva8') | (_EVA + SHOTGUN) | (_EVA + EIGHT))
FLATLINE = Term('Flatline', 'flat line')
G7_SCOUT = Term('G7 Scout', 'G7')
HAVOC = Term('havoc', 'havok')
HEMLOK = Term('Hemlok', 'hemlock')
KRABER = Term('Kraber')
LONGBOW = Term('Longbow', 'long bow')
L_STAR = Term('L-STAR', 'L star', 'LStar')
MASTIFF = Term('Mastiff')
MOZAMBIQUE = Term('Mozambique', 'mojambique')
HAMMERPOINT = Term('Hammerpoint', 'hammer point', 'hammer points', 'hammerpoints')
AKIMBO = Term('akimbo')
AKIMBO_OPT = AKIMBO.opt()
AKIMBO_WITH_HAMMERPOINT = AKIMBO.append_order_agnostic(HAMMERPOINT)
NEMESIS = Term('Nemesis')
P2020 = Term('P2020', 'P20')
PEACEKEEPER = Term('Peacekeeper', 'P.K.', 'PK')
DISRUPTOR = Term('Disruptor')
PROWLER = Term('Prowler')
_R = Term('R')
_R3 = Term('R3')
_R301 = Term('R-301', 'R 301', 'R301') | _R3 | (_R + THREE)
_CARBINE = Term('Carbine')
R301_CARBINE = _R301 + _CARBINE.opt()

R99 = Term('R 99', 'R-99', 'R99', 'R9', 'R-9') | (_R.opt() + Term('99'))
REVVED = Term('revved up', 'rev\'d', 'revved', 'revd', 'rev')
RE_45 = Term('RE-45', 'R45', 'RE 45', 'RE45', 'R. 45', 'R.A.45', 'R.E.45', 'R.E. 45', 'R.E.45')
SENTINEL = Term('sentinel', 'sentinelle')
AMPED = Term('amped')
SPITFIRE = Term('Spitfire')
TRIPLE_TAKE = Term('Triple Take', 'tripletake')
VOLT = Term('Volt') + SMG_OPT
WINGMAN = Term('Wingman', 'win man')
BOOSTED_LOADER = Term('boosted') + Term('loader')
SELECTFIRE_RECEIVER = ((Term('selectfire') | (Term('select') + Term('fire'))) +
                       Term('receiver').opt(include_in_output=False))
RELIC = Term('relic')
BEAM_SHOT = Term('beam shot')
SHEILA = Term('Sheila')
SKULLPIERCER = Term('skullpiercer', 'skull piercer')
PREFIRED = (Term('prefired', 'prefire', 'pre-fire', 'pre-fired') |
            (Term('pre') + Term('fired', 'fire')))


_SUFFIX_TO_TYPE_DICT: Mapping[TermBase, SuffixedArchetypeType] = MappingProxyType({
    BOOSTED_LOADER: SuffixedArchetypeType.HOPPED_UP,
    HAMMERPOINT: SuffixedArchetypeType.HOPPED_UP,
    DISRUPTOR: SuffixedArchetypeType.HOPPED_UP,
    SLOW: SuffixedArchetypeType.SLOW,
    MINIMAL_DRAW: SuffixedArchetypeType.SLOW,
    BEAM_SHOT: SuffixedArchetypeType.HOPPED_UP,
    FIRE_MODE_SINGLE: SuffixedArchetypeType.SLOW,
    CHARGED: SuffixedArchetypeType.REVVED_UP,
    REVVED: SuffixedArchetypeType.REVVED_UP,
    AMPED: SuffixedArchetypeType.REVVED_UP,
    AKIMBO: SuffixedArchetypeType.AKIMBO,
    SELECTFIRE_RECEIVER: SuffixedArchetypeType.MODDED,
    TURBOCHARGED: SuffixedArchetypeType.MODDED,
    GRAFFITI_MOD: SuffixedArchetypeType.MODDED,
    DOUBLE_TAP: SuffixedArchetypeType.MODDED,
    PREFIRED: SuffixedArchetypeType.SLOW
})

ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIXES_DICT: (
    MappingProxyType[Term, Optional[Tuple[Suffix, ...]]]
) = MappingProxyType({
    THIRTY_THIRTY_REPEATER: Suffix(SLOW),
    ALTERNATOR: Suffix(DOUBLE_TAP),
    BOCEK: (Suffix(MINIMAL_DRAW), Suffix(REVVED), Suffix(REVVED, MINIMAL_DRAW)),
    CAR.append(SMG_OPT).append(CARE_PACKAGE_OPT): None,
    CHARGE_RIFLE: Suffix(SELECTFIRE_RECEIVER),
    DEVOTION: Suffix(TURBOCHARGED),
    EVA_8: None,
    FLATLINE: None,
    G7_SCOUT: None,
    HAVOC: Suffix(TURBOCHARGED),
    HEMLOK: Suffix(FIRE_MODE_SINGLE),
    KRABER: None,
    LONGBOW: None,
    L_STAR: None,
    MASTIFF: Suffix(GRAFFITI_MOD),
    MOZAMBIQUE: (Suffix(AKIMBO, GRAFFITI_MOD), Suffix(AKIMBO), Suffix(GRAFFITI_MOD)),
    NEMESIS: Suffix(CHARGED),
    P2020.append(CARE_PACKAGE_OPT): Suffix(AKIMBO),
    PEACEKEEPER: None,
    PROWLER: Suffix(SELECTFIRE_RECEIVER),
    R301_CARBINE: None,
    R99: None,
    RAMPAGE: Suffix(REVVED),
    RE_45: Suffix(PREFIRED),
    SENTINEL: Suffix(AMPED),
    SPITFIRE: None,
    TRIPLE_TAKE: None,
    VOLT.append(SMG_OPT): Suffix(GRAFFITI_MOD),
    WINGMAN: Suffix(GRAFFITI_MOD),
    SHEILA: None
})
for key, value in ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIXES_DICT.items():
    check_type(TermBase, key=key)
    check_type((Suffix, tuple), optional=True, value=value)
