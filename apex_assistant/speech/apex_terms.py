from types import MappingProxyType
from typing import Optional, TypeAlias, Union

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


ZERO = IntTerm(0, 'zero', 'oh', 'you', 'out', 'own', 'we\'re', 'go', 'I go', 'you at')
ONE = IntTerm(1, 'one', 'won', 'a', 'on', 'in', 'run', 'when in', 'learn', 'the', 'in a', 'when',
              'of', 'into')
TWO = IntTerm(2, 'two', 'too', 'to')
THREE = IntTerm(3, 'three', 'see', 'think', 'say', 'fear', 'throw', 'threw', 'should', 'show',
                'a lot', 'stay', 'day', 'take')
FOUR = IntTerm(4, 'four')
EIGHT = IntTerm(8, 'eight', 'eat', 'rate', 'right', 'bait', 'ate', 'wait', 'eat', 'may')
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
COMPARE = Term('compare', 'which is better', 'which weapon is better', 'which one is better',
               'what\'s better', 'air', 'bear', 'compared')
_BEST = Term('best', 'that\'s', 'this', 'this is', 'test', 'miss')
BEST = _BEST | _BEST + Term('the')
CREATE_SUMMARY_REPORT = Term('create summary report')
WEAPON: RequiredTerm = Term('weapon', 'gun')
WEAPONS_OPT = Term('weapons', 'guns').opt(include_in_speech=True)
_SIDE = Term('side', 'secondary', 'starred')
_ARM = Term('arm', 'weapon')
_ARMS = Term('arms', 'weapons')
SIDEARM: RequiredTerm = Term('sidearm', 'sodarm', 'sadarm', 'sad arm') | (_SIDE + _ARM)
SIDEARMS: RequiredTerm = Term('sidearms', 'sodarms', 'sadarms', 'sad arms') | (_SIDE + _ARMS)
LOADOUT: RequiredTerm = Term('loadout', 'load out', 'load up', 'ludo', 'loan out')
LOADOUTS: RequiredTerm = Term('loadouts', 'load outs', 'loanouts', 'loan outs')
STOP = Term('stop')
CONFIGURE: RequiredTerm = Term('configure', 'set') + Term('default').opt()

TRUE = Term('true', 'on') | ONE
FALSE = Term('false', 'off') | ZERO

WITH = Term('with')
OPT_WITH_INCL: OptTerm = WITH.opt(include_in_speech=True)
OPT_WITH_EXCL: OptTerm = WITH.opt(include_in_speech=False)
AND = Term('and', 'an', 'any')
SWITCHING_TO = Term('switching to', 'and switching to', 'and then switching to')
SIDEARM_OPT_OF = SIDEARM + Term('of').opt()
WITH_SIDEARM = AND | (AND | SIDEARM) | ((WITH | SWITCHING_TO) + SIDEARM_OPT_OF) | SIDEARM_OPT_OF
MAIN = Term('main', 'may', 'name', 'primary')
_THE_SAME = Term('the').opt() + Term('same')
THE_SAME_SIDEARM: RequiredTerm = _THE_SAME + SIDEARM
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
BASE = Term('base', 'bass', 'face') | _LEVEL_0
WITHOUT = (Term('no', 'without', 'with no', 'without a', 'without any', 'without an') | BASE |
           (WITH + BASE))
NONE = Term('none', 'nun', 'left out', 'excluded')

MAG: RequiredTerm = (Term('extended').opt() +
                     Term('sniper', 'light', 'energy', 'heavy').opt() +
                     Term('mag', 'magazine'))
ALL_MAG_TERMS = _create_level_terms(MAG, Term('nomag', 'nomad'))

STOCK = Term('stock', 'standard stock', 'sniper stock')
ALL_STOCK_TERMS = _create_level_terms(STOCK)

SINGLE_SHOT = Term('single shot', 'single-shot')
SKIPPED = Term('skipped')
FIRE_MODE_SINGLE = Term('single', 'single fire')
CHARGED = Term('charged')

FULLY_KITTED: RequiredTerm = (Term('fully', 'full').opt() + Term('kitted', 'kidded') |
                              Term('fully-kitted'))

BOLT = Term('bolt')
SHOTGUN = Term('shotgun')
SHOTGUN_BOLT = SHOTGUN.opt() + BOLT
ALL_BOLT_TERMS = _create_level_terms(SHOTGUN_BOLT, Term('no-bolt'))

SMG = Term('SMG')
SMG_OPT = SMG.opt()

THIRTY_THIRTY_REPEATER = \
    Term('30-30', '33', 'there are three', '30 seconds ready', 'very very', '3030')
_BOW = Term('bow', 'both', 'bo', 'bill')
_CHECK = Term('check', 'checks')
BOCEK = Term('bowcheck', 'Bocek', 'Bochek', 'bocheck', 'bo-check', 'bochek\'s') | (_BOW + _CHECK)
CAR = Term('car', 'C.A.R.', 'TARG', 'cut', 'tar', 'sorry', 'tower', 'tart', 'CAR-SMG')
CARE_PACKAGE_OPT: OptTerm = Term('care package', 'supply drop').opt()
ALTERNATOR = Term('Alternator', 'I don\'t need her', 'I\'ll do neither')
CHARGE_RIFLE = Term('Charge Rifle', 'charger full', 'charged rifle', 'Ciao Dreffel', 'charger')
FAR = Term('far', 'bar', 'bark')
RAMPAGE = Term('Rampage', 'my page', 'webpage')
DEVOTION = Term('Devotion', 'version', 'the ocean', 'lotion', 'the motion', 'motion', 'ocean')
SHATTER_CAPS = Term('shatter caps', 'shattercaps', 'set our caps', 'share our caps', 'sorry caps',
                    'still our gaps', 'shower caps', 'shutter caps', 'shadowcaps')
MINIMAL_DRAW = Term('minimal draw', 'minimal', 'no more', 'I\'m in a moment', 'maybe more')
SLOW = Term('slow', 'so', 'hello')
QUICK_OPT: OptTerm = Term('quick', 'quit', 'pick').opt(include_in_speech=True)
DRAWN = Term('full draw', 'drawn', 'drawing', 'John', 'drone', 'you\'re on', 'fully drawn',
             'max draw', 'maximum draw')
TURBOCHARGER = Term(
    'Turbocharger', 'dipper charger', 'gerber charger', 'to a charger', 'turbo charger',
    'temperature', 'timber charger', 'supercharger', 'derpacharger', 'double charger',
    'There we are Roger', 'terroir charger', 'to butch rogers', 'gibber-chatter')
_EVA = Term('eva', 'you', 'eat the', 'if a', 'either', 'ebay', 'you\'re', 'eve at', 'eve',
            'of the')
EVA_8 = (Term('eva 8', 'eva', 'e-rate', 'ebay', 'ev8', 'eva', 'evade', 'eva8', 'E-V-A-T-E', 'ebate',
              'ewait', 'evate', 'if a', 'every', 'elley', 'EVEE') |
         (_EVA + SHOTGUN) |
         (_EVA + EIGHT))
FLATLINE = Term(
    'Flatline', 'flat line', 'fly line', 'da-von', 'well then', 'batman', 'that one', 'it\'s fine',
    'that line', 'it\'s not mine', 'flatlined', 'plant line', 'Flatlock'
    # Also sounds like "Prowler": 'bye-bye'
)
G7_SCOUT = Term(
    'G7 Scout', 'do 7th scout', 'd7 scout', 'houston scout', 'g7 let\'s go',
    'she\'s having a scalp', 'do 7scout', 'u7s go', 'do you see him on the scale',
    'u7scout', 'use 7th scout', 'he\'s from scout', 'G7 scalp', '7 scout', '7scout', 'do 7',
    'she\'s having a scowl', 'd7', 'you seven', 'This is XanathScout', 'G-Zone Scout',
    'let\'s go', 'ciao')
HAVOC = Term(
    'havoc', 'have it', 'add it', 'have a', 'evoke', 'have a look', 'avok', 'havok', 'HAPIC', 'epic'
                                                                                              'Hathic'
    # Also sounds like "Hammerpoint": 'have a good one',
)
HEMLOCK = Term('Hemlock', 'm-lok', 'and look', 'good luck', 'hemba', 'I\'m not', 'have a lot',
               'M.L.A.', 'M-LOT', 'mwah', 'ma')
KRABER = Term(
    'Kraber', 'credit', 'KBIR', 'paper', 'kripper', 'grayer', 'Creepers', 'Taylor', 'Creeper',
    'covered', 'Khyber', 'Kramer', 'Krabber', 'Craber', 'Craver', 'Krabour', 'Graber', 'Krabber\'s',
    'Graberr\'s', 'Kraber\'s'
    # Sounds like 30-30: 'thank you'
)
LONGBOW = Term('Longbow', 'Wombo', 'I\'m well', 'Buh-bye', 'Bumbo', 'Number', 'Lambo', 'Lombo')
L_STAR = Term('L-STAR', 'It is done', 'I\'ll star', 'L star', 'that\'ll start', 'I\'ll start')
MASTIFF = Term('Mastiff', 'Matthew', 'massive', 'bastard', 'next'
               # Also sounds like "Nemesis": 'that\'s it',
               )
MOZAMBIQUE = Term(
    'Mozambique', 'mojambique', 'or jim beak', 'what would it be', 'what does it mean',
    'how\'s the beat', 'what does it beat', 'what does it be', 'well that\'s gonna be it',
    'that was in B', 'buzz-a-bee', 'that\'s a beat', 'was me', 'musts\'t be', 'let\'s meet',
    'wasn\'t me', 'was me', 'that\'s me', 'it wasn\'t me', 'must be', 'well it\'ll be',
    'that was a beat', 'listen to the beat', 'how\'s it be', 'BuzzMe', 'Flows in B', 'more than me',
    'was it me', 'it wasn\'t me', 'wasn\'t beat', 'Buzzard B', 'Wasn\'t big', 'BuzzBee',
    'what does that mean', 'that was a bit', 'must be', 'with me', 'let\'s beat', 'bows and beats',
    'Nozambique', 'oh it\'s a beak', 'that was a meek', 'was it weak', 'mosebeek', 'lithium beak',
    'that was an Amiga', 'it was a Mieko', 'Lozenbeak', 'Rosenbeek', 'It doesn\'t make')
HAMMERPOINT = Term(
    'Hammerpoint', 'hemmorhoid', 'end report', 'Ember Hoyt', 'never mind', 'nevermind',
    'error point', 'airpoint', 'camera point', 'here I\'ll play it', 'him right point',
    'fair point', 'him our point', 'hear my point', 'your own point', 'hammer point',
    'hammer points', 'And we\'re playing', 'her point')
NEMESIS = Term(
    'Nemesis', 'and this is', 'now what\'s this', 'Namaste', 'messes', 'nervousness', 'yes',
    'gracias', 'there it is', 'no messes', 'and that\'s this', 'he misses', 'and that\'s just')
P2020 = Term('P2020', 'be 2020', 'B-2020', 'P-220', 'P20')
PEACEKEEPER = Term('Peacekeeper', 'today', '2k', 'BK', 'P.K.', 'PK', 'piecekeeper',
                   'Peacekeeper\'s', 'Casekeeper')
DISRUPTOR = Term('Disruptor', 'it\'s Raptor', 'the softer', 'stopping', 'disrupted')
PROWLER = Term(
    'Prowler', 'power', 'browler', 'howdy', 'probably', 'brawler', 'powler', 'howler', 'fowler',
    'brother', 'totaler', 'teller', 'proudly', 'prouler', 'plowler', 'Sprawler', 'Proller',
    'parlor')
_R = Term('I\'ll', 'bye', 'or', 'I', 'oh I', 'wash', 'or I\'ll', 'that\'s', 'I\'m')
_R3 = Term('R3', 'R2-D2')
_THREE_O_ALT = Term('ruined', 'a second', 'is there', 'see you all', 'forgot')
_O_ONE_ALT = Term('everyone')
_R301 = (Term('R-301', 'after they weren\'t', 'wash your own', 'R3-A1', 'R3-O1',
              'thanks everyone for', 'I\'ll also go into', 'I actually want', 'R3-O-1', 'R3-1') |
         Term('R31') |
         _R.append(THREE.append_int(ONE) | THREE.append_int(ZERO, ONE) |
                   THREE.append(_O_ONE_ALT) | _THREE_O_ALT.append(ONE)) |
         _R3.append(ONE, ZERO.append_int(ONE)) |
         _R3)
_CARBINE = Term('to').opt() + Term(
    'Carbine', 'covering', 'caught mine', 'carbon', 'cop mine', 'car by then', 'comment', 'coming',
    'copying', 'cut me in', 'carpet', 'Karmine', 'carbene', 'come in', 'car bye', 'copy', 'copy me',
    'combine')
R301_CARBINE = (_R301 + _CARBINE.opt()) | Term('R31Carbine', 'R31cabine', 'I have to uncover him')

R99 = \
    (Term('R 99', 'R-99', 'R99', '$5.99', 'or nine nine', 'or nine-to-nine', 'or ninety-nine',
          'I don\'t know', 'R9', 'all done', 'I had a dead eye', 'hard on your nine', 'hard 99',
          'all right any line', 'I don\'t need nine', 'irony 9', 'I already know',
          'I already need a 9', '$1.99', 'R-89', 'iron 9', 'oh I don\'t even know', 'R-9', 'are 99',
          'I\'m 99', 'on and none', 'Auto-Nine', 'on a new 9', 'on 8 to 9', 'on 9 to 9',
          'oddity nine', '099', 'Ardney to 9', '90 to 9', 'R-909', 'R29', 'alrighty nine',
          'arlington nine', 'Ardena 9') |
     (_R.opt() + Term('99', '89')))
REVVED = Term('revved up', 'wrapped up', 'rev it up', 'ribbed up', 'revved it', 'rev\'d', 'revved',
              'R.I.P.', 'round')
RE_45 = Term('are e forty five', 'RE-45', 'RA-45', 'R45', 'RD-45', 'are we 45', 'RU45',
             'are you 45', 'R8-45', 'R445', 'RE 45', 'RE45', 'R. 45', 'R.A.45', 'R.E.45',
             'R.A. 45', 'R.E. 45', 'R.E.45', 'RIA-45', 'r a forty five', 'R435')
SENTINEL = Term('sentinel', 'what\'s that now', 'is that not', 'setting\' off', 'that\'s it now',
                'techno', 'is that no', 'said no', 'such an old', '7-0')
AMPED = Term('amped', 'ant', 'it', 'end', 'yipped')
SPITFIRE = Term('Spitfire', 'step out of the car', 'is that her', 'it\'s a bit better', 'fitzpire',
                'skip her', 'zip fire', 'stay fire', 'set fire')
TRIPLE_TAKE = Term('Triple Take', 'triple-tick', 'triple T', 'chipotle', 'sure thing', 'chilti',
                   'Chanté', 'triple-click', 'it\'s real thick')
VOLT = ((Term('Volt', 'oh', 'bull', 'boop', 'what', 'well', 'vote', 'voltz', 'volts', 'vault') +
         SMG_OPT) |
        (BOLT + SMG))
WINGMAN = Term('Wingman', 'we\'ll be back', 'wing then', 'wing men', 'wingmen')
BOOSTED_LOADER = (Term('boosted', 'who\'s dead', 'that\'s it') +
                  Term('loader', 'loaded', 'love you', 'odor'))
SHEILA = Term('Sheila', 'CELA', 'Sila')

_T: TypeAlias = MappingProxyType[RequiredTerm, Union[Optional[TermBase], Optional[TermBase]]]
ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIX_DICT: _T = MappingProxyType({
    THIRTY_THIRTY_REPEATER: SLOW,
    ALTERNATOR: DISRUPTOR,
    # Want to make sure that "Bocek", "Devotion", and "EVA-8" resolve to weapons till they switch
    # back to not having shatter caps.
    # BOCEK: (MINIMAL_DRAW.opt(),
    #         DRAWN,
    #         SHATTER_CAPS + OPT_WITH_INCL + MINIMAL_DRAW,
    #         SHATTER_CAPS + DRAWN),
    BOCEK.append(OPT_WITH_EXCL, SHATTER_CAPS.opt()): MINIMAL_DRAW,
    CAR.append(SMG_OPT): None,
    CHARGE_RIFLE: None,
    # Devotion is in the care package right now.
    # DEVOTION: TURBOCHARGER,
    DEVOTION.append_order_agnostic(CARE_PACKAGE_OPT): None,
    # EVA_8: None,
    EVA_8.append_order_agnostic(CARE_PACKAGE_OPT): None,
    FLATLINE: None,
    G7_SCOUT: None,
    HAVOC: TURBOCHARGER,
    HEMLOCK: FIRE_MODE_SINGLE,
    KRABER: None,
    LONGBOW: None,
    L_STAR: None,
    MASTIFF: None,
    MOZAMBIQUE: HAMMERPOINT,
    NEMESIS: CHARGED,
    P2020: HAMMERPOINT,
    PEACEKEEPER: DISRUPTOR,
    PROWLER: None,
    R301_CARBINE: None,
    R99: None,
    RAMPAGE: REVVED,
    RE_45: HAMMERPOINT,
    SENTINEL: AMPED,
    SPITFIRE: None,
    TRIPLE_TAKE: None,
    VOLT.append(SMG_OPT): None,
    WINGMAN: BOOSTED_LOADER,
    SHEILA: None,
})
