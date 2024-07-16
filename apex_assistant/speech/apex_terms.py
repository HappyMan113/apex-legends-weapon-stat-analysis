from typing import Tuple

from apex_assistant.speech.term import IntTerm, OptTerm, RequiredTerm, Term


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
              'the', 'of', 'into')
TWO = IntTerm(2, 'two')
THREE = IntTerm(3, 'three', 'see', 'think', 'say', 'fear', 'throw', 'threw', 'should', 'show',
                'a lot', 'stay', 'day', 'take')
FOUR = IntTerm(4, 'four')
NUMBER_TERMS = (ONE,
                TWO,
                THREE,
                FOUR,
                IntTerm(5, 'five'),
                IntTerm(6, 'six'),
                IntTerm(7, 'seven'),
                IntTerm(8, 'eight'),
                IntTerm(9, 'nine'),
                IntTerm(10, 'ten'))
COMPARE = Term('compare', 'which is better', 'which weapon is better', 'which one is better',
               'what\'s better', 'air')
BEST = Term('best') + Term('weapons').opt()
STOP = Term('stop')
CONFIGURE: RequiredTerm = Term('configure', 'configure default', 'set default')

TRUE = Term('true', 'on') | ONE
FALSE = Term('false', 'off') | ZERO

WITH = Term('with')
OPT_WITH_INCL: OptTerm = WITH.opt(include_in_speech=True)
OPT_WITH_EXCL: OptTerm = WITH.opt(include_in_speech=False)
SWITCHING_TO = WITH | Term('and', 'switching to', 'and switching to', 'and then switching to')
SIDEARM = Term('sidearm', 'sidearm of', 'secondary weapon', 'secondary weapon of')
SWITCHING_TO_SIDEARM = (SWITCHING_TO + SIDEARM) | SIDEARM

OR = Term('or')

LEVEL = Term('level', 'lv.')
WHITE = Term('white')
BLUE = Term('blue')
PURPLE = Term('purple')
GOLDEN = Term('golden')
LEVEL_1 = (LEVEL + ONE) | WHITE
LEVEL_2 = (LEVEL + TWO) | BLUE
LEVEL_3 = (LEVEL + THREE) | PURPLE
LEVEL_4 = (LEVEL + FOUR) | GOLDEN
LEVEL_3_OR_4 = LEVEL_3 | LEVEL_4 | LEVEL.combine(THREE, OR, FOUR)
LEVEL_TERMS: tuple[RequiredTerm, ...] = (LEVEL_1, LEVEL_2, LEVEL_3_OR_4)
BASE = Term('base')
WITHOUT = (Term('no', 'without', 'with no', 'without a', 'without any', 'without an') | BASE |
           (WITH + BASE))
NONE = Term('none', 'nun', 'left out', 'excluded')

MAG: RequiredTerm = (Term('extended').opt() +
                     Term('sniper', 'light', 'energy', 'heavy').opt() +
                     Term('mag', 'magazine'))
ALL_MAG_TERMS = _create_level_terms(MAG, Term('nomag', 'nomad'))

STOCK = Term('stock', 'standard stock', 'sniper stock')
ALL_STOCK_TERMS = _create_level_terms(STOCK)

RELOAD = Term('reloads', 'reload', 'reloading')
WITHOUT_RELOAD = WITHOUT + RELOAD
WITH_RELOAD: RequiredTerm = (WITH + RELOAD) | RELOAD
WITH_RELOAD_OPT: OptTerm = WITH_RELOAD.opt()

BOLT = Term('bolt')
SHOTGUN_BOLT = Term('shotgun').opt() + BOLT
ALL_BOLT_TERMS = _create_level_terms(SHOTGUN_BOLT, Term('no-bolt'))

SMG = Term('SMG')
SMG_OPT = SMG.opt()

THIRTY_THIRTY_REPEATER = \
    Term('30-30', '33', 'there are three', '30 seconds ready', 'very very', '3030')
BOCEK = Term('Bocek', 'bow check', 'Bochek')
CAR = Term('C.A.R.', 'car', 'TARG', 'cut', 'tar', 'sorry', 'tower', 'tart', 'CAR-SMG')
CARE_PACKAGE = Term('care package')
ALTERNATOR = Term('Alternator', 'I don\'t need her', 'I\'ll do neither')
CHARGE_RIFLE = Term('Charge Rifle', 'charger full', 'charged rifle', 'Ciao Dreffel', 'charger')
FAR = Term('far', 'bar', 'bark')
RAMPAGE = Term('Rampage', 'my page', 'webpage')
DEVOTION = Term('Devotion', 'version', 'the ocean', 'lotion', 'the motion', 'motion', 'ocean')
SHATTER_CAPS = Term('shatter caps', 'shattercaps', 'set our caps', 'share our caps', 'sorry caps',
                    'still our gaps', 'shower caps', 'shutter caps', 'shadowcaps')
MINIMAL_DRAW = Term('minimal draw', 'minimal', 'no more', 'I\'m in a moment', 'maybe more')
SLOW = Term('slow', 'so', 'hello')
QUICK_OPT = Term('quick', 'quit', 'pick').opt(include_in_speech=True)
DRAWN = Term('full draw', 'drawn', 'drawing', 'John', 'drone', 'you\'re on', 'fully drawn')
TURBOCHARGER = Term(
    'Turbocharger', 'dipper charger', 'gerber charger', 'to a charger', 'turbo charger',
    'temperature', 'timber charger', 'supercharger', 'derpacharger', 'double charger',
    'There we are Roger', 'terroir charger', 'to butch rogers', 'gibber-chatter')
EVA_8 = Term(
    'EVA-8', 'either eat', 'e-rate', 'eve at 8', 'eva 8', 'ebay', 'you\'re right', 'you bait',
    'ev8', 'eva', 'you ate', 'evade', 'eva8', 'E-V-A-T-E', 'ebate', 'you wait', 'ewait', 'evate',
    'eat the eat', 'you may', 'if a', 'every', 'elley')
FLATLINE = Term(
    'Flatline', 'fly line', 'da-von', 'well then', 'batman', 'that one', 'it\'s fine',
    'that line', 'it\'s not mine', 'flatlined'
    # Also sounds like "Prowler": 'bye-bye'
)
G7_SCOUT = Term(
    'G7 Scout', 'do 7th scout', 'd7 scout', 'houston scout', 'g7 let\'s go',
    'she\'s having a scalp', 'do 7scout', 'u7s go', 'do you see him on the scale',
    'u7scout', 'use 7th scout', 'he\'s from scout', 'G7 scalp', '7 scout', '7scout', 'do 7',
    'she\'s having a scowl', 'd7', 'you seven', 'This is XanathScout', 'G-Zone Scout',
    'let\'s go', 'ciao')
HAVOC = Term(
    'HAVOC', 'have it', 'add it', 'have a', 'evoke', 'have a look', 'avok', 'havok', 'HAPIC'
    # Also sounds like "Hammerpoint": 'have a good one',
)
HEMLOCK = Term('Hemlock', 'm-lok', 'and look', 'good luck', 'hemba', 'I\'m not', 'have a lot',
               'M.L.A.', 'M-LOT', 'mwah', 'ma')
KRABER = Term(
    'Kraber', 'credit', 'KBIR', 'paper', 'kripper', 'grayer', 'Creepers', 'Taylor', 'Creeper',
    'covered', 'Khyber', 'Kramer', 'Krabber',
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
    'And we\'re playing', 'her point')
NEMESIS = Term(
    'Nemesis', 'and this is', 'now what\'s this', 'Namaste', 'messes', 'nervousness', 'yes',
    'gracias', 'there it is', 'no messes', 'and that\'s this', 'he misses', 'and that\'s just')
P2020 = Term('P2020', 'be 2020', 'B-2020', 'P-220')
PEACEKEEPER = Term('Peacekeeper', 'today', '2k', 'BK', 'P.K.')
DISRUPTOR = Term('Disruptor', 'it\'s Raptor', 'the softer', 'stopping', 'disrupted')
PROWLER = Term(
    'Prowler', 'power', 'browler', 'howdy', 'probably', 'brawler', 'powler', 'howler', 'fowler',
    'brother', 'totaler', 'teller', 'proudly')
R = Term('I\'ll', 'bye', 'or', 'I', 'oh I', 'wash', 'or I\'ll', 'that\'s')
R3 = Term('R3', 'R2-D2')
THREE_O_ALT = Term('ruined', 'a second', 'is there', 'see you all', 'forgot')
O_ONE_ALT = Term('everyone')
R301 = (Term('R-301', 'after they weren\'t', 'wash your own', 'R3-A1', 'R3-O1',
             'thanks everyone for', 'I\'ll also go into', 'I actually want', 'R3-O-1', 'R3-1') |
        Term('R31') |
        R.combine(THREE.combine_int(ONE) | THREE.combine_int(ZERO, ONE) |
                  THREE.combine(O_ONE_ALT) | THREE_O_ALT.combine(ONE)) |
        R3.combine(ONE, ZERO.combine_int(ONE)) |
        R3)
CARBINE = Term('to').opt() + Term(
    'Carbine', 'covering', 'caught mine', 'carbon', 'cop mine', 'car by then', 'comment', 'coming',
    'copying', 'cut me in', 'carpet', 'Karmine', 'carbene', 'come in', 'car bye', 'copy', 'copy me',
    'combine')
R301_CARBINE = (R301 + CARBINE.opt()) | Term('R31Carbine', 'R31cabine', 'I have to uncover him')

R99 = Term(
    'R-99', 'R99', '$5.99', 'or nine nine', 'or nine-to-nine', 'or ninety-nine', 'I don\'t know',
    'R9', 'all done', 'I had a dead eye', 'hard on your nine', 'hard 99', 'all right any line',
    'I don\'t need nine', 'irony 9', 'I already know', 'I already need a 9', '$1.99', 'R-89',
    'iron 9', 'oh I don\'t even know')
REVVED = Term('revved up', 'wrapped up', 'rev it up', 'ribbed up', 'revved it', 'rev\'d', 'revved',
              'R.I.P.', 'round')
RE_45 = Term('R/E 45', 'RE-45', 'RA-45', 'R45', 'RD-45', 'are we 45', 'RU45', 'are you 45',
             'R8-45')
SENTINEL = Term('sentinel', 'what\'s that now', 'is that not', 'setting\' off', 'that\'s it now',
                'techno', 'is that no', 'said no', 'such an old', '7-0')
AMPED = Term('amped', 'ant', 'it', 'end', 'yipped')
SPITFIRE = Term('Spitfire', 'step out of the car', 'is that her', 'it\'s a bit better', 'fitzpire',
                'skip her', 'zip fire', 'stay fire', 'set fire')
TRIPLE_TAKE = Term('Triple Take', 'triple-tick', 'triple T', 'chipotle', 'sure thing', 'chilti',
                   'Chanté', 'triple-click', 'it\'s real thick')
VOLT = ((Term('Volt', 'oh', 'bull', 'boop', 'what', 'well', 'vote', 'voltz', 'volts') + SMG_OPT) |
        (BOLT + SMG))
WINGMAN = Term('Wingman', 'we\'ll be back', 'wing then', 'wing men', 'wingmen')
BOOSTED_LOADER = (Term('boosted', 'who\'s dead', 'that\'s it') +
                  Term('loader', 'loaded', 'love you', 'odor'))

WEAPON_ARCHETYPE_TERMS: Tuple[RequiredTerm, ...] = (
    THIRTY_THIRTY_REPEATER.combine(SLOW),
    THIRTY_THIRTY_REPEATER.combine(QUICK_OPT),
    ALTERNATOR,
    ALTERNATOR.combine(OPT_WITH_INCL, DISRUPTOR),
    # Want to make sure that "Bocek", "Devotion", and "EVA-8" resolve to weapons till they switch
    # back to not having shatter caps.
    # BOCEK,
    # BOCEK.combine(DRAWN),
    # BOCEK.combine(OPT_WITH_INCL, SHATTER_CAPS, OPT_WITH_INCL, MINIMAL),
    # BOCEK.combine(OPT_WITH_INCL, SHATTER_CAPS, DRAWN),
    BOCEK.combine(OPT_WITH_EXCL, SHATTER_CAPS.opt(), OPT_WITH_INCL, MINIMAL_DRAW),
    BOCEK.combine(OPT_WITH_EXCL,
                  SHATTER_CAPS.opt(),
                  OPT_WITH_INCL,
                  DRAWN.opt(include_in_speech=True)),
    CAR.combine(SMG_OPT),
    CHARGE_RIFLE,
    # DEVOTION,
    # DEVOTION.combine(CARE_PACKAGE),
    # DEVOTION.combine(OPT_WITH_INCL, TURBOCHARGER),
    DEVOTION.combine(CARE_PACKAGE.opt()),
    # EVA_8,
    # EVA_8.combine(CARE_PACKAGE),
    EVA_8.combine(CARE_PACKAGE.opt()),
    FLATLINE,
    G7_SCOUT,
    HAVOC,
    HAVOC.combine(OPT_WITH_INCL, TURBOCHARGER),
    HEMLOCK,
    KRABER,
    LONGBOW,
    L_STAR,
    MASTIFF,
    MOZAMBIQUE,
    MOZAMBIQUE.combine(OPT_WITH_INCL, HAMMERPOINT),
    NEMESIS,
    P2020,
    P2020.combine(OPT_WITH_INCL, HAMMERPOINT),
    PEACEKEEPER,
    PEACEKEEPER.combine(OPT_WITH_INCL, DISRUPTOR),
    PROWLER,
    PROWLER.combine(CARE_PACKAGE),
    R301_CARBINE,
    R99,
    RAMPAGE,
    RAMPAGE.combine(REVVED),
    RE_45,
    RE_45.combine(OPT_WITH_INCL, HAMMERPOINT),
    SENTINEL,
    SENTINEL.combine(AMPED),
    SPITFIRE,
    TRIPLE_TAKE,
    VOLT.combine(SMG_OPT),
    WINGMAN,
    WINGMAN.combine(OPT_WITH_INCL, BOOSTED_LOADER)
)
