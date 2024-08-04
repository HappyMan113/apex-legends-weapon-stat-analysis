from enum import StrEnum

from apex_assistant.speech.term import RequiredTerm, Term


class WeaponClass(StrEnum):
    ASSAULT_RIFLE = 'AR'
    LMG = 'LMG'
    MARKSMAN = 'Marksman'
    PISTOL = 'Pistol'
    SHOTGUN = 'Shotgun'
    SMG = 'SMG'
    SNIPER = 'Sniper'

    __ALIASES_DICT: dict['WeaponClass', RequiredTerm] = {
        ASSAULT_RIFLE: Term('ARs'),
        LMG: Term('LNGE', 'LMGs', 'LNG'),
        MARKSMAN: Term('marksmans'),
        PISTOL: Term('pistols'),
        SHOTGUN: Term('shotguns'),
        SMG: Term('SMGs'),
        SNIPER: Term('snipers')
    }

    def get_term(self: 'WeaponClass') -> RequiredTerm:
        # noinspection PyTypeChecker
        aliases: dict[WeaponClass, RequiredTerm] = WeaponClass.__ALIASES_DICT
        term = Term(str(self))
        if self in aliases:
            term = term | aliases[self]
        return term
