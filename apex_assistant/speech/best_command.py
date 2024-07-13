import logging

from apex_assistant.checker import check_type
from apex_assistant.speech.apex_terms import BEST, NUMBER_TERMS
from apex_assistant.speech.command import Command
from apex_assistant.speech.term import Words
from apex_assistant.speech.term_translator import Translator
from apex_assistant.weapon_comparer import WeaponComparer
from apex_assistant.weapon_translator import WeaponTranslator


LOGGER = logging.getLogger()


class BestCommand(Command):
    def __init__(self, weapon_translator: WeaponTranslator, weapon_comparer: WeaponComparer):
        check_type(WeaponTranslator, weapon_translator=weapon_translator)
        check_type(WeaponComparer, weapon_comparer=weapon_comparer)
        super().__init__(BEST)
        self._weapon_translator = weapon_translator
        self._comparer = weapon_comparer
        self._number_translator = Translator[int]({number_term: int(number_term)
                                                   for number_term in NUMBER_TERMS})

    def _execute(self, arguments: Words) -> str:
        numbers = set(self._number_translator.translate_terms(arguments))
        if len(numbers) > 1:
            return 'Too many numbers specified. Only specify one.'
        if len(numbers) == 0:
            vals = self._number_translator.values()
            min_val = min(vals)
            max_val = max(vals)
            return f'Must specify a number between {min_val} and {max_val}'
        number = next(num.get_value() for num in numbers)

        LOGGER.debug(f'Getting {number} best weapons.')
        weapons = tuple(weapon.combine_with_sidearm(weapon).reload()
                        for weapon in self._weapon_translator.get_all_weapons())
        comparison_result = self._comparer.compare_weapons(weapons).limit_to_best_num(number)
        LOGGER.info(f'Best {number} weapons:\n'
                    f'  {comparison_result.get_archetypes()}')
        audible_names = ' '.join([weapon_archetype.get_term().to_audible_str()
                                  for weapon_archetype in comparison_result.get_archetypes()])
        return audible_names
