import abc
import logging
from typing import Dict, Optional, Tuple, final

from apex_assistant.checker import check_mapping, check_type
from apex_assistant.loadout_comparator import LoadoutComparator
from apex_assistant.loadout_translator import LoadoutTranslator
from apex_assistant.speech.apex_command import ApexCommand
from apex_assistant.speech.apex_terms import (BEST,
                                              LOADOUT,
                                              LOADOUTS,
                                              SIDEARM,
                                              SIDEARMS,
                                              SINGLE_SHOT,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import RequiredTerm, Term, TermBase, Words
from apex_assistant.speech.term_translator import IntTranslator, Translator
from apex_assistant.weapon import FullLoadout, Loadout, SingleWeaponLoadout, Weapon
from apex_assistant.weapon_class import WeaponClass


LOGGER = logging.getLogger()


class BestCommand(ApexCommand):
    def __init__(self,
                 loadout_translator: LoadoutTranslator,
                 loadout_comparator: LoadoutComparator):
        check_type(LoadoutTranslator, loadout_translator=loadout_translator)
        check_type(LoadoutComparator, loadout_comparator=loadout_comparator)

        subcommands = (_BestSidearmCommand(self), _BestLoadoutCommand(self))
        super().__init__(term=BEST,
                         loadout_translator=loadout_translator,
                         loadout_comparator=loadout_comparator)
        self._subcommand_translator = Translator[Tuple[_BestSubcommand, bool]]({
            subcommand.get_term(plural): (subcommand, plural)
            for subcommand in subcommands
            for plural in (False, True)})
        self._number_translator = IntTranslator()

    @final
    def _execute(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)

        subcommands = self._subcommand_translator.translate_terms(arguments)
        subcommand_translation = subcommands.get_first_term()
        if subcommand_translation is None:
            return 'Must specify a valid subcommand.'

        if len(subcommands) > 1:
            LOGGER.warning(
                'More than one "best" subcommand specified. Only the first will be parsed.')

        subcommand, plural = subcommand_translation.get_value()
        subcommand_translation.get_word_start_idx()

        number_args = (subcommand_translation.get_preceding_words() +
                       subcommand_translation.get_following_words())
        number_term = self._number_translator.translate_at_start(number_args)

        if number_term is None:
            if len(number_args) > 0 and subcommand.is_default():
                LOGGER.warning('No number specified amidst arguments. Skipping.')

                # It's fairly common for the assistant to mishear the word "best" randomly in
                # speech, so we skip don't play back any message in this case.
                return ''

            number = 3 if plural else 1
        elif number_term.get_value() <= 0:
            return 'Number must be positive.'
        else:
            number = number_term.get_value()

        loadouts_and_terms = subcommand.get_loadouts(number_args, number=number)
        check_mapping(allowed_key_types=Loadout,
                      allowed_value_types=TermBase,
                      **{f'{subcommand.get_term()}.get_loadouts()': loadouts_and_terms})
        if len(loadouts_and_terms) == 0:
            return f'Must specify a {subcommand.get_what_to_specify()}.'

        loadouts = tuple(loadouts_and_terms.keys())
        comparison_result = self._comparator.compare_loadouts(loadouts).limit_to_best_num(number)
        loadout_terms = tuple(loadouts_and_terms[loadout]
                              for loadout in comparison_result.get_sorted_loadouts())
        number = len(loadout_terms)
        subcommand_term = subcommand.get_term()
        best_weapons_str = (f'{number} best {subcommand_term}s' if number != 1 else
                            f'Best {subcommand_term}')
        LOGGER.info(f'{best_weapons_str}: {comparison_result}')
        s_suffix = 's' if number != 1 else ''
        prefix = f'Best {subcommand_term.to_audible_str()}{s_suffix}: '
        return prefix + ' '.join((term.to_audible_str() for term in loadout_terms))


class _BestSubcommand:
    def __init__(self,
                 singular_subcommand_term: RequiredTerm,
                 plural_subcommand_term: TermBase,
                 main_command: BestCommand):
        check_type(RequiredTerm, singular_subcommand_term=singular_subcommand_term)
        check_type(TermBase, plural_subcommand_term=plural_subcommand_term)
        check_type(BestCommand, main_command=main_command)
        if singular_subcommand_term == plural_subcommand_term:
            raise ValueError('Singular and plural terms must be different.')
        self._singular_subcommand_term = singular_subcommand_term
        self._plural_subcommand_term = plural_subcommand_term
        self._main_command = main_command

    def get_term(self, plural: bool = False) -> TermBase:
        return self._plural_subcommand_term if plural else self._singular_subcommand_term

    def get_translator(self) -> LoadoutTranslator:
        return self._main_command.get_translator()

    def get_comparator(self) -> LoadoutComparator:
        return self._main_command.get_comparator()

    @abc.abstractmethod
    def get_loadouts(self, arguments: Words, number: int) -> Dict[FullLoadout, TermBase]:
        raise NotImplementedError()

    def is_default(self):
        return self._plural_subcommand_term.is_opt()

    def get_what_to_specify(self) -> str:
        return self.get_term().to_audible_str()


class _BestLoadoutCommand(_BestSubcommand):
    def __init__(self, main_command: BestCommand):
        super().__init__(singular_subcommand_term=LOADOUT,
                         plural_subcommand_term=LOADOUTS,
                         main_command=main_command)
        self._weapon_class_translator: Translator[WeaponClass] = Translator({
            Term(weapon_class): weapon_class
            for weapon_class in WeaponClass
        })

    def get_loadouts(self, arguments: Words, number: int) -> Dict[Loadout, TermBase]:
        translator = self.get_translator()
        weapon_class_terms = self._weapon_class_translator.translate_terms(arguments)
        main_weapon_class = weapon_class_terms.get_latest_value()
        best_loadouts = self.get_comparator().get_best_loadouts(
            weapons=translator.get_fully_kitted_weapons(),
            max_num_loadouts=number,
            main_weapon_class=main_weapon_class)
        best_loadouts_and_terms: Dict[Loadout, TermBase] = {
            best_loadout: self._get_term(best_loadout.get_main_loadout(),
                                         best_loadout.get_sidearm())
            for best_loadout in best_loadouts
        }

        return best_loadouts_and_terms

    @staticmethod
    def _get_term(main_loadout: SingleWeaponLoadout, sidearm: Optional[Weapon]):
        check_type(SingleWeaponLoadout, main_loadout=main_loadout)
        check_type(Weapon, optional=True, sidearm=sidearm)
        main_weapon, is_single_shot = main_loadout.unwrap()
        main_term = main_weapon.get_archetype().get_base_term()
        if is_single_shot:
            main_term = main_term.append(SINGLE_SHOT)
        if sidearm is None:
            return main_term
        side_term = sidearm.get_archetype().get_base_term()
        return main_term.append(WITH_SIDEARM, side_term)
