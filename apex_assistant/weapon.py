import abc
import logging
import math
from enum import StrEnum
from types import MappingProxyType
from typing import Dict, Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, \
    Union, final

import numpy as np

from apex_assistant.checker import (check_bool,
                                    check_equal_length,
                                    check_float,
                                    check_int,
                                    check_str,
                                    check_tuple, check_type,
                                    to_kwargs)
from apex_assistant.legend import Legend
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_terms import (ALL_BOLT_TERMS,
                                              ALL_MAG_TERMS,
                                              ALL_STOCK_TERMS,
                                              BASE,
                                              LEVEL_TERMS,
                                              MAIN,
                                              SIDEARM,
                                              SINGLE_SHOT,
                                              SKIPPED, WITH_SIDEARM)
from apex_assistant.speech.suffix import Suffix, SuffixedArchetypeType
from apex_assistant.speech.term import RequiredTerm, TermBase, Words
from apex_assistant.speech.term_translator import SingleTermFinder, Translator
from apex_assistant.speech.translations import TranslatedValue
from apex_assistant.weapon_class import WeaponClass


T = TypeVar('T')
_LOGGER = logging.getLogger()


class StatsBase(abc.ABC, Generic[T]):
    _LEVEL_TERMS: tuple[RequiredTerm, ...] = ((BASE,) + LEVEL_TERMS)
    _LEVEL_TRANSLATOR = Translator[int]({term: level for level, term in enumerate(_LEVEL_TERMS)})

    def __init__(self,
                 all_terms: Optional[Union[RequiredTerm, Tuple[RequiredTerm, ...]]],
                 *all_values: T):
        if not isinstance(all_terms, tuple):
            all_terms: Tuple[Optional[RequiredTerm], ...] = (all_terms,)
        else:
            all_terms: Tuple[Optional[RequiredTerm], ...] = all_terms
        check_type(RequiredTerm, optional=True, **to_kwargs(all_terms=all_terms))
        check_equal_length(all_terms=all_terms,
                           all_values=all_values,
                           **{'set(all_terms)': set(all_terms)})
        if len(all_terms) > len(self._LEVEL_TERMS):
            raise ValueError('Cannot have more terms than levels.')
        term_to_val_dict: MappingProxyType[Optional[RequiredTerm], T] = MappingProxyType({
            term: time for term, time in zip(all_terms, all_values)
        })
        self._all_terms = all_terms
        self._all_values = all_values
        self._translator = Translator({term: val
                                       for term, val in term_to_val_dict.items()
                                       if term is not None})
        self._term_to_val_dict = term_to_val_dict

    def __len__(self) -> int:
        return len(self._term_to_val_dict)

    def get_best_stats(self) -> Tuple[Optional[RequiredTerm], T]:
        return self._all_terms[-1], self._all_values[-1]

    def get_all_stats(self) -> MappingProxyType[Optional[RequiredTerm], T]:
        return self._term_to_val_dict

    def translate_stats(self, words: Words) -> \
            Tuple[Optional[Tuple[Optional[RequiredTerm], T]], Words]:
        check_type(Words, words=words)
        translation = self._translator.translate_terms(words)
        first_term = translation.get_first_term()
        if first_term is None:
            term_and_val = None
        else:
            term_and_val = first_term.get_term(), first_term.get_value()
        return term_and_val, translation.get_untranslated_words()

    def _get_highest_level(self):
        return len(self._all_terms) - 1

    def get_stats_for_level(self, level: Optional[int]) -> Tuple[RequiredTerm, T]:
        check_int(min_value=0, optional=True, level=level)
        max_level = self._get_highest_level()
        if level is None:
            # Get the highest level by default.
            level = max_level
        else:
            level = min(level, max_level)

        term: Optional[RequiredTerm] = self._all_terms[level]
        value: T = self._all_values[level]
        return term, value

    def best_to_worst(self) -> \
            Generator[Tuple[Optional[RequiredTerm], T],
            None,
            None]:
        for level in range(self._get_highest_level(), -1, -1):
            yield self.get_stats_for_level(level)


class MagazineCapacityBase(StatsBase[int]):
    pass


class MagazineCapacity(MagazineCapacityBase):
    def __init__(self, base_capacity: int):
        check_int(min_val=1, base_capacity=base_capacity)
        super().__init__(None, base_capacity)


class MagazineCapacities(MagazineCapacityBase):
    def __init__(self,
                 base_capacity: int,
                 level_1_capacity: int,
                 level_2_capacity: int,
                 level_3_capacity: int):
        check_int(min_val=1,
                  base_capacity=base_capacity,
                  level_1_capacity=level_1_capacity,
                  level_2_capacity=level_2_capacity,
                  level_3_capacity=level_3_capacity)
        super().__init__(ALL_MAG_TERMS,
                         base_capacity,
                         level_1_capacity,
                         level_2_capacity,
                         level_3_capacity)


class StockStatValues:
    def __init__(self,
                 tactical_reload_time_secs: Optional[float],
                 full_reload_time_secs: Optional[float],
                 holster_time_secs: float,
                 ready_to_fire_time_secs: float):
        check_float(min_value=0,
                    min_is_exclusive=True,
                    optional=True,
                    tactical_reload_time_secs=tactical_reload_time_secs,
                    full_reload_time_secs=full_reload_time_secs)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    holster_time_secs=holster_time_secs,
                    ready_to_fire_time_secs=ready_to_fire_time_secs)
        self._tactical_reload_time_secs = tactical_reload_time_secs
        self._full_reload_time_secs = full_reload_time_secs
        self._holster_time_secs = holster_time_secs
        self._ready_to_fire_time_secs = ready_to_fire_time_secs

    def get_tactical_reload_time_secs(self) -> Optional[float]:
        return self._tactical_reload_time_secs

    def get_full_reload_time_secs(self) -> Optional[float]:
        return self._full_reload_time_secs

    def get_holster_time_secs(self) -> float:
        return self._holster_time_secs

    def get_ready_to_fire_time_secs(self) -> float:
        return self._ready_to_fire_time_secs


class StockStatsBase(StatsBase[StockStatValues | None]):
    pass


class StockStat(StockStatsBase):
    def __init__(self, stats: StockStatValues | None):
        super().__init__(None, stats)


class StockStats(StockStatsBase):
    def __init__(self,
                 no_stock_stats: StockStatValues,
                 level_1_stock_stats: StockStatValues,
                 level_2_stock_stats: StockStatValues,
                 level_3_stock_stats: StockStatValues):
        check_type(StockStatValues,
                   no_stock_stats=no_stock_stats,
                   level_1_stock_time_secs=level_1_stock_stats,
                   level_2_stock_time_secs=level_2_stock_stats,
                   level_3_stock_time_secs=level_3_stock_stats)
        super().__init__(ALL_STOCK_TERMS,
                         no_stock_stats,
                         level_1_stock_stats,
                         level_2_stock_stats,
                         level_3_stock_stats)


class RoundsPerMinuteBase(StatsBase[float]):
    pass


class RoundsPerMinute(RoundsPerMinuteBase):
    def __init__(self, base_rounds_per_minute: float):
        check_float(base_rounds_per_minute=base_rounds_per_minute,
                    min_value=0,
                    min_is_exclusive=True)
        super().__init__(None, base_rounds_per_minute)


class RoundsPerMinutes(RoundsPerMinuteBase):
    def __init__(self,
                 base_rounds_per_minute: float,
                 level_1_rounds_per_minute: float,
                 level_2_rounds_per_minute: float,
                 level_3_rounds_per_minute: float):
        check_float(min_value=0,
                    min_is_exclusive=True,
                    base_rounds_per_minute=base_rounds_per_minute,
                    level_1_rounds_per_minute=level_1_rounds_per_minute,
                    level_2_rounds_per_minute=level_2_rounds_per_minute,
                    level_3_rounds_per_minute=level_3_rounds_per_minute)
        super().__init__(ALL_BOLT_TERMS,
                         base_rounds_per_minute,
                         level_1_rounds_per_minute,
                         level_2_rounds_per_minute,
                         level_3_rounds_per_minute)


class RoundTimingType(StrEnum):
    DEVOTION = 'devotion'
    DELAY = 'delay'
    BURST = 'burst'
    NEMESIS = 'nemesis'
    NONE = 'none'


HUMAN_REACTION_TIME_SECONDS = 0.2


class RoundTiming(abc.ABC):
    @abc.abstractmethod
    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        raise NotImplementedError('Must implement.')

    @final
    def get_cumulative_damage(self, base_weapon: 'Weapon', time_seconds: float) -> float:
        num_rounds = self.get_num_rounds_shot(
            base_weapon=base_weapon,
            time_seconds=time_seconds)
        if num_rounds < 0:
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a non-negative number of rounds.')
        if num_rounds > base_weapon.get_magazine_capacity():
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a number of rounds greater than the magazine capacity.')
        return base_weapon.get_damage_per_round() * num_rounds

    @abc.abstractmethod
    def get_magazine_duration_seconds(self,
                                      base_weapon: 'Weapon',
                                      tactical: bool = False) -> float:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError('Must implement.')


class RoundTimingDevotion(RoundTiming):
    def __init__(self, rounds_per_minute_initial: float, spinup_time_seconds: float):
        check_float(rounds_per_minute_initial=rounds_per_minute_initial,
                    min_value=0,
                    min_is_exclusive=True)
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)

        self._rounds_per_minute_initial = rounds_per_minute_initial
        rounds_per_second_initial = rounds_per_minute_initial / 60
        self._rounds_per_second_initial = rounds_per_second_initial
        self._spinup_time_seconds = spinup_time_seconds

    def _get_shot_times_seconds(self, base_weapon: 'Weapon') -> np.ndarray[np.floating]:
        # As a sanity check, magazine duration for Care Package Devotion was measured as ~227-229
        # frames at 60 FPS (~3.783-3.817 seconds).
        magazine_capacity = base_weapon.get_magazine_capacity()
        rps_initial = self._rounds_per_second_initial
        rps_final = base_weapon.get_rounds_per_second()
        rps_per_sec = (rps_final - rps_initial) / self._spinup_time_seconds

        times: np.ndarray[np.floating] = np.empty(magazine_capacity)
        times[0] = 0
        for round_num in range(1, magazine_capacity):
            prev_t = times[round_num - 1]
            rps_t = min(rps_final, rps_initial + rps_per_sec * prev_t)
            seconds_per_round_t = 1 / rps_t
            times[round_num] = prev_t + seconds_per_round_t

        return times

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        return np.count_nonzero(time_seconds >= self._get_shot_times_seconds(base_weapon))

    def get_magazine_duration_seconds(self,
                                      base_weapon: 'Weapon',
                                      tactical: bool = False) -> float:
        return self._get_shot_times_seconds(base_weapon)[-1] + HUMAN_REACTION_TIME_SECONDS

    def __hash__(self):
        return hash(self.__class__) ^ hash((self._rounds_per_minute_initial,
                                            self._spinup_time_seconds))

    def __eq__(self, other):
        return (isinstance(other, RoundTimingDevotion) and
                self._rounds_per_minute_initial == other._rounds_per_minute_initial and
                self._spinup_time_seconds == other._spinup_time_seconds)


class RoundTimingBurst(RoundTiming):
    def __init__(self, rounds_per_burst: int, burst_fire_delay: float):
        check_int(min_value=2, rounds_per_burst=rounds_per_burst)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_fire_delay=burst_fire_delay)

        self._rounds_per_burst = rounds_per_burst
        self._burst_fire_delay = burst_fire_delay

    def _get_burst_duration_seconds(self, base_weapon: 'Weapon'):
        seconds_per_round = 1 / base_weapon.get_rounds_per_second()
        burst_duration_seconds = seconds_per_round * (self._rounds_per_burst - 1)
        return burst_duration_seconds

    def _get_burst_period_seconds(self, base_weapon: 'Weapon') -> float:
        burst_duration_seconds = self._get_burst_duration_seconds(base_weapon)
        return burst_duration_seconds + self._burst_fire_delay

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        period_seconds = self._get_burst_period_seconds(base_weapon)

        num_full_bursts_shot, remainder_secs = divmod(time_seconds, period_seconds)
        num_full_bursts_shot = int(num_full_bursts_shot)

        num_rounds_shot_in_partial_period = min(
            1 + math.floor(remainder_secs * base_weapon.get_rounds_per_second()),
            self._rounds_per_burst)

        return min(
            num_full_bursts_shot * self._rounds_per_burst + num_rounds_shot_in_partial_period,
            base_weapon.get_magazine_capacity())

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: bool = False) -> float:
        period_seconds = self._get_burst_period_seconds(base_weapon)
        num_bursts = math.ceil((base_weapon.get_magazine_capacity() - tactical) /
                               self._rounds_per_burst)
        num_last_burst_rounds = (base_weapon.get_magazine_capacity() -
                                 (num_bursts - 1) * self._rounds_per_burst)

        seconds_per_round = 1 / base_weapon.get_rounds_per_second()
        last_burst_duration_seconds = seconds_per_round * (num_last_burst_rounds - 1)

        return ((period_seconds * (num_bursts - 1) + last_burst_duration_seconds) +
                HUMAN_REACTION_TIME_SECONDS)

    def __hash__(self):
        return hash(self.__class__) ^ hash((self._rounds_per_burst,
                                            self._burst_fire_delay))

    def __eq__(self, other):
        return (isinstance(other, RoundTimingBurst) and
                self._rounds_per_burst == other._rounds_per_burst and
                self._burst_fire_delay == other._burst_fire_delay)


class RoundTimingNemesis(RoundTiming):
    def __init__(self,
                 rounds_per_burst: int,
                 burst_fire_delay_initial: float,
                 burst_fire_delay_final: float,
                 burst_charge_fraction: float):
        check_int(min_value=2, rounds_per_burst=rounds_per_burst)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_fire_delay_final=burst_fire_delay_final)
        check_float(min_value=burst_fire_delay_final,
                    min_is_exclusive=True,
                    burst_fire_delay_initial=burst_fire_delay_initial)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_charge_fraction=burst_charge_fraction)

        self._rounds_per_burst = rounds_per_burst
        self._burst_fire_delay_initial = burst_fire_delay_initial
        self._burst_fire_delay_final = burst_fire_delay_final
        self._burst_charge_fraction = burst_charge_fraction

    def _get_num_bursts(self, base_weapon: 'Weapon') -> int:
        return math.ceil(base_weapon.get_magazine_capacity() / self._rounds_per_burst)

    def _get_burst_duration_seconds(self, base_weapon: 'Weapon') -> float:
        seconds_per_round = 1 / base_weapon.get_rounds_per_second()
        burst_duration_seconds = seconds_per_round * (self._rounds_per_burst - 1)
        return burst_duration_seconds

    def _get_burst_periods_seconds(self, base_weapon: 'Weapon') -> np.ndarray[np.floating]:
        burst_duration_seconds = self._get_burst_duration_seconds(base_weapon)
        num_bursts = self._get_num_bursts(base_weapon)
        charge_fractions = (np.arange(num_bursts - 1) * self._burst_charge_fraction).clip(max=1)
        burst_fire_delays = ((1 - charge_fractions) * self._burst_fire_delay_initial +
                             charge_fractions * self._burst_fire_delay_final)

        # Delay after last burst is an elephant.
        burst_fire_delays = np.append(burst_fire_delays, 0)

        return burst_duration_seconds + burst_fire_delays

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        periods_seconds = self._get_burst_periods_seconds(base_weapon)
        period_stop_times_seconds = periods_seconds.cumsum()

        num_full_periods_shot = np.count_nonzero(time_seconds >= period_stop_times_seconds)
        full_periods_shot = num_full_periods_shot * self._rounds_per_burst

        partial_period_start_time_secs = (
            0 if num_full_periods_shot == 0 else
            float(period_stop_times_seconds[num_full_periods_shot - 1]))
        rel_burst_time_secs = time_seconds - partial_period_start_time_secs
        if rel_burst_time_secs < 0:
            num_rounds_shot_in_partial_period = 0
        else:
            num_rounds_shot_in_partial_period = min(
                1 + math.floor(rel_burst_time_secs * base_weapon.get_rounds_per_second()),
                self._rounds_per_burst)

        return min(full_periods_shot + num_rounds_shot_in_partial_period,
                   base_weapon.get_magazine_capacity())

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: bool = False) -> float:
        periods_seconds = self._get_burst_periods_seconds(base_weapon)
        return float(periods_seconds.sum()) + HUMAN_REACTION_TIME_SECONDS

    def __hash__(self):
        return hash(self.__class__) ^ hash((self._rounds_per_burst,
                                            self._burst_fire_delay_initial,
                                            self._burst_fire_delay_final,
                                            self._burst_charge_fraction))

    def __eq__(self, other):
        return (isinstance(other, RoundTimingNemesis) and
                self._rounds_per_burst == other._rounds_per_burst and
                self._burst_fire_delay_initial == other._burst_fire_delay_initial and
                self._burst_fire_delay_final == other._burst_fire_delay_final and
                self._burst_charge_fraction == other._burst_charge_fraction)


class RoundTimingNone(RoundTiming):
    _INSTANCE: Optional['RoundTimingNone'] = None
    __create_key = object()

    def __init__(self, create_key):
        super().__init__()
        if create_key != RoundTimingNone.__create_key:
            raise RuntimeError(f'Cannot instantiate {RoundTimingNone.__name__} except from '
                               'get_instance() method.')

    @staticmethod
    def get_instance():
        if RoundTimingNone._INSTANCE is None:
            RoundTimingNone._INSTANCE = RoundTimingNone(RoundTimingNone.__create_key)
        return RoundTimingNone._INSTANCE

    @staticmethod
    def _get_magazine_num_rounds(base_weapon: 'Weapon', tactical: bool = False) -> int:
        return base_weapon.get_magazine_capacity() - tactical

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: bool = False) -> float:
        check_type(Weapon, base_weapon=base_weapon)
        check_bool(tactical=tactical)

        # We subtract by one to take into account that the first round is shot instantly.
        num_rounds = self._get_magazine_num_rounds(base_weapon, tactical=tactical) - 1
        rounds_per_minute = base_weapon.get_rounds_per_minute()
        rounds_per_second = rounds_per_minute / 60
        magazine_duration_seconds = num_rounds / rounds_per_second
        return magazine_duration_seconds + HUMAN_REACTION_TIME_SECONDS

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        rounds_per_second = base_weapon.get_rounds_per_second()
        num_rounds = min(1 + math.floor(time_seconds * rounds_per_second),
                         base_weapon.get_magazine_capacity())
        return num_rounds

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, RoundTimingNone)


class RoundTimingHavoc(RoundTiming):
    _NO_SPINUP = RoundTimingNone.get_instance()

    def __init__(self, spinup_time_seconds: float):
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)
        self._spinup_time_seconds = spinup_time_seconds

    def get_magazine_duration_seconds(self,
                                      base_weapon: 'Weapon',
                                      tactical: bool = False) -> float:
        return self._NO_SPINUP.get_magazine_duration_seconds(base_weapon=base_weapon,
                                                             tactical=tactical)

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float) -> int:
        spinup_time_seconds = self._spinup_time_seconds
        if time_seconds < spinup_time_seconds:
            return 0

        return self._NO_SPINUP.get_num_rounds_shot(
            base_weapon=base_weapon,
            time_seconds=time_seconds - spinup_time_seconds)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._spinup_time_seconds)

    def __eq__(self, other):
        return (isinstance(other, RoundTimingHavoc) and
                other._spinup_time_seconds == self._spinup_time_seconds)


class Loadout(abc.ABC):
    def __init__(self, name: str, term: RequiredTerm):
        check_str(allow_blank=False, name=name)
        check_type(RequiredTerm, term=term)
        self.name = name
        self.term = term

    def get_term(self) -> RequiredTerm:
        return self.term

    @final
    def get_main_weapon(self) -> 'Weapon':
        return self.get_main_loadout().get_weapon()

    @abc.abstractmethod
    def get_main_loadout(self) -> 'SingleWeaponLoadout':
        raise NotImplementedError()

    def get_name(self):
        return self.name

    def __repr__(self):
        return self.name

    @abc.abstractmethod
    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_holster_time_secs(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ready_to_fire_time_secs(self) -> float:
        raise NotImplementedError()


class _ReloadConfiguration(StrEnum):
    RELOAD_MAIN = 'main reload'
    RELOAD_SIDEARM = 'sidearm reload'
    RELOAD_NEITHER = 'reload N/A'


class FullLoadout(Loadout):
    NUM_WEAPONS = 2

    def __init__(self, main_loadout: 'SingleWeaponLoadout', sidearm: 'Weapon'):
        check_type(SingleWeaponLoadout, main_weapon=main_loadout)
        check_type(Weapon, sidearm=sidearm)
        self.main_loadout = main_loadout
        self.sidearm: Weapon = sidearm

        main_reload_time_secs = main_loadout.get_weapon().get_tactical_reload_time_secs(True)
        sidearm_reload_time_secs = sidearm.get_tactical_reload_time_secs(False)
        if main_reload_time_secs is None and sidearm_reload_time_secs is None:
            reloading_loadout_config = _ReloadConfiguration.RELOAD_NEITHER
        elif main_reload_time_secs is None:
            reloading_loadout_config = _ReloadConfiguration.RELOAD_SIDEARM
        elif (sidearm_reload_time_secs is None or
              (sidearm.get_swap_time_seconds(main_loadout) + main_reload_time_secs <
               sidearm_reload_time_secs)):
            reloading_loadout_config = _ReloadConfiguration.RELOAD_MAIN
        else:
            reloading_loadout_config = _ReloadConfiguration.RELOAD_SIDEARM

        reload_str = (f' ({reloading_loadout_config})'
                      if (reloading_loadout_config is _ReloadConfiguration.RELOAD_MAIN) else '')
        super().__init__(f'{main_loadout.get_name()}, {SIDEARM} {sidearm.get_name()}{reload_str}',
                         MAIN.append(main_loadout.get_term(), WITH_SIDEARM, sidearm.get_term()))

        self._reloading_loadout_config = reloading_loadout_config

    def get_holster_time_secs(self) -> float:
        return self.sidearm.get_holster_time_secs()

    def get_ready_to_fire_time_secs(self) -> float:
        return self.main_loadout.get_ready_to_fire_time_secs()

    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float) -> float:
        main_total_duration_seconds = self.main_loadout.get_magazine_duration_seconds()
        cum_damage = self.main_loadout.get_single_mag_cumulative_damage(
            min(time_seconds, main_total_duration_seconds),
            distance_meters=distance_meters)

        swap_time_seconds = self.main_loadout.get_swap_time_seconds(self.sidearm)
        sidearm_start_time_seconds = main_total_duration_seconds + swap_time_seconds
        if time_seconds >= sidearm_start_time_seconds:
            time_seconds -= sidearm_start_time_seconds
            cum_damage += self.sidearm.get_single_mag_cumulative_damage(
                time_seconds,
                distance_meters=distance_meters)

        return cum_damage

    def _get_cumulative_damage(self,
                               reloading_loadout: 'SingleWeaponLoadout',
                               time_seconds: float,
                               distance_meters: float,
                               swap_time_seconds: float = 0,
                               swapped: bool = False) -> float:
        single_mag_cumulative_damage = self.get_single_mag_cumulative_damage(
            time_seconds,
            distance_meters=distance_meters)
        reload_time_secs = reloading_loadout.get_weapon().get_tactical_reload_time_secs(swapped)
        if reload_time_secs is None:
            raise ValueError('Reloading loadout must be reloadable.')

        reload_stop_time_seconds = (self.get_magazine_duration_seconds() +
                                    reload_time_secs +
                                    swap_time_seconds)

        if time_seconds < reload_stop_time_seconds:
            return single_mag_cumulative_damage

        return (single_mag_cumulative_damage +
                reloading_loadout.get_cumulative_damage(time_seconds - reload_stop_time_seconds,
                                                        distance_meters=distance_meters))

    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        check_float(min_value=0, time_seconds=time_seconds)
        reloading_loadout_config = self._reloading_loadout_config
        if reloading_loadout_config is _ReloadConfiguration.RELOAD_MAIN:
            return self._get_cumulative_damage(
                reloading_loadout=self.main_loadout,
                time_seconds=time_seconds,
                swap_time_seconds=self.sidearm.get_swap_time_seconds(self.main_loadout),
                swapped=True,
                distance_meters=distance_meters)
        elif reloading_loadout_config is _ReloadConfiguration.RELOAD_SIDEARM:
            return self._get_cumulative_damage(reloading_loadout=self.sidearm,
                                               time_seconds=time_seconds,
                                               distance_meters=distance_meters)
        else:
            return self.get_single_mag_cumulative_damage(time_seconds,
                                                         distance_meters=distance_meters)

    def get_magazine_duration_seconds(self) -> float:
        tactical_main = self._reloading_loadout_config is _ReloadConfiguration.RELOAD_MAIN
        tactical_sidearm = self._reloading_loadout_config is _ReloadConfiguration.RELOAD_SIDEARM
        return (self.main_loadout.get_magazine_duration_seconds(tactical=tactical_main) +
                self.sidearm.get_ready_to_fire_time_secs() +
                self.sidearm.get_magazine_duration_seconds(tactical=tactical_sidearm))

    def __hash__(self):
        return hash(self.__class__) ^ hash((self.main_loadout, self.sidearm))

    def __eq__(self, other):
        return (isinstance(other, FullLoadout) and
                other.main_loadout == self.main_loadout and other.sidearm == self.sidearm)

    def get_main_loadout(self) -> 'SingleWeaponLoadout':
        return self.main_loadout

    def get_sidearm(self) -> 'Weapon':
        return self.sidearm

    @staticmethod
    def get_loadouts(required_weapons: Iterable['Weapon']) -> Generator['FullLoadout', None, None]:
        duplicated_weapons = FullLoadout._get_duplicates(required_weapons)
        return (FullLoadout(main_loadout, sidearm)
                for main_weapon in required_weapons
                for sidearm in required_weapons
                if (sidearm != main_weapon or main_weapon in duplicated_weapons)
                for main_loadout in main_weapon.get_main_loadout_variants())

    @staticmethod
    def _get_duplicates(elements: Iterable['T']) -> set[T]:
        singles: set[T] = set()
        duplicates: set[T] = set()

        for element in elements:
            if element in duplicates:
                continue
            if element in singles:
                duplicates.add(element)
                continue
            singles.add(element)
        return duplicates


class SingleWeaponLoadout(Loadout, abc.ABC):
    def __init__(self, name: str, term: RequiredTerm, variant_term: Optional[TermBase]):
        if variant_term is not None:
            term = term + variant_term
            name = f'{name} ({variant_term})'
        super().__init__(name=name, term=term)
        self.variant_term = variant_term

    @abc.abstractmethod
    def get_archetype(self) -> 'WeaponArchetype':
        raise NotImplementedError()

    @final
    def get_main_loadout(self) -> 'SingleWeaponLoadout':
        return self

    @final
    def get_variant_term(self) -> Optional[TermBase]:
        return self.variant_term

    @abc.abstractmethod
    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        raise NotImplementedError()

    @final
    def add_sidearm(self, sidearm: 'Weapon') -> 'FullLoadout':
        return FullLoadout(self, sidearm)

    @abc.abstractmethod
    def get_weapon(self) -> 'Weapon':
        raise NotImplementedError()

    def get_swap_time_seconds(self, weapon_swap_to: 'SingleWeaponLoadout'):
        check_type(SingleWeaponLoadout, weapon_swap_to=weapon_swap_to)
        return self.get_holster_time_secs() + weapon_swap_to.get_ready_to_fire_time_secs()


class Weapon(SingleWeaponLoadout):
    EIGHTY_PERCENT = 0.8

    def __init__(self,
                 archetype: 'WeaponArchetype',
                 name: str,
                 term: RequiredTerm,
                 weapon_class: WeaponClass,
                 eighty_percent_accuracy_range: int,
                 damage_body: float,
                 holster_time_secs: float,
                 ready_to_fire_time_secs: float,
                 rounds_per_minute: float,
                 magazine_capacity: int,
                 spinup: RoundTiming,
                 heat_based: bool,
                 tactical_reload_time_secs: Optional[float],
                 legend: Optional[Legend]):
        check_type(WeaponArchetype, archetype=archetype)
        check_str(name=name)
        check_type(RequiredTerm, term=term)
        check_type(WeaponClass, weapon_class=weapon_class)
        check_int(min_value=1, eighty_percent_accuracy_range=eighty_percent_accuracy_range)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    damage_body=damage_body,
                    holster_time_secs=holster_time_secs,
                    ready_to_fire_time_secs=ready_to_fire_time_secs,
                    rounds_per_minute=rounds_per_minute)
        check_int(min_value=1, magazine_capacity=magazine_capacity)
        check_type(RoundTiming, spinup=spinup)
        check_float(optional=True,
                    min_value=0,
                    min_is_exclusive=True,
                    tactical_reload_time_secs=tactical_reload_time_secs)
        check_bool(heat_based=heat_based)
        check_type(Legend, optional=True, legend=legend)

        if (legend is Legend.RAMPART and
                weapon_class is WeaponClass.LMG and
                tactical_reload_time_secs is not None):
            # Take into account Rampart's Modded Loader passive ability.
            magazine_capacity = round(magazine_capacity * 1.15)
            tactical_reload_time_secs *= 0.75

        super().__init__(name=name, term=term, variant_term=None)
        self.archetype = archetype
        self.weapon_class = weapon_class
        self.eighty_percent_accuracy_range = eighty_percent_accuracy_range
        self.damage_body = damage_body
        self.holster_time_secs = holster_time_secs
        self.ready_to_fire_time_secs = ready_to_fire_time_secs
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.spinup = spinup
        self.heat_based = heat_based
        self.tactical_reload_time_secs = tactical_reload_time_secs

        damage_per_minute_body = damage_body * rounds_per_minute
        self._damage_per_second_body = damage_per_minute_body / 60

    def get_archetype(self) -> 'WeaponArchetype':
        return self.archetype

    def get_holster_time_secs(self) -> float:
        return self.holster_time_secs

    def get_ready_to_fire_time_secs(self) -> float:
        return self.ready_to_fire_time_secs

    def get_tactical_reload_time_secs(self, from_swap: bool = False) -> float | None:
        if self.tactical_reload_time_secs is None:
            return None
        return self.tactical_reload_time_secs if not from_swap or not self.heat_based else 0

    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float) -> float:
        return (self.spinup.get_cumulative_damage(self, time_seconds) *
                self._get_accuracy_fraction(distance_meters))

    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        reload_time_seconds = self.get_tactical_reload_time_secs()
        if reload_time_seconds is None:
            return self.get_single_mag_cumulative_damage(time_seconds=time_seconds,
                                                         distance_meters=distance_meters)

        mag_duration_seconds = self.get_magazine_duration_seconds(tactical=True)
        period_seconds = mag_duration_seconds + reload_time_seconds
        num_completed_cycles, rel_time_seconds = divmod(time_seconds, period_seconds)
        if rel_time_seconds >= mag_duration_seconds:
            cum_damage = 0
            num_completed_cycles += 1
        else:
            cum_damage = self.get_single_mag_cumulative_damage(time_seconds=rel_time_seconds,
                                                               distance_meters=distance_meters)
        cum_damage += (num_completed_cycles *
                       self.get_single_mag_cumulative_damage(time_seconds=mag_duration_seconds,
                                                             distance_meters=distance_meters))

        return cum_damage

    def _get_accuracy_fraction(self, distance_meters: float) -> float:
        return min(self.EIGHTY_PERCENT * self.eighty_percent_accuracy_range / distance_meters, 1)

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return self.spinup.get_magazine_duration_seconds(self, tactical=tactical)

    def get_damage_per_second_body(self) -> float:
        return self._damage_per_second_body

    def get_eighty_percent_accuracy_range(self) -> int:
        return self.eighty_percent_accuracy_range

    def get_damage_per_round(self) -> float:
        return self.damage_body

    def get_magazine_capacity(self):
        return self.magazine_capacity

    def get_rounds_per_minute(self):
        return self.rounds_per_minute

    def get_rounds_per_second(self) -> float:
        return self.get_rounds_per_minute() / 60

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.name)

    def __eq__(self, other: 'Weapon'):
        return isinstance(other, Weapon) and self.name == other.name

    def get_weapon_class(self) -> WeaponClass:
        return self.weapon_class

    def get_weapon(self) -> 'Weapon':
        return self

    def get_main_loadout_variants(self, allow_skipping: bool = False) -> \
            Generator[SingleWeaponLoadout, None, None]:
        yield self
        if self.is_single_shot_advisable():
            yield self.single_shot()
        if allow_skipping:
            yield self.skip()

    def skip(self) -> 'SkippedLoadout':
        return SkippedLoadout(self)

    def single_shot(self) -> 'SingleShotLoadout':
        return SingleShotLoadout(self)

    def is_single_shot_advisable(self) -> bool:
        seconds_per_round = 1 / self.get_rounds_per_second()
        return seconds_per_round >= self.holster_time_secs


class SingleShotLoadout(SingleWeaponLoadout):
    def __init__(self, wrapped_weapon: Weapon):
        if not wrapped_weapon.is_single_shot_advisable():
            raise ValueError(f'Weapon {wrapped_weapon} must have single shots being advisable!')
        check_type(Weapon, wrapped_weapon=wrapped_weapon)
        super().__init__(wrapped_weapon.get_name(), wrapped_weapon.get_term(), SINGLE_SHOT)
        self.wrapped_weapon = wrapped_weapon

    def get_archetype(self) -> 'WeaponArchetype':
        return self.wrapped_weapon.get_archetype()

    def get_holster_time_secs(self) -> float:
        return self.wrapped_weapon.get_holster_time_secs()

    def get_ready_to_fire_time_secs(self) -> float:
        return self.wrapped_weapon.get_ready_to_fire_time_secs()

    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float) -> float:
        return self.wrapped_weapon.get_damage_per_round()

    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        return self.wrapped_weapon.get_cumulative_damage(time_seconds=time_seconds,
                                                         distance_meters=distance_meters)

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return HUMAN_REACTION_TIME_SECONDS

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.wrapped_weapon)

    def __eq__(self, other):
        return (isinstance(other, SingleShotLoadout) and
                self.wrapped_weapon == other.wrapped_weapon)

    def get_weapon(self) -> 'Weapon':
        return self.wrapped_weapon


class SkippedLoadout(SingleWeaponLoadout):
    def __init__(self, skipped_weapon: Weapon):
        check_type(Weapon, skipped_weapon=skipped_weapon)
        super().__init__(skipped_weapon.get_name(), skipped_weapon.get_term(), SKIPPED)
        self.skipped_weapon = skipped_weapon

    def get_archetype(self) -> 'WeaponArchetype':
        return self.skipped_weapon.get_archetype()

    def get_holster_time_secs(self) -> float:
        return 0

    def get_ready_to_fire_time_secs(self) -> float:
        return 0

    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float) -> float:
        return 0

    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        return 0

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return 0

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.skipped_weapon)

    def __eq__(self, other):
        return isinstance(other, SkippedLoadout) and self.skipped_weapon == other.skipped_weapon

    def get_weapon(self) -> 'Weapon':
        return self.skipped_weapon

    def get_swap_time_seconds(self, weapon_swap_to: 'SingleWeaponLoadout'):
        return 0


class WeaponArchetype:
    def __init__(self,
                 name: str,
                 base_term: RequiredTerm,
                 suffix: Optional[Suffix],
                 weapon_class: WeaponClass,
                 eighty_percent_accuracy_range: int,
                 damage_body: float,
                 rounds_per_minute: RoundsPerMinuteBase,
                 magazine_capacity: MagazineCapacityBase,
                 stock_dependant_stats: StockStatsBase,
                 spinup: RoundTiming,
                 heat_based: bool,
                 associated_legend: Optional[Legend]):
        check_type(Suffix, optional=True, suffix=suffix)

        self.name = name
        self.base_term = base_term
        self.suffix = suffix
        self.full_term: RequiredTerm = (base_term.append(*suffix.get_terms()) if suffix is not None
                                        else base_term)
        self.weapon_class = weapon_class
        self.eighty_percent_accuracy_range = eighty_percent_accuracy_range
        self.damage_body = damage_body
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.stock_dependant_stats = stock_dependant_stats
        self.spinup = spinup
        self.heat_based = heat_based
        self.associated_legend = associated_legend

    def get_associated_legend(self) -> Optional[Legend]:
        return self.associated_legend

    def _get_weapon(self,
                    rounds_per_minute: float,
                    magazine_capacity: Optional[int],
                    stock_stats: Optional[StockStatValues],
                    rpm_term: Optional[RequiredTerm],
                    mag_term: Optional[RequiredTerm],
                    stock_term: Optional[RequiredTerm],
                    legend: Optional[Legend]) -> Weapon:
        name = self.name
        full_term = self.full_term
        weapon_class = self.weapon_class
        eighty_percent_accuracy_range = self.eighty_percent_accuracy_range
        damage_body = self.damage_body
        spinup = self.spinup
        heat_based = self.heat_based

        rpm_name = f' ({rpm_term})' if rpm_term is not None else ''
        mag_name = f' ({mag_term})' if mag_term is not None else ''
        stock_name = f' ({stock_term})' if stock_term is not None else ''

        full_name = f'{name}{rpm_name}{mag_name}{stock_name}'
        more_terms: Tuple[RequiredTerm, ...] = tuple(
            _term
            for _term in (rpm_term, mag_term, stock_term)
            if _term is not None)
        if len(more_terms) > 0:
            full_term = full_term.append(*more_terms)

        tactical_reload_time_secs = stock_stats.get_tactical_reload_time_secs()
        holster_time_secs = stock_stats.get_holster_time_secs()
        ready_to_fire_time_secs = stock_stats.get_ready_to_fire_time_secs()

        return Weapon(archetype=self,
                      name=full_name,
                      term=full_term,
                      weapon_class=weapon_class,
                      eighty_percent_accuracy_range=eighty_percent_accuracy_range,
                      damage_body=damage_body,
                      holster_time_secs=holster_time_secs,
                      ready_to_fire_time_secs=ready_to_fire_time_secs,
                      rounds_per_minute=rounds_per_minute,
                      magazine_capacity=magazine_capacity,
                      spinup=spinup,
                      heat_based=heat_based,
                      tactical_reload_time_secs=tactical_reload_time_secs,
                      legend=legend)

    def get_base_term(self) -> RequiredTerm:
        return self.base_term

    def get_suffix(self) -> Optional[Suffix]:
        return self.suffix

    def get_term(self) -> RequiredTerm:
        return self.full_term

    def get_name(self) -> str:
        return self.name

    def get_class(self) -> str:
        return self.weapon_class

    def __repr__(self) -> str:
        return self.name

    def get_best_weapon(self, legend: Optional[Legend] = None) -> Weapon:
        rpm_term, rpm = self.rounds_per_minute.get_best_stats()
        mag_term, mag = self.magazine_capacity.get_best_stats()
        stock_term, stock_stats = self.stock_dependant_stats.get_best_stats()
        best_weapon = self._get_weapon(rounds_per_minute=rpm,
                                       magazine_capacity=mag,
                                       stock_stats=stock_stats,
                                       rpm_term=rpm_term,
                                       mag_term=mag_term,
                                       stock_term=stock_term,
                                       legend=legend)
        return best_weapon

    def get_best_match(self,
                       words: Words,
                       overall_level: OverallLevel = OverallLevel.PARSE_WORDS,
                       legend: Optional[Legend] = None) -> \
            TranslatedValue[Weapon]:
        check_type(Words, words=words)
        check_type(OverallLevel, overall_level=overall_level)
        check_type(Legend, optional=True, legend=legend)

        rpm_term_and_val, words = self.rounds_per_minute.translate_stats(words)
        mag_term_and_val, words = self.magazine_capacity.translate_stats(words)
        stock_term_and_val, words = self.stock_dependant_stats.translate_stats(words)

        if any(term is None
               for term in (rpm_term_and_val, mag_term_and_val, stock_term_and_val)):
            # TODO: Maybe allow level to be specified after archetype term. Could be confusing
            #  though with trying to decide if it applies to the previous or the next archetype.
            default_level = (int(overall_level)
                             if overall_level is not OverallLevel.PARSE_WORDS
                             else None)

            if rpm_term_and_val is None:
                rpm_term_and_val = self.rounds_per_minute.get_stats_for_level(default_level)

            if mag_term_and_val is None:
                mag_term_and_val = self.magazine_capacity.get_stats_for_level(default_level)

            if stock_term_and_val is None:
                stock_term_and_val = \
                    self.stock_dependant_stats.get_stats_for_level(default_level)

        rpm_term, rpm = rpm_term_and_val
        mag_term, mag = mag_term_and_val
        stock_term, stock_stats = stock_term_and_val

        weapon = self._get_weapon(rounds_per_minute=rpm,
                                  magazine_capacity=mag,
                                  stock_stats=stock_stats,
                                  rpm_term=rpm_term,
                                  mag_term=mag_term,
                                  stock_term=stock_term,
                                  legend=legend)
        return TranslatedValue(weapon, words)

    @staticmethod
    def find_suffix(suffix: Suffix, words: Words):
        check_type(Suffix, suffix=suffix)
        check_type(Words, words=words)
        return all(SingleTermFinder(term).find_all(words) for term in suffix.get_terms())


class WeaponArchetypes:
    def __init__(self,
                 base_archetype: WeaponArchetype,
                 suffixed_archetypes: Tuple[WeaponArchetype, ...]):
        check_type(WeaponArchetype, base_archetype=base_archetype)
        check_tuple(WeaponArchetype, suffixed_archetypes=suffixed_archetypes)

        base_term = base_archetype.get_base_term()
        associated_legend = base_archetype.get_associated_legend()

        for suffixed_archetype in suffixed_archetypes:
            if base_term != suffixed_archetype.get_base_term():
                raise ValueError(
                    f'Both weapon archetypes must have the same base term ('
                    f'{repr(base_term)} != {repr(suffixed_archetype.get_base_term())}).')

            if associated_legend != suffixed_archetype.get_associated_legend():
                raise ValueError(
                    'Both weapon archetypes must have the same required legend '
                    f'({associated_legend} != {suffixed_archetype.get_associated_legend()}).')

            if suffixed_archetype.get_suffix() is None:
                raise ValueError('with_hopup_archetype must have a suffix.')

        suffixes_and_archetypes: Tuple[Tuple[Suffix, WeaponArchetype], ...] = \
            tuple(sorted(((suffixed_archetype.get_suffix(), suffixed_archetype)
                          for suffixed_archetype in suffixed_archetypes),
                         key=lambda suffix_and_archetype: len(suffix_and_archetype[0]),
                         reverse=True))

        self._base_archetype = base_archetype
        self._suffixes_and_archetypes = suffixes_and_archetypes

        all_archetypes = (base_archetype,) + suffixed_archetypes
        sort_key = lambda a: a.get_best_weapon().get_damage_per_second_body()

        self._best_archetype = max(all_archetypes, key=sort_key)
        self._base_term = base_term
        self._associated_legend = associated_legend

        any_suffix_is_hopped_up = any(self.is_hopped_up_suffix(suffix)
                                      for suffix, _ in suffixes_and_archetypes)
        fully_kitted_archetypes = [archetype for suffix, archetype in suffixes_and_archetypes]
        if not any_suffix_is_hopped_up:
            fully_kitted_archetypes.append(base_archetype)
        self._fully_kitted_archetypes = tuple(sorted(fully_kitted_archetypes,
                                                     key=sort_key,
                                                     reverse=True))

    @staticmethod
    def is_hopped_up_suffix(suffix: Suffix):
        check_type(Suffix, suffix=suffix)
        return SuffixedArchetypeType.HOPPED_UP in suffix.get_types()

    def get_associated_legend(self) -> Optional[Legend]:
        return self._associated_legend

    def _get_archetype(self,
                       words: Words,
                       overall_level: OverallLevel) -> WeaponArchetype:
        check_type(Words, words=words)
        if overall_level in (OverallLevel.FULLY_KITTED, OverallLevel.LEVEL_3):
            return self._best_archetype

        for suffix, suffixed_archetype in self._suffixes_and_archetypes:
            if WeaponArchetype.find_suffix(suffix, words):
                return suffixed_archetype

        return self._base_archetype

    def get_base_term(self) -> RequiredTerm:
        return self._base_term

    def get_fully_kitted_weapons(self, legend: Optional[Legend] = None) -> \
            Generator[Weapon, None, None]:
        for archetype in self._fully_kitted_archetypes:
            yield archetype.get_best_weapon(legend=legend)

    def get_best_match(self,
                       words: Words,
                       overall_level: OverallLevel = OverallLevel.PARSE_WORDS,
                       legend: Optional[Legend] = None) -> \
            TranslatedValue[Weapon]:
        archetype = self._get_archetype(words, overall_level=overall_level)
        return archetype.get_best_match(words=words,
                                        overall_level=overall_level,
                                        legend=legend)

    @staticmethod
    def group_archetypes(archetypes: Iterable[WeaponArchetype]) -> \
            Tuple['WeaponArchetypes', ...]:
        base_archetypes: Dict[RequiredTerm, WeaponArchetype] = {}
        suffixed_archetypes: Dict[RequiredTerm, List[WeaponArchetype]] = {}

        for archetype in archetypes:
            check_type(WeaponArchetype, archetype=archetype)
            base_term = archetype.get_base_term()
            hopup_suffix = archetype.get_suffix()
            if hopup_suffix is None:
                if base_term in base_archetypes:
                    raise RuntimeError(
                        f'Duplicate base weapon archetype base term found: {base_term}')
                base_archetypes[base_term] = archetype
            elif base_term not in suffixed_archetypes:
                suffixed_archetypes[base_term] = [archetype]
            elif archetype in suffixed_archetypes[base_term]:
                raise RuntimeError(
                    f'Duplicate suffixed weapon archetype base term found: {base_term}')
            else:
                suffixed_archetypes[base_term].append(archetype)

        archetype_groups: list[WeaponArchetypes] = []
        for base_term, base_archetype in base_archetypes.items():
            suffixed_archetypes_list = suffixed_archetypes.pop(base_term, [])
            archetype_groups.append(WeaponArchetypes(
                base_archetype=base_archetype,
                suffixed_archetypes=tuple(suffixed_archetypes_list)))

        if len(suffixed_archetypes) > 0:
            raise RuntimeError(
                'Suffixed weapons found with no non-hopped-up equivalents: '
                f'{set(suffixed_archetypes.values())}')

        return tuple(archetype_groups)

    def __repr__(self):
        return repr(self._base_term)

    def __str__(self):
        return str(self._base_term)
