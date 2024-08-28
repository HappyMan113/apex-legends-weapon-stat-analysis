import abc
import logging
import math
import re
from enum import StrEnum
from types import MappingProxyType
from typing import (AbstractSet,
                    Dict,
                    FrozenSet,
                    Generator,
                    Generic,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Tuple,
                    TypeVar,
                    Union,
                    final)

import numpy as np
from numpy.typing import NDArray

from apex_assistant.checker import (check_bool,
                                    check_equal_length,
                                    check_float,
                                    check_float_vec,
                                    check_int,
                                    check_str,
                                    check_tuple,
                                    check_type,
                                    to_kwargs)
from apex_assistant.legend import Legend
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_terms import (ALL_BOLT_TERMS,
                                              ALL_MAG_TERMS,
                                              ALL_STOCK_TERMS,
                                              BASE,
                                              LEVEL_TERMS,
                                              WITH_SIDEARM)
from apex_assistant.speech.suffix import Suffix, SuffixedArchetypeType
from apex_assistant.speech.term import RequiredTerm, Words
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


class StockStatsBase(StatsBase[StockStatValues]):
    pass


class StockStat(StockStatsBase):
    def __init__(self, stats: StockStatValues):
        check_type(StockStatValues, stats=stats)
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


_HUMAN_REACTION_TIME_SECONDS = 0.2


class RoundTiming(abc.ABC):
    @abc.abstractmethod
    def get_num_rounds_shot(self,
                            base_weapon: 'Weapon',
                            time_seconds: float,
                            tactical: int) -> int:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_nums_rounds_shot(self,
                             base_weapon: 'Weapon',
                             times_seconds: NDArray[np.float64],
                             tactical: int) -> NDArray[np.integer]:
        raise NotImplementedError('Must implement.')

    @final
    def get_cumulative_damage(self,
                              base_weapon: 'Weapon',
                              time_seconds: float,
                              tactical: int) -> float:
        num_rounds = self.get_num_rounds_shot(base_weapon=base_weapon,
                                              time_seconds=time_seconds,
                                              tactical=tactical)
        if num_rounds < 0:
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a non-negative number of rounds.')
        if num_rounds > base_weapon.get_magazine_capacity(tactical):
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a number of rounds greater than the magazine capacity.')
        return base_weapon.get_damage_per_round() * num_rounds

    @final
    def get_cumulative_damage_vec(self,
                                  base_weapon: 'Weapon',
                                  times_seconds: NDArray[np.float64],
                                  tactical: int) -> NDArray[np.float64]:
        nums_rounds = self.get_nums_rounds_shot(base_weapon=base_weapon,
                                                times_seconds=times_seconds,
                                                tactical=tactical)
        if (nums_rounds < 0).any():
            raise RuntimeError(
                f'{self.get_nums_rounds_shot.__name__}() in {self.__class__.__name__} must not '
                'return a non-negative number of rounds.')
        if (nums_rounds > base_weapon.get_magazine_capacity(tactical)).any():
            raise RuntimeError(
                f'{self.get_nums_rounds_shot.__name__}() in {self.__class__.__name__} must not '
                'return a number of rounds greater than the magazine capacity.')
        return base_weapon.get_damage_per_round() * nums_rounds

    @abc.abstractmethod
    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: int) -> float:
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

    def _get_shot_times_seconds(self, base_weapon: 'Weapon', tactical: int) -> \
            np.ndarray[np.float64]:
        # As a sanity check, magazine duration for Care Package Devotion was measured as ~227-229
        # frames at 60 FPS (~3.783-3.817 seconds).
        magazine_capacity = base_weapon.get_magazine_capacity(tactical)
        rps_initial = self._rounds_per_second_initial
        rps_final = base_weapon.get_rounds_per_second()
        rps_per_sec = (rps_final - rps_initial) / self._spinup_time_seconds

        times: np.ndarray[np.float64] = np.empty(magazine_capacity)
        times[0] = 0
        for round_num in range(1, magazine_capacity):
            prev_t = times[round_num - 1]
            rps_t = min(rps_final, rps_initial + rps_per_sec * prev_t)
            seconds_per_round_t = 1 / rps_t
            times[round_num] = prev_t + seconds_per_round_t

        return times

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float, tactical: int) -> int:
        shot_times_seconds = self._get_shot_times_seconds(base_weapon=base_weapon,
                                                          tactical=tactical)
        return np.count_nonzero(time_seconds >= shot_times_seconds)

    def get_nums_rounds_shot(self,
                             base_weapon: 'Weapon',
                             times_seconds: NDArray[np.float64],
                             tactical: int) -> NDArray[np.integer]:
        shot_times_seconds = self._get_shot_times_seconds(base_weapon=base_weapon,
                                                          tactical=tactical)
        nums_rounds_shot = np.count_nonzero(times_seconds.reshape(-1, 1) >=
                                            shot_times_seconds.reshape(1, -1), axis=1)
        assert nums_rounds_shot.shape == times_seconds.shape
        return nums_rounds_shot

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: int) -> float:
        shot_times = self._get_shot_times_seconds(base_weapon=base_weapon, tactical=tactical)
        return float(shot_times[-1])

    def __hash__(self):
        return hash(self.__class__) ^ hash((self._rounds_per_minute_initial,
                                            self._spinup_time_seconds))

    def __eq__(self, other):
        return (isinstance(other, RoundTimingDevotion) and
                self._rounds_per_minute_initial == other._rounds_per_minute_initial and
                self._spinup_time_seconds == other._spinup_time_seconds)


class _RoundTimingBurstBase(RoundTiming, abc.ABC):
    def __init__(self, rounds_per_burst: int):
        super().__init__()
        check_int(min_value=2, rounds_per_burst=rounds_per_burst)
        self._rounds_per_burst = rounds_per_burst

    def _get_burst_nums_rounds(self, base_weapon: 'Weapon', tactical: int) -> \
            np.ndarray[np.integer, ...]:
        num_full_bursts, remainder_rounds = divmod(
            base_weapon.get_magazine_capacity(tactical),
            self._rounds_per_burst)
        nums_rounds = [self._rounds_per_burst] * num_full_bursts
        if remainder_rounds > 0:
            nums_rounds.append(remainder_rounds)
        return np.array(nums_rounds, dtype=int)

    @staticmethod
    def _get_burst_duration_seconds(base_weapon: 'Weapon',
                                    num_rounds: NDArray[np.integer]) -> NDArray[np.float64]:
        seconds_per_round = 1 / base_weapon.get_rounds_per_second()
        burst_duration_seconds = seconds_per_round * (num_rounds - 1)
        return burst_duration_seconds

    def _get_burst_durations_seconds(self, base_weapon: 'Weapon', tactical: int) -> \
            np.ndarray[np.float64]:
        nums_rounds = self._get_burst_nums_rounds(base_weapon, tactical)
        burst_durations_seconds = self._get_burst_duration_seconds(base_weapon, nums_rounds)
        return burst_durations_seconds

    @abc.abstractmethod
    def _get_burst_periods_seconds(self,
                                   base_weapon: 'Weapon',
                                   tactical: int) -> np.ndarray[np.float64]:
        raise NotImplementedError()

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float, tactical: int) -> int:
        check_float(min_value=0, time_seconds=time_seconds)
        periods_seconds = self._get_burst_periods_seconds(base_weapon, tactical=tactical)
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
                   base_weapon.get_magazine_capacity(tactical))

    def get_nums_rounds_shot(self,
                             base_weapon: 'Weapon',
                             times_seconds: NDArray[np.float64],
                             tactical: int) -> NDArray[np.integer]:
        periods_seconds = self._get_burst_periods_seconds(base_weapon, tactical=tactical)
        period_stop_times_seconds = periods_seconds.cumsum()

        num_full_periods_shot = np.count_nonzero(times_seconds.reshape(-1, 1) >=
                                                 period_stop_times_seconds.reshape(1, -1),
                                                 axis=1)
        assert num_full_periods_shot.shape == times_seconds.shape
        full_periods_shot = num_full_periods_shot * self._rounds_per_burst

        partial_period_start_times_secs = np.zeros_like(times_seconds)
        non_zeros = np.flatnonzero(num_full_periods_shot)
        partial_period_start_times_secs[non_zeros] = \
            period_stop_times_seconds[num_full_periods_shot[non_zeros] - 1]
        rel_burst_times_secs = times_seconds - partial_period_start_times_secs
        ge_zeros = np.flatnonzero(rel_burst_times_secs >= 0)
        num_rounds_shot_in_partial_period = np.zeros_like(times_seconds)
        num_rounds_shot_in_partial_period[ge_zeros] = (
                1 + np.floor(rel_burst_times_secs[ge_zeros] * base_weapon.get_rounds_per_second())
        ).clip(max=self._rounds_per_burst).astype(int)

        return (full_periods_shot + num_rounds_shot_in_partial_period).clip(
            max=base_weapon.get_magazine_capacity(tactical))

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: int) -> float:
        periods_seconds = self._get_burst_periods_seconds(base_weapon, tactical=tactical)
        return float(periods_seconds.sum())

    def __hash__(self) -> int:
        return hash(self.__class__) ^ hash(self._rounds_per_burst)

    def __eq__(self, other):
        return (isinstance(other, _RoundTimingBurstBase) and
                other._rounds_per_burst == self._rounds_per_burst)


class RoundTimingBurst(_RoundTimingBurstBase):
    def __init__(self, rounds_per_burst: int, burst_fire_delay: float):
        super().__init__(rounds_per_burst=rounds_per_burst)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_fire_delay=burst_fire_delay)

        self._burst_fire_delay = burst_fire_delay

    def _get_burst_periods_seconds(self,
                                   base_weapon: 'Weapon',
                                   tactical: int) -> np.ndarray[np.float64]:
        burst_periods_periods = self._get_burst_durations_seconds(base_weapon=base_weapon,
                                                                  tactical=tactical)
        burst_periods_periods[:-1] += self._burst_fire_delay
        return burst_periods_periods

    def __hash__(self):
        return super().__hash__() ^ hash(self._burst_fire_delay)

    def __eq__(self, other):
        return (super().__eq__(other) and
                isinstance(other, RoundTimingBurst) and
                self._burst_fire_delay == other._burst_fire_delay)


class RoundTimingNemesis(_RoundTimingBurstBase):
    def __init__(self,
                 rounds_per_burst: int,
                 burst_fire_delay_initial: float,
                 burst_fire_delay_final: float,
                 burst_charge_fraction: float):
        super().__init__(rounds_per_burst=rounds_per_burst)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_fire_delay_final=burst_fire_delay_final)
        check_float(min_value=burst_fire_delay_final,
                    min_is_exclusive=True,
                    burst_fire_delay_initial=burst_fire_delay_initial)
        check_float(min_value=0,
                    min_is_exclusive=True,
                    burst_charge_fraction=burst_charge_fraction)

        self._burst_fire_delay_initial = burst_fire_delay_initial
        self._burst_fire_delay_final = burst_fire_delay_final
        self._burst_charge_fraction = burst_charge_fraction

    def _get_burst_periods_seconds(self,
                                   base_weapon: 'Weapon',
                                   tactical: int) -> np.ndarray[np.float64]:
        burst_durations_seconds = self._get_burst_durations_seconds(base_weapon=base_weapon,
                                                                    tactical=tactical)
        num_bursts = len(burst_durations_seconds)
        if num_bursts == 0:
            return burst_durations_seconds

        charge_fractions = (np.arange(num_bursts - 1) * self._burst_charge_fraction).clip(max=1)
        burst_fire_delays = ((1 - charge_fractions) * self._burst_fire_delay_initial +
                             charge_fractions * self._burst_fire_delay_final)

        # Delay after last burst is an elephant.
        burst_fire_delays = np.append(burst_fire_delays, 0)

        return burst_durations_seconds + burst_fire_delays

    def __hash__(self):
        return super().__hash__() ^ hash((self._burst_fire_delay_initial,
                                          self._burst_fire_delay_final,
                                          self._burst_charge_fraction))

    def __eq__(self, other):
        return (super().__eq__(other) and
                isinstance(other, RoundTimingNemesis) and
                self._burst_fire_delay_initial == other._burst_fire_delay_initial and
                self._burst_fire_delay_final == other._burst_fire_delay_final and
                self._burst_charge_fraction == other._burst_charge_fraction)


class RoundTimingRegular(RoundTiming):
    _INSTANCE: Optional['RoundTimingRegular'] = None
    __create_key = object()

    def __init__(self, create_key):
        super().__init__()
        if create_key != RoundTimingRegular.__create_key:
            raise RuntimeError(f'Cannot instantiate {RoundTimingRegular.__name__} except from '
                               'get_instance() method.')

    @staticmethod
    def get_instance():
        if RoundTimingRegular._INSTANCE is None:
            RoundTimingRegular._INSTANCE = RoundTimingRegular(RoundTimingRegular.__create_key)
        return RoundTimingRegular._INSTANCE

    def get_magazine_duration_seconds(self, base_weapon: 'Weapon', tactical: int) -> float:
        check_type(Weapon, base_weapon=base_weapon)
        check_int(min_value=0, tactical=tactical)
        num_rounds = base_weapon.get_magazine_capacity(tactical)
        if num_rounds == 0:
            return 0

        num_delays = num_rounds - 1
        round_delay_seconds = 1 / base_weapon.get_rounds_per_second()
        magazine_duration_seconds = num_delays * round_delay_seconds
        return magazine_duration_seconds

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float, tactical: int) -> int:
        check_float(min_value=0, time_seconds=time_seconds)
        rounds_per_second = base_weapon.get_rounds_per_second()
        num_rounds = min(1 + math.floor(time_seconds * rounds_per_second),
                         base_weapon.get_magazine_capacity(tactical))
        return num_rounds

    def get_nums_rounds_shot(self,
                             base_weapon: 'Weapon',
                             times_seconds: NDArray[np.float64],
                             tactical: int) -> NDArray[np.integer]:
        rounds_per_second = base_weapon.get_rounds_per_second()
        num_rounds = (1 + np.floor(times_seconds * rounds_per_second)).clip(
            max=base_weapon.get_magazine_capacity(tactical)).astype(int)
        return num_rounds

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, RoundTimingRegular)


class RoundTimingHavoc(RoundTiming):
    _NO_SPINUP = RoundTimingRegular.get_instance()

    def __init__(self, spinup_time_seconds: float):
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)
        self._spinup_time_seconds = spinup_time_seconds

    def get_magazine_duration_seconds(self,
                                      base_weapon: 'Weapon',
                                      tactical: int) -> float:
        return (self._spinup_time_seconds +
                self._NO_SPINUP.get_magazine_duration_seconds(base_weapon=base_weapon,
                                                              tactical=tactical))

    def get_num_rounds_shot(self, base_weapon: 'Weapon', time_seconds: float, tactical: int) -> int:
        check_float(min_value=0, time_seconds=time_seconds)
        spinup_time_seconds = self._spinup_time_seconds
        if time_seconds < spinup_time_seconds:
            return 0

        return self._NO_SPINUP.get_num_rounds_shot(
            base_weapon=base_weapon,
            time_seconds=time_seconds - spinup_time_seconds,
            tactical=tactical)

    def get_nums_rounds_shot(self,
                             base_weapon: 'Weapon',
                             times_seconds: NDArray[np.float64],
                             tactical: int) -> NDArray[np.integer]:
        spinup_time_seconds = self._spinup_time_seconds
        result = np.zeros_like(times_seconds, dtype=int)
        non_zeros = np.flatnonzero(times_seconds >= spinup_time_seconds)
        result[non_zeros] = self._NO_SPINUP.get_nums_rounds_shot(
            base_weapon=base_weapon,
            times_seconds=times_seconds[non_zeros] - spinup_time_seconds,
            tactical=tactical)
        return result

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._spinup_time_seconds)

    def __eq__(self, other):
        return (isinstance(other, RoundTimingHavoc) and
                other._spinup_time_seconds == self._spinup_time_seconds)


class Loadout(abc.ABC):
    @abc.abstractmethod
    def get_term(self) -> RequiredTerm:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    def __repr__(self):
        return self.get_name()

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()


class FullLoadoutConfiguration(StrEnum):
    A = 'A'
    AB = 'AB'
    ABA = 'ABA'
    A1B = 'A1B'
    A1BA = 'A1BA'
    B = 'B'
    BA = 'BA'
    BAB = 'BAB'
    B1A = 'B1A'
    B1AB = 'B1AB'

    __PATT = re.compile(r'^[AB]([AB1]([AB][AB]?)?)?$')
    __SINGLE_SHOT_CHAR = '1'
    __INVERT_TABLE = str.maketrans('AB', 'BA')

    def __init__(self, configuration_string: str):
        check_str(allow_blank=False, configuration_string=configuration_string)
        if re.match(self.__PATT, configuration_string) is None:
            raise ValueError(f'configuration_string {configuration_string} must match pattern: '
                             f'{self.__PATT}.')

    @staticmethod
    def _get_char_to_weapon_dict(weapon_a: 'Weapon', weapon_b: 'Weapon') -> Mapping[str, 'Weapon']:
        return MappingProxyType({'A': weapon_a, 'B': weapon_b})

    def is_advisable(self, weapon_a: 'Weapon', weapon_b: 'Weapon') -> bool:
        configuration_string = self._get_configuration_str()
        if len(configuration_string) == 1:
            return True

        char_to_weapon_dict = self._get_char_to_weapon_dict(weapon_a, weapon_b)
        for cur_char, next_char in zip(configuration_string[:-1], configuration_string[1:]):
            if next_char != self.__SINGLE_SHOT_CHAR:
                continue

            cur_weapon = char_to_weapon_dict.get(cur_char, None)
            if cur_weapon is None:
                raise ValueError('\'1\' must be specified only after \'A\' or \'B\'')

            if not cur_weapon.is_single_shot_advisable():
                return False

        return True

    def should_invert(self) -> bool:
        return self._get_configuration_str()[0] != 'A'

    def invert(self) -> 'FullLoadoutConfiguration':
        table = FullLoadoutConfiguration.__INVERT_TABLE
        config = self._get_configuration_str().translate(table)
        return FullLoadoutConfiguration(config)

    # noinspection DuplicatedCode
    def get_cumulative_damage(self,
                              weapon_a: 'Weapon',
                              weapon_b: 'Weapon',
                              time_seconds: float,
                              distance_meters: float) -> float:
        check_float(min_value=0, time_seconds=time_seconds, distance_meters=distance_meters)
        prev_weapon_and_char: Optional[Tuple['SingleWeaponLoadout', str]] = None

        damage: float = 0
        configuration_string = self._get_configuration_str()
        char_to_weapon_dict = self._get_char_to_weapon_dict(weapon_a, weapon_b)

        for index, cur_char in enumerate(configuration_string):
            cur_weapon = char_to_weapon_dict.get(cur_char, None)
            if cur_weapon is None:
                if index != 1:
                    raise ValueError('\'1\' can only come at index 1.')

                if cur_char != self.__SINGLE_SHOT_CHAR:
                    raise ValueError('Must contain only \'A\', \'B\', and \'1\'.')

                if prev_weapon_and_char is None:
                    raise ValueError('\'1\' can only come after \'A\' or \'B\'')

                prev_weapon, prev_char = prev_weapon_and_char
                if not isinstance(prev_weapon, Weapon):
                    raise ValueError('Cannot say single shot multiple times.')

                if not prev_weapon.is_single_shot_advisable():
                    raise ValueError(f'Single shot configuration {self} is not advisable.')

                cur_weapon = prev_weapon.single_shot()
                cur_char = prev_char

            elif prev_weapon_and_char is not None:
                prev_weapon, prev_char = prev_weapon_and_char
                if cur_char == prev_char:
                    raise ValueError('Cannot have repeat weapons. Makes no sense.')

                prev_idx = index - 1
                will_reload = prev_weapon.can_reload() and self._want_to_reload(prev_idx)
                used_single_shot = self._used_single_shot(prev_idx)
                tactical = int(will_reload) + int(used_single_shot)

                damage += prev_weapon.get_single_mag_cumulative_damage(
                    time_seconds=time_seconds,
                    distance_meters=distance_meters,
                    tactical=tactical)
                assert isinstance(damage, (int, float))
                time_seconds -= (prev_weapon.get_magazine_duration_seconds(tactical) +
                                 _HUMAN_REACTION_TIME_SECONDS +
                                 prev_weapon.get_swap_time_seconds(cur_weapon))
                if time_seconds < 0:
                    return damage

            prev_weapon_and_char = cur_weapon, cur_char

        if prev_weapon_and_char is not None:
            prev_weapon, _ = prev_weapon_and_char
            prev_index = len(configuration_string) - 1
            reload_time_secs = prev_weapon.get_weapon().get_tactical_reload_time_secs()
            will_reload = reload_time_secs is not None
            used_single_shot = self._used_single_shot(prev_index)
            if used_single_shot:
                tactical = int(will_reload) + 1
                damage += prev_weapon.get_single_mag_cumulative_damage(
                    time_seconds=time_seconds,
                    distance_meters=distance_meters,
                    tactical=tactical)
                assert isinstance(damage, (int, float))
                if not will_reload:
                    return damage

                time_seconds -= (prev_weapon.get_magazine_duration_seconds(tactical=tactical) +
                                 _HUMAN_REACTION_TIME_SECONDS +
                                 reload_time_secs)
                if time_seconds < 0:
                    return damage

            if not will_reload:
                damage += prev_weapon.get_single_mag_cumulative_damage(
                    time_seconds=time_seconds,
                    distance_meters=distance_meters,
                    tactical=0)
            else:
                tactical = 1
                mag_duration_seconds = prev_weapon.get_magazine_duration_seconds(tactical=tactical)
                period_seconds = (mag_duration_seconds +
                                  _HUMAN_REACTION_TIME_SECONDS +
                                  reload_time_secs)
                num_completed_cycles, rem_time_seconds = divmod(time_seconds, period_seconds)
                full_mag_cum_damage = prev_weapon.get_single_mag_cumulative_damage(
                    time_seconds=mag_duration_seconds,
                    distance_meters=distance_meters,
                    tactical=tactical)
                rem_mag_cum_damage = prev_weapon.get_single_mag_cumulative_damage(
                    time_seconds=rem_time_seconds,
                    distance_meters=distance_meters,
                    tactical=tactical)

                damage += num_completed_cycles * full_mag_cum_damage + rem_mag_cum_damage

            assert isinstance(damage, (int, float))

        return damage

    # noinspection DuplicatedCode
    def get_cumulative_damage_vec(self,
                                  weapon_a: 'Weapon',
                                  weapon_b: 'Weapon',
                                  times_seconds: NDArray[np.float64],
                                  distances_meters: NDArray[np.float64]) -> NDArray[np.float64]:
        check_float_vec(min_value=0, times_seconds=times_seconds, distances_meters=distances_meters)
        times_seconds = times_seconds.copy()

        prev_weapon_and_char: Optional[Tuple['SingleWeaponLoadout', str]] = None

        damages: NDArray[np.float64] = np.zeros_like(times_seconds)
        valid_indices: NDArray[np.integer] = np.arange(len(times_seconds))
        configuration_string = self._get_configuration_str()
        char_to_weapon_dict = self._get_char_to_weapon_dict(weapon_a, weapon_b)

        for index, cur_char in enumerate(configuration_string):
            cur_weapon = char_to_weapon_dict.get(cur_char, None)
            if cur_weapon is None:
                if index != 1:
                    raise ValueError('\'1\' can only come at index 1.')

                if cur_char != self.__SINGLE_SHOT_CHAR:
                    raise ValueError('Must contain only \'A\', \'B\', and \'1\'.')

                if prev_weapon_and_char is None:
                    raise ValueError('\'1\' can only come after \'A\' or \'B\'')

                prev_weapon, prev_char = prev_weapon_and_char
                if not isinstance(prev_weapon, Weapon):
                    raise ValueError('Cannot say single shot multiple times.')

                if not prev_weapon.is_single_shot_advisable():
                    raise ValueError(f'Single shot configuration {self} is not advisable.')

                cur_weapon = prev_weapon.single_shot()
                cur_char = prev_char

            elif prev_weapon_and_char is not None:
                prev_weapon, prev_char = prev_weapon_and_char
                if cur_char == prev_char:
                    raise ValueError('Cannot have repeat weapons. Makes no sense.')

                prev_idx = index - 1
                will_reload = prev_weapon.can_reload() and self._want_to_reload(prev_idx)
                used_single_shot = self._used_single_shot(prev_idx)
                tactical = int(will_reload) + int(used_single_shot)

                damages[valid_indices] += prev_weapon.get_single_mag_cumulative_damage_vec(
                    times_seconds=times_seconds[valid_indices],
                    distances_meters=distances_meters[valid_indices],
                    tactical=tactical)
                times_seconds -= (prev_weapon.get_magazine_duration_seconds(tactical) +
                                  _HUMAN_REACTION_TIME_SECONDS +
                                  prev_weapon.get_swap_time_seconds(cur_weapon))
                valid_indices = np.flatnonzero(times_seconds >= 0)
                if len(valid_indices) == 0:
                    return damages

            prev_weapon_and_char = cur_weapon, cur_char

        if prev_weapon_and_char is not None:
            prev_weapon, _ = prev_weapon_and_char
            prev_index = len(configuration_string) - 1
            reload_time_secs = prev_weapon.get_weapon().get_tactical_reload_time_secs()
            will_reload = reload_time_secs is not None
            used_single_shot = self._used_single_shot(prev_index)
            if used_single_shot:
                tactical = int(will_reload) + 1
                damages[valid_indices] += prev_weapon.get_single_mag_cumulative_damage_vec(
                    times_seconds=times_seconds[valid_indices],
                    distances_meters=distances_meters[valid_indices],
                    tactical=tactical)

                if not will_reload:
                    return damages

                times_seconds -= (prev_weapon.get_magazine_duration_seconds(tactical=tactical) +
                                  _HUMAN_REACTION_TIME_SECONDS +
                                  reload_time_secs)
                valid_indices = np.flatnonzero(times_seconds >= 0)
                if len(valid_indices) == 0:
                    return damages

            if not will_reload:
                damages[valid_indices] += prev_weapon.get_single_mag_cumulative_damage_vec(
                    times_seconds=times_seconds[valid_indices],
                    distances_meters=distances_meters[valid_indices],
                    tactical=0)
            else:
                tactical = 1
                mag_duration_seconds = prev_weapon.get_magazine_duration_seconds(tactical=tactical)
                period_seconds = (mag_duration_seconds +
                                  _HUMAN_REACTION_TIME_SECONDS +
                                  reload_time_secs)
                nums_completed_cycles, rem_times_seconds = divmod(times_seconds[valid_indices],
                                                                  period_seconds)
                full_mag_cum_damage = prev_weapon.get_single_mag_cumulative_damage_vec(
                    times_seconds=np.full_like(valid_indices,
                                               fill_value=mag_duration_seconds,
                                               dtype=np.float64),
                    distances_meters=distances_meters[valid_indices],
                    tactical=tactical)
                rem_mag_cum_damage = prev_weapon.get_single_mag_cumulative_damage_vec(
                    times_seconds=rem_times_seconds,
                    distances_meters=distances_meters[valid_indices],
                    tactical=tactical)

                damages[valid_indices] += (nums_completed_cycles * full_mag_cum_damage +
                                           rem_mag_cum_damage)

        return damages

    def _want_to_reload(self, index: int) -> bool:
        configuration_string = self._get_configuration_str()
        if index == len(configuration_string) - 1:
            return True
        char = configuration_string[index]
        return char in configuration_string[index + 1:]

    def _used_single_shot(self, index: int) -> bool:
        configuration_string = self._get_configuration_str()
        char = configuration_string[index]
        prev_found_idx = configuration_string[:index].rfind(char)
        return (prev_found_idx != -1 and
                configuration_string[prev_found_idx + 1] == self.__SINGLE_SHOT_CHAR)

    def _get_configuration_str(self) -> str:
        return str(self)


CONFIGURATIONS: Tuple[FullLoadoutConfiguration, ...] = tuple(FullLoadoutConfiguration)


def invert_config_idx(config_idx: T) -> T:
    num_configs = len(CONFIGURATIONS)
    return (config_idx + num_configs // 2) % num_configs


def __validate():
    """Make sure inverting the config index gets the index of the inverted config."""
    for idx, config in enumerate(CONFIGURATIONS):
        inv_idx = invert_config_idx(idx)
        assert config.invert() == CONFIGURATIONS[inv_idx], (
            f'{config}.invert() ({config.invert()}) != CONFIGURATIONS[{inv_idx}] '
            f'({CONFIGURATIONS[inv_idx]})')


__validate()


class FullLoadout(Loadout):
    NUM_WEAPONS = 2

    # Arbitrary.
    _CONFIG_FIGURING_TIME_SECONDS: float = 5
    _MAX_CONFIG_FIGURING_DISTANCE_METERS: int = 300

    def __init__(self, weapon_a: 'Weapon', weapon_b: 'Weapon'):
        check_type(Weapon, weapon_a=weapon_a, weapon_b=weapon_b)
        self._weapons_set = frozenset((weapon_a, weapon_b))

        self._weapon_a: Weapon | None = None
        self._weapon_b: Weapon | None = None
        self._distance_to_config_idx_map: NDArray[np.integer] | None = None
        self._name: str | None = None
        self._term: RequiredTerm | None = None

    def get_term(self) -> RequiredTerm:
        self._lazy_load_internals()
        return self._term

    def get_name(self) -> str:
        self._lazy_load_internals()
        return self._name

    @staticmethod
    def _get_config_description(distance_to_config_idx_map: NDArray[np.integer]) -> str:
        diff_indices = np.flatnonzero(np.diff(distance_to_config_idx_map))
        far_config = CONFIGURATIONS[distance_to_config_idx_map[-1]]
        if len(diff_indices) == 0:
            return far_config

        config_indices = distance_to_config_idx_map[diff_indices]

        configs: List[FullLoadoutConfiguration] = [CONFIGURATIONS[config_idx]
                                                   for config_idx in config_indices]
        configs.append(far_config)

        diff_indices = np.pad(diff_indices,
                              (1, 1),
                              constant_values=(0, len(distance_to_config_idx_map)))
        assert len(diff_indices) == len(configs) + 1

        return ', '.join(f'{start_m}-{stop_m - 1}m: {config}'
                         for config, start_m, stop_m in
                         zip(configs, diff_indices[:-1], diff_indices[1:]))

    def _lazy_load_internals(self):
        if self._distance_to_config_idx_map is not None:
            return

        weapons_iter = iter(self._weapons_set)
        weapon_a = next(weapons_iter)
        weapon_b = next(weapons_iter, weapon_a)

        advisable_configurations: Tuple[Tuple[int, FullLoadoutConfiguration], ...] = tuple(
            (idx, config)
            for idx, config in enumerate(CONFIGURATIONS)
            if config.is_advisable(weapon_a, weapon_b))
        if len(advisable_configurations) == 0:
            raise RuntimeError('No configurations were advisable.')

        close_config: FullLoadoutConfiguration = \
            CONFIGURATIONS[FullLoadout._get_best_configuration_idx(
                weapon_a=weapon_a,
                weapon_b=weapon_b,
                time_seconds=FullLoadout._CONFIG_FIGURING_TIME_SECONDS,
                distance_meters=0,
                advisable_configurations=advisable_configurations)]
        if close_config.should_invert():
            weapon_a, weapon_b = weapon_b, weapon_a
            advisable_configurations = tuple((invert_config_idx(idx), config.invert())
                                             for idx, config in advisable_configurations)

        distance_to_config_idx_map: NDArray[np.integer] = np.full(
            FullLoadout._MAX_CONFIG_FIGURING_DISTANCE_METERS + 1,
            fill_value=-1)
        lower_idx = 1
        upper_idx = len(distance_to_config_idx_map) - 1
        FullLoadout._populate_distance_to_config_idx_map(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            time_seconds=FullLoadout._CONFIG_FIGURING_TIME_SECONDS,
            ranges=distance_to_config_idx_map,
            advisable_configurations=advisable_configurations,
            lower_idx=lower_idx,
            upper_idx=upper_idx)
        first_config_idx = distance_to_config_idx_map[1]
        distance_to_config_idx_map[0] = first_config_idx
        assert not np.any(distance_to_config_idx_map == -1)
        FullLoadout._remove_in_between_configs(distance_to_config_idx_map)

        config_descriptor = FullLoadout._get_config_description(distance_to_config_idx_map)

        self._weapon_a = weapon_a
        self._weapon_b = weapon_b
        self._distance_to_config_idx_map = distance_to_config_idx_map
        self._name = (f'{weapon_a.get_name()} {WITH_SIDEARM} {weapon_b.get_name()} '
                      f'({config_descriptor})')
        self._term = weapon_a.get_term().append(WITH_SIDEARM, weapon_b.get_term())

    @staticmethod
    def _populate_distance_to_config_idx_map(
            weapon_a: 'Weapon',
            weapon_b: 'Weapon',
            time_seconds: float,
            ranges: NDArray[np.integer],
            advisable_configurations: Tuple[Tuple[int, FullLoadoutConfiguration], ...],
            lower_idx: int,
            upper_idx: int):
        if lower_idx > upper_idx:
            # Base case.
            return

        lower_config_idx = FullLoadout._get_best_configuration_idx(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            time_seconds=time_seconds,
            distance_meters=lower_idx,
            advisable_configurations=advisable_configurations)
        ranges[lower_idx] = lower_config_idx

        if lower_idx == upper_idx:
            # No need to get the best configuration for the same index again.
            return

        upper_config_idx = FullLoadout._get_best_configuration_idx(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            time_seconds=time_seconds,
            distance_meters=upper_idx,
            advisable_configurations=advisable_configurations)
        ranges[upper_idx] = upper_config_idx

        if lower_config_idx == upper_config_idx:
            # If same configuration is best for two distances, we can pretty safely assume that the
            # same configuration will be best for any distance in between those two distances.
            ranges[lower_idx + 1:upper_idx] = lower_config_idx
            return

        # We will have to do a kind of binary search to figure out at what distance the best
        # configuration changes.
        mid_idx = round(((math.sqrt(lower_idx) + math.sqrt(upper_idx)) / 2) ** 2)
        FullLoadout._populate_distance_to_config_idx_map(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            time_seconds=time_seconds,
            ranges=ranges,
            advisable_configurations=advisable_configurations,
            lower_idx=lower_idx + 1,
            upper_idx=mid_idx)
        FullLoadout._populate_distance_to_config_idx_map(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            time_seconds=time_seconds,
            ranges=ranges,
            advisable_configurations=advisable_configurations,
            lower_idx=mid_idx + 1,
            upper_idx=upper_idx - 1)

    @staticmethod
    def _get_best_configuration_idx(
            weapon_a: 'Weapon',
            weapon_b: 'Weapon',
            time_seconds: float,
            distance_meters: float,
            advisable_configurations: Tuple[Tuple[int, FullLoadoutConfiguration], ...]) -> int:
        check_float(min_value=0,
                    time_seconds=time_seconds,
                    distance_meters=distance_meters)
        check_tuple(tuple,
                    allow_empty=False,
                    advisable_configurations=advisable_configurations)

        idx, _ = max(advisable_configurations,
                     key=lambda _idx_and_config: _idx_and_config[1].get_cumulative_damage(
                         weapon_a=weapon_a,
                         weapon_b=weapon_b,
                         time_seconds=time_seconds,
                         distance_meters=distance_meters))
        return idx

    @staticmethod
    def _remove_in_between_configs(distance_to_config_idx_map: NDArray[np.integer]):
        diff_indices = np.flatnonzero(np.diff(distance_to_config_idx_map, prepend=-1, append=-1))

        for sl_start_idx, sl_stop_idx in zip(diff_indices[:-1], diff_indices[1:]):
            config_idx = distance_to_config_idx_map[sl_start_idx]
            where_it_again = np.flatnonzero(distance_to_config_idx_map[sl_stop_idx:] == config_idx)
            if len(where_it_again) != 0:
                where_it_again_last = where_it_again[-1]
                distance_to_config_idx_map[sl_stop_idx:
                                           sl_stop_idx + where_it_again_last] = config_idx

    def _get_best_config_for_distance(self, distance_meters: float) -> FullLoadoutConfiguration:
        distance_int = min(round(distance_meters), len(self._distance_to_config_idx_map) - 1)
        config_idx: int = self._distance_to_config_idx_map[distance_int]
        return CONFIGURATIONS[config_idx]

    def _get_best_config_indices_for_distances_vec(
            self,
            distances_meters: NDArray[np.float64]) -> NDArray[np.integer]:
        self._lazy_load_internals()

        distances_int = distances_meters.round().clip(
            max=len(self._distance_to_config_idx_map) - 1).astype(int)
        config_indices: NDArray[np.integer] = self._distance_to_config_idx_map[distances_int]
        return config_indices

    def get_cumulative_damage(self, time_seconds: float, distance_meters: float) -> float:
        check_float(min_value=0, time_seconds=time_seconds, distance_meters=distance_meters)
        self._lazy_load_internals()

        configuration = self._get_best_config_for_distance(distance_meters)
        return configuration.get_cumulative_damage(weapon_a=self._weapon_a,
                                                   weapon_b=self._weapon_b,
                                                   time_seconds=time_seconds,
                                                   distance_meters=distance_meters)

    def get_cumulative_damage_vec(self,
                                  times_seconds: NDArray[np.float64],
                                  distances_meters: NDArray[np.float64]) -> NDArray[np.float64]:
        if (times_seconds < 0).any():
            raise ValueError('No time values can be negative.')
        if (distances_meters < 0).any():
            raise ValueError('No distance values can be negative.')

        self._lazy_load_internals()

        sorti = distances_meters.argsort()
        sorti_inv = sorti.argsort()
        times_seconds = times_seconds[sorti]
        distances_meters = distances_meters[sorti]

        config_indices = self._get_best_config_indices_for_distances_vec(distances_meters)

        diff_indices = np.flatnonzero(np.diff(config_indices, prepend=-1, append=-1))
        weapon_a = self._weapon_a
        weapon_b = self._weapon_b
        result = np.full_like(distances_meters, fill_value=-1)

        for start_idx, stop_idx in zip(diff_indices[:-1], diff_indices[1:]):
            sl = slice(start_idx, stop_idx)

            times_slice = times_seconds[sl]
            distances_slice = distances_meters[sl]
            configs_slice = config_indices[sl]

            assert not np.diff(configs_slice).any()
            config: FullLoadoutConfiguration = CONFIGURATIONS[configs_slice[0]]
            result[sl] = config.get_cumulative_damage_vec(weapon_a=weapon_a,
                                                          weapon_b=weapon_b,
                                                          times_seconds=times_slice,
                                                          distances_meters=distances_slice)
        assert (result >= 0).all()

        result = result[sorti_inv]
        return result

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._weapons_set)

    def __eq__(self, other):
        return isinstance(other, FullLoadout) and hash(self._weapons_set)

    def get_weapons(self) -> FrozenSet['Weapon']:
        return self._weapons_set

    @staticmethod
    def get_loadouts(required_weapons: Iterable['Weapon']) -> Generator['FullLoadout', None, None]:
        duplicated_weapons = FullLoadout._get_duplicates(required_weapons)
        return (FullLoadout(weapon_a, weapon_b)
                for weapon_a in required_weapons
                for weapon_b in required_weapons
                if weapon_b != weapon_a or weapon_a in duplicated_weapons)

    @staticmethod
    def filter_loadouts(loadouts: Iterable['FullLoadout'],
                        weapons_to_exclude: AbstractSet['Weapon']) -> \
            Generator['FullLoadout', None, None]:
        return (loadout
                for loadout in loadouts
                if loadout.get_weapons().isdisjoint(weapons_to_exclude))

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

    def get_weapon_a(self) -> 'Weapon':
        self._lazy_load_internals()
        return self._weapon_a

    def get_weapon_b(self):
        self._lazy_load_internals()
        return self._weapon_b


class SingleWeaponLoadout(Loadout, abc.ABC):
    def __init__(self, name: str, term: RequiredTerm):
        check_str(allow_blank=False, name=name)
        check_type(RequiredTerm, term=term)
        self._name = name
        self._term = term

    def get_term(self) -> RequiredTerm:
        return self._term

    def get_name(self) -> str:
        return self._name

    @abc.abstractmethod
    def get_archetype(self) -> 'WeaponArchetype':
        raise NotImplementedError()

    @abc.abstractmethod
    def get_magazine_duration_seconds(self, tactical: int) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_weapon(self) -> 'Weapon':
        raise NotImplementedError()

    @abc.abstractmethod
    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float,
                                         tactical: int) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_single_mag_cumulative_damage_vec(self,
                                             times_seconds: NDArray[np.float64],
                                             distances_meters: NDArray[np.float64],
                                             tactical: int) -> NDArray[np.float64]:
        raise NotImplementedError()

    def get_swap_time_seconds(self, weapon_swap_to: 'SingleWeaponLoadout') -> float:
        check_type(SingleWeaponLoadout, weapon_swap_to=weapon_swap_to)
        return self.get_holster_time_secs() + weapon_swap_to.get_ready_to_fire_time_secs()

    @abc.abstractmethod
    def get_holster_time_secs(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ready_to_fire_time_secs(self) -> float:
        raise NotImplementedError()

    def can_reload(self):
        return self.get_weapon().get_tactical_reload_time_secs() is not None


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

        super().__init__(name=name, term=term)
        self.archetype = archetype
        self.weapon_class = weapon_class
        self.eighty_percent_accuracy_range = eighty_percent_accuracy_range
        self.damage_body = damage_body
        self.holster_time_secs = holster_time_secs
        self.ready_to_fire_time_secs = ready_to_fire_time_secs
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.round_timing = spinup
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
                                         distance_meters: float,
                                         tactical: int) -> float:
        return (self.round_timing.get_cumulative_damage(base_weapon=self,
                                                        time_seconds=time_seconds,
                                                        tactical=tactical) *
                self._get_accuracy_fraction(distance_meters))

    def get_single_mag_cumulative_damage_vec(self,
                                             times_seconds: NDArray[np.float64],
                                             distances_meters: NDArray[np.float64],
                                             tactical: int) -> NDArray[np.float64]:
        return (self.round_timing.get_cumulative_damage_vec(base_weapon=self,
                                                            times_seconds=times_seconds,
                                                            tactical=tactical) *
                self._get_accuracy_fraction_vec(distances_meters))

    def _get_accuracy_fraction(self, distance_meters: float) -> float:
        check_float(min_value=0, distance_meters=distance_meters)
        if distance_meters == 0:
            return 1
        return min(self.EIGHTY_PERCENT * self.eighty_percent_accuracy_range / distance_meters, 1)

    def _get_accuracy_fraction_vec(self, distances_meters: NDArray[np.float64]) -> \
            NDArray[np.float64]:
        result = np.ones_like(distances_meters)
        non_zeros = np.flatnonzero(distances_meters)
        result[non_zeros] = ((1 / distances_meters[non_zeros]) *
                             (self.EIGHTY_PERCENT * self.eighty_percent_accuracy_range)).clip(max=1)
        return result

    def get_magazine_duration_seconds(self, tactical: int) -> float:
        duration_seconds = self.round_timing.get_magazine_duration_seconds(self, tactical=tactical)
        assert isinstance(duration_seconds, (int, float))
        return duration_seconds

    def get_damage_per_second_body(self) -> float:
        return self._damage_per_second_body

    def get_eighty_percent_accuracy_range(self) -> int:
        return self.eighty_percent_accuracy_range

    def get_damage_per_round(self) -> float:
        return self.damage_body

    def get_magazine_capacity(self, tactical: int) -> int:
        return max(self.magazine_capacity - tactical, 0)

    def get_rounds_per_minute(self):
        return self.rounds_per_minute

    def get_rounds_per_second(self) -> float:
        return self.get_rounds_per_minute() / 60

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._name)

    def __eq__(self, other: 'Weapon'):
        return isinstance(other, Weapon) and self._name == other._name

    def get_weapon_class(self) -> WeaponClass:
        return self.weapon_class

    def get_weapon(self) -> 'Weapon':
        return self

    def single_shot(self) -> '_SingleShotLoadout':
        return _SingleShotLoadout(self)

    def is_single_shot_advisable(self) -> bool:
        seconds_per_round = 1 / self.get_rounds_per_second()
        return seconds_per_round >= self.holster_time_secs


class _SingleShotLoadout(SingleWeaponLoadout):
    def __init__(self, wrapped_weapon: Weapon):
        if not wrapped_weapon.is_single_shot_advisable():
            raise ValueError(f'Weapon {wrapped_weapon} must have single shots being advisable!')
        check_type(Weapon, wrapped_weapon=wrapped_weapon)
        super().__init__(wrapped_weapon.get_name(), wrapped_weapon.get_term())
        self.wrapped_weapon = wrapped_weapon

    def get_archetype(self) -> 'WeaponArchetype':
        return self.wrapped_weapon.get_archetype()

    def get_holster_time_secs(self) -> float:
        return self.wrapped_weapon.get_holster_time_secs()

    def get_ready_to_fire_time_secs(self) -> float:
        return self.wrapped_weapon.get_ready_to_fire_time_secs()

    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: float,
                                         tactical: int) -> float:
        return self.wrapped_weapon.get_single_mag_cumulative_damage(time_seconds=0,
                                                                    distance_meters=distance_meters,
                                                                    tactical=tactical)

    def get_single_mag_cumulative_damage_vec(self,
                                             times_seconds: NDArray[np.float64],
                                             distances_meters: NDArray[np.float64],
                                             tactical: int) -> NDArray[np.float64]:
        return self.wrapped_weapon.get_single_mag_cumulative_damage_vec(
            times_seconds=np.zeros_like(times_seconds),
            distances_meters=distances_meters,
            tactical=tactical)

    def get_magazine_duration_seconds(self, tactical: int) -> float:
        return 0

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.wrapped_weapon)

    def __eq__(self, other):
        return (isinstance(other, _SingleShotLoadout) and
                self.wrapped_weapon == other.wrapped_weapon)

    def get_weapon(self) -> 'Weapon':
        return self.wrapped_weapon


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
                 spinup: RoundTiming, heat_based: bool,
                 associated_legend: Optional[Legend],
                 care_package: bool):
        check_type(Suffix, optional=True, suffix=suffix)
        check_bool(care_package=care_package)

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
        self.care_package = care_package

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

    def is_hopped_up(self) -> bool:
        return (self.suffix is not None and
                SuffixedArchetypeType.HOPPED_UP in self.suffix.get_types())

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

    def is_care_package(self) -> bool:
        return self.care_package


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
                    f'Both weapon archetypes must have the same base term ({repr(base_term)} != '
                    f'{repr(suffixed_archetype.get_base_term())}).')

            if associated_legend != suffixed_archetype.get_associated_legend():
                raise ValueError(
                    'Both weapon archetypes must have the same required legend '
                    f'({associated_legend} != {suffixed_archetype.get_associated_legend()}).')

            if suffixed_archetype.get_suffix() is None:
                raise ValueError('with_hopup_archetype must have a suffix.')

        sort_key = lambda _archetype: _archetype.get_best_weapon().get_damage_per_second_body()

        all_archetypes = (base_archetype,) + suffixed_archetypes
        suffixes_and_archetypes: Tuple[Tuple[Optional[Suffix], WeaponArchetype], ...] = \
            tuple(sorted(((suffixed_archetype.get_suffix(), suffixed_archetype)
                          for suffixed_archetype in all_archetypes),
                         key=lambda suffix_and_archetype: sort_key(suffix_and_archetype[1]),
                         reverse=True))

        self._base_archetype = base_archetype
        self._suffixes_and_archetypes = suffixes_and_archetypes
        self._best_archetype = max(all_archetypes, key=sort_key)
        self._base_term = base_term
        self._associated_legend = associated_legend
        self._has_hopped_up_archetype = any(archetype.is_hopped_up()
                                            for archetype in all_archetypes)

        self._is_care_package = base_archetype.is_care_package()
        if not all(archetype.is_care_package() == base_archetype.is_care_package()
                   for archetype in suffixed_archetypes):
            raise ValueError('Archetypes must all be care package or non-care package.')

    def get_associated_legend(self) -> Optional[Legend]:
        return self._associated_legend

    def _get_archetype(self,
                       words: Words,
                       overall_level: OverallLevel) -> WeaponArchetype:
        check_type(Words, words=words)
        if overall_level in (OverallLevel.FULLY_KITTED, OverallLevel.LEVEL_3):
            return self._best_archetype

        for suffix, suffixed_archetype in self._suffixes_and_archetypes:
            if suffix is not None and WeaponArchetype.find_suffix(suffix, words):
                return suffixed_archetype

        return self._base_archetype

    def get_base_term(self) -> RequiredTerm:
        return self._base_term

    def get_fully_kitted_weapons(self,
                                 legend: Optional[Legend] = None,
                                 include_non_hopped_up: bool = False) -> \
            Generator[Weapon, None, None]:
        for _, archetype in self._suffixes_and_archetypes:
            if (not self._has_hopped_up_archetype or
                    include_non_hopped_up or
                    archetype.is_hopped_up()):
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

    def is_care_package(self) -> bool:
        return self._is_care_package
