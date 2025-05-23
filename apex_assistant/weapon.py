import abc
import logging
import math
import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import (Any,
                    Dict,
                    FrozenSet,
                    Generator,
                    Generic,
                    Iterable,
                    List,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    TypeVar,
                    Union,
                    final)

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from scipy.stats import binom

from apex_assistant.checker import (check_bool,
                                    check_equal_length,
                                    check_float,
                                    check_float_vec,
                                    check_int,
                                    check_mapping,
                                    check_str,
                                    check_tuple,
                                    check_type,
                                    to_kwargs)
from apex_assistant.config import (CONFIG,
                                   MAX_NUM_PERMUTED_WEAPONS,
                                   PLAYER_ACCURACY,
                                   SupportedDistanceMeters,
                                   get_distance_time_multiplier,
                                   get_p_kill_for_damages_to_kill,
                                   get_p_kill_for_distances,
                                   get_p_kill_for_times)
from apex_assistant.legend import Legend
from apex_assistant.overall_level import OverallLevel
from apex_assistant.speech.apex_terms import (ALL_BOLT_TERMS,
                                              ALL_MAG_TERMS,
                                              ALL_STOCK_TERMS,
                                              BASE,
                                              BOOSTED_LOADER,
                                              LEVEL_TERMS,
                                              Suffix,
                                              SuffixedArchetypeType,
                                              WITH_SIDEARM)
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import SingleTermFinder, Translator
from apex_assistant.speech.translations import TranslatedValue
from apex_assistant.weapon_class import WeaponClass

T = TypeVar('T')
_LOGGER = logging.getLogger()


@dataclass
class WeaponStats:
    weapon: 'SingleWeaponLoadout'
    shot_times_seconds: NDArray[np.float64]


ERROR_TOL = 1e-9


def __get_p_kill(damage_to_kill: float,
                 p_nums_hits_and_damages: Tuple[Tuple[NDArray[np.float64], float], ...],
                 num_permuted_weapons: int,
                 max_num_permuted_weapons: int) -> float:
    assert 0 <= num_permuted_weapons <= max_num_permuted_weapons
    if num_permuted_weapons == max_num_permuted_weapons:
        expected_damage = sum(
            (p_nums_hits * np.arange(len(p_nums_hits))).sum() * damage_per_round
            for p_nums_hits, damage_per_round in p_nums_hits_and_damages
        )
        return 1.0 if expected_damage >= damage_to_kill - ERROR_TOL else 0.0

    p_num_hits_0, damage_per_round_0 = p_nums_hits_and_damages[0]
    num_shots_0 = len(p_num_hits_0) - 1
    hits_to_kill_0_min = math.ceil((damage_to_kill - ERROR_TOL) / damage_per_round_0)

    hits_not_to_kill_0_min = 0
    hits_not_to_kill_0_max = min(hits_to_kill_0_min - 1, num_shots_0)
    hits_not_to_kill_0 = np.arange(hits_not_to_kill_0_min, hits_not_to_kill_0_max + 1)
    damage_0 = damage_per_round_0 * hits_not_to_kill_0
    remaining_damages_to_kill = damage_to_kill - damage_0

    p_num_hits_not_to_kill_0 = p_num_hits_0[hits_not_to_kill_0]
    p_kill_0 = 1 - p_num_hits_not_to_kill_0.sum()

    if len(p_nums_hits_and_damages) <= 1:
        return p_kill_0

    next_num_permuted_weapons = num_permuted_weapons + 1
    next_p_nums_hits_and_damages = tuple(p_nums_hits_and_damages[1:])
    p_kill_given_num_hits_0 = np.fromiter(
        (__get_p_kill(damage_to_kill=remaining_damage_to_kill,
                      p_nums_hits_and_damages=next_p_nums_hits_and_damages,
                      num_permuted_weapons=next_num_permuted_weapons,
                      max_num_permuted_weapons=max_num_permuted_weapons)
         for remaining_damage_to_kill in remaining_damages_to_kill),
        dtype=float,
        count=len(remaining_damages_to_kill))
    p_kill_subsequent_weapons = (p_num_hits_not_to_kill_0 * p_kill_given_num_hits_0).sum()
    p_kill = p_kill_0 + p_kill_subsequent_weapons

    return p_kill


def get_p_kill(damage_to_kill: float,
               p_nums_hits_and_damages: Tuple[Tuple[NDArray[np.float64], float], ...],
               max_num_permuted_weapons: int) -> float:
    if len(p_nums_hits_and_damages) == 0:
        # This can happen if you have a time given which is lower than the first shot time.
        return 0

    return __get_p_kill(damage_to_kill=damage_to_kill,
                        p_nums_hits_and_damages=p_nums_hits_and_damages,
                        num_permuted_weapons=0,
                        max_num_permuted_weapons=max_num_permuted_weapons)


def _get_p_nums_hits_and_damages(weapons_and_shot_times: Iterable[WeaponStats],
                                 time_given_seconds: float,
                                 distance_meters: SupportedDistanceMeters) -> \
        Generator[Tuple[NDArray[np.float64], float], None, None]:
    for weapon_stats in weapons_and_shot_times:
        shot_times_seconds = weapon_stats.shot_times_seconds
        last_round_time = shot_times_seconds[-1]
        weapon = weapon_stats.weapon
        damage_per_round = weapon.get_damage_per_round()

        if last_round_time < time_given_seconds:
            # We still potentially have time to fire one or more additional rounds.
            p_nums_hits = weapon.get_num_hit_probabilities(
                num_shots=len(shot_times_seconds),
                distance_meters=distance_meters)
            assert math.isclose(p_nums_hits.sum(), 1), p_nums_hits.sum()
            yield p_nums_hits, damage_per_round
            continue

        too_late_indices = np.flatnonzero(shot_times_seconds >= time_given_seconds)
        num_rounds = too_late_indices[0]
        if num_rounds > 0:
            p_nums_hits_slice = weapon.get_num_hit_probabilities(
                num_shots=num_rounds,
                distance_meters=distance_meters)
            yield p_nums_hits_slice, damage_per_round

        break


class ExcludeFlag(IntEnum):
    NONE = 0x0
    CARE_PACKAGE = 0x1
    HOPPED_UP = 0x2
    NON_HOPPED_UP = 0x4
    REVVED_UP = 0x8
    NON_REVVED_UP = 0x10
    AKIMBO = 0x20
    NON_AKIMBO = 0x40
    RELIC = 0x80

    def find(self, exclude_flags: int) -> bool:
        return (exclude_flags & self.value) != 0


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
    def get_shot_times_seconds(self, base_weapon: 'Weapon') -> NDArray[np.float64]:
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

    def get_shot_times_seconds(self, base_weapon: 'Weapon') -> NDArray[np.float64]:
        # As a sanity check, magazine duration for Care Package Devotion was measured as ~227-229
        # frames at 60 FPS (~3.783-3.817 seconds).
        magazine_capacity = base_weapon.get_magazine_capacity(tactical=0)
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

    def get_shot_times_seconds(self, base_weapon: 'Weapon') -> NDArray[np.float64]:
        burst_periods_seconds = self._get_burst_periods_seconds(base_weapon, tactical=0)
        burst_stop_times_seconds = burst_periods_seconds.cumsum()
        burst_start_times_seconds = np.append(0, burst_stop_times_seconds[:-1])
        round_relative_start_times_seconds = (np.arange(self._rounds_per_burst) /
                                              base_weapon.get_rounds_per_second())
        round_times_seconds = (burst_start_times_seconds[:, np.newaxis] +
                               round_relative_start_times_seconds[np.newaxis, :]).flatten()
        return round_times_seconds

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
                                   tactical: int) -> NDArray[np.float64]:
        burst_periods_seconds = self._get_burst_durations_seconds(base_weapon=base_weapon,
                                                                  tactical=tactical)
        burst_periods_seconds[:-1] += self._burst_fire_delay
        return burst_periods_seconds

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

    def get_shot_times_seconds(self, base_weapon: 'Weapon') -> NDArray[np.float64]:
        num_rounds = base_weapon.get_magazine_capacity(tactical=0)
        return np.arange(num_rounds) / base_weapon.get_rounds_per_second()

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, RoundTimingRegular)


class RoundTimingHavoc(RoundTiming):
    _NO_SPINUP = RoundTimingRegular.get_instance()

    def __init__(self, spinup_time_seconds: float):
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)
        self._spinup_time_seconds = spinup_time_seconds

    def get_shot_times_seconds(self, base_weapon: 'Weapon') -> NDArray[np.float64]:
        return self._NO_SPINUP.get_shot_times_seconds(base_weapon) + self._spinup_time_seconds

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

    def get_cumulative_damage(self,
                              weapon_a: 'Weapon',
                              weapon_b: 'Weapon',
                              time_seconds: float,
                              distance_meters: SupportedDistanceMeters) -> float:
        check_float(min_value=0, time_seconds=time_seconds)
        check_type(SupportedDistanceMeters, distance_meters=distance_meters)

        damage: float = 0

        for weapon, tactical, start_time_seconds in self._iter_weapons(weapon_a, weapon_b):
            valid = time_seconds >= start_time_seconds
            if not valid:
                break

            damage += weapon.get_single_mag_cumulative_damage(
                time_seconds=time_seconds - start_time_seconds,
                distance_meters=distance_meters,
                tactical=tactical)
            assert isinstance(damage, (int, float))

        return damage

    def get_cumulative_damage_vec(self,
                                  weapon_a: 'Weapon',
                                  weapon_b: 'Weapon',
                                  times_seconds: NDArray[np.float64]) -> NDArray[np.float64]:
        check_float_vec(min_value=0, times_seconds=times_seconds)

        damages: NDArray[np.float64] = np.zeros_like(times_seconds)

        for weapon, tactical, start_time_seconds in self._iter_weapons(weapon_a, weapon_b):
            valid_indices = np.flatnonzero(times_seconds >= start_time_seconds)
            if len(valid_indices) == 0:
                break

            damages[valid_indices] += weapon.get_single_mag_cumulative_damage_vec(
                times_seconds=times_seconds[valid_indices] - start_time_seconds,
                tactical=tactical)

        return damages

    def get_p_kill_config(self,
                          weapon_a: 'Weapon',
                          weapon_b: 'Weapon',
                          distance_meters: SupportedDistanceMeters) -> float:
        check_type(SupportedDistanceMeters, distance_meters=distance_meters)

        times_given_seconds = (get_distance_time_multiplier(distance_meters) *
                               CONFIG.times_given_seconds)

        max_time_given_seconds = times_given_seconds.max()
        weapon_stats = tuple(self.__iter_weapons_stats(
            weapon_a=weapon_a,
            weapon_b=weapon_b,
            max_time_given_seconds=max_time_given_seconds))

        p_kill = get_p_kill_for_times(
            FullLoadoutConfiguration.__get_p_kill_config(
                weapon_stats=weapon_stats,
                time_given_seconds=time_given_seconds,
                distance_meters=distance_meters)
            for time_given_seconds in times_given_seconds)

        return p_kill

    @staticmethod
    def __get_p_kill_config(weapon_stats: Tuple[WeaponStats, ...],
                            time_given_seconds: float,
                            distance_meters: SupportedDistanceMeters):
        damages_to_kill = CONFIG.damages_to_kill
        p_nums_hits_and_damages = tuple(_get_p_nums_hits_and_damages(
            weapons_and_shot_times=weapon_stats,
            time_given_seconds=time_given_seconds,
            distance_meters=distance_meters))

        return get_p_kill_for_damages_to_kill(
            get_p_kill(damage_to_kill=damage_to_kill,
                       p_nums_hits_and_damages=p_nums_hits_and_damages,
                       max_num_permuted_weapons=MAX_NUM_PERMUTED_WEAPONS)
            for damage_to_kill in damages_to_kill)

    @staticmethod
    def _get_damage_permutations_for_rounds(damages: NDArray[np.float64],
                                            accuracies: NDArray[np.float64]) -> \
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
        num_rounds = len(damages)
        if len(accuracies) != num_rounds:
            raise ValueError(f'len(accuracies) != len(damages)')
        probability_of_hit = accuracies
        probability_of_miss = 1 - probability_of_hit

        # Get possible permutations in terms of integer.
        num_permutations = 2 ** num_rounds
        permutations = np.arange(num_permutations, dtype=int)
        possible_permutation_indices = np.ones_like(permutations, dtype=bool)
        guaranteed_hits = probability_of_hit >= 1
        guaranteed_misses = probability_of_miss >= 1
        for round_idx, (guaranteed_hit, guaranteed_miss) in enumerate(zip(guaranteed_hits,
                                                                          guaranteed_misses)):
            round_hit_indices = (permutations & (1 << round_idx)) != 0
            round_miss_indices = np.logical_not(round_hit_indices)

            if guaranteed_miss:
                possible_permutation_indices[round_hit_indices] = False
            if guaranteed_hit:
                possible_permutation_indices[round_miss_indices] = False

        permutations = permutations[possible_permutation_indices]
        num_permutations = len(permutations)

        # Get possible permutations in terms of Boolean integer arrays.
        shots_hit = np.array([(permutations & (1 << round_idx)) != 0
                              for round_idx in range(num_rounds)]).T.astype(int)
        assert shots_hit.shape == (num_permutations, num_rounds)

        shots_miss = 1 - shots_hit
        shots_bools_probabilities = (shots_hit * probability_of_hit +
                                     shots_miss * probability_of_miss)
        assert shots_bools_probabilities.shape == (num_permutations, num_rounds)

        permutation_probabilities = shots_bools_probabilities.cumprod(axis=1)[:, -1]

        assert permutation_probabilities.shape == (num_permutations,)
        damage_permutations = shots_hit * damages
        assert damage_permutations.shape == (num_permutations, num_rounds)

        return permutation_probabilities, damage_permutations

    def __iter_weapons_stats(self,
                             weapon_a: 'Weapon',
                             weapon_b: 'Weapon',
                             max_time_given_seconds: float) -> \
            Generator[WeaponStats, None, None]:
        for weapon, tactical, time_seconds in self._iter_weapons(weapon_a, weapon_b):
            if time_seconds > max_time_given_seconds:
                break

            shot_times_seconds = time_seconds + weapon.get_shot_times_seconds(tactical=tactical)

            yield WeaponStats(weapon=weapon, shot_times_seconds=shot_times_seconds)

    def _iter_weapons(self, weapon_a: 'Weapon', weapon_b: 'Weapon') -> \
            Generator[Tuple['SingleWeaponLoadout', int, float], None, None]:
        configuration_string = self._get_configuration_str()
        char_to_weapon_dict = self._get_char_to_weapon_dict(weapon_a, weapon_b)
        prev_weapon_and_char: Optional[Tuple['SingleWeaponLoadout', str]] = None
        time_seconds: float = 0

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
                will_reload = prev_weapon.can_reload() and self._want_to_reload_later(prev_idx)
                used_single_shot = self._used_single_shot_previously(prev_idx)
                tactical = ((prev_weapon.get_tactical() if will_reload else 0) +
                            int(used_single_shot))

                yield prev_weapon, tactical, time_seconds

                time_seconds += (prev_weapon.get_magazine_duration_seconds(tactical) +
                                 _HUMAN_REACTION_TIME_SECONDS +
                                 prev_weapon.get_swap_time_seconds(cur_weapon))

            prev_weapon_and_char = cur_weapon, cur_char

        if prev_weapon_and_char is None:
            return

        final_weapon, _ = prev_weapon_and_char
        prev_index = len(configuration_string) - 1
        will_reload = final_weapon.can_reload()
        used_single_shot = self._used_single_shot_previously(prev_index)
        # Using up the remaining rounds in the magazine.
        tactical = (final_weapon.get_tactical() if will_reload else
                    # This was 1 at one point, but I don't know why.
                    0) + int(used_single_shot)
        yield final_weapon, tactical, time_seconds

        if not will_reload:
            return

        while True:
            time_seconds += (final_weapon.get_magazine_duration_seconds(tactical) +
                             _HUMAN_REACTION_TIME_SECONDS +
                             final_weapon.get_tactical_reload_time_seconds())

            yield final_weapon, tactical, time_seconds

    def _want_to_reload_later(self, index: int) -> bool:
        configuration_string = self._get_configuration_str()
        if index == len(configuration_string) - 1:
            return True
        char = configuration_string[index]
        return char in configuration_string[index + 1:]

    def _used_single_shot_previously(self, index: int) -> bool:
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

    def __init__(self, weapon_a: 'Weapon', weapon_b: 'Weapon'):
        check_type(Weapon, weapon_a=weapon_a, weapon_b=weapon_b)
        self._weapons_set = frozenset((weapon_a, weapon_b))

        self._weapon_a: Weapon | None = None
        self._weapon_b: Weapon | None = None
        self._distance_to_config_map: \
            Mapping[SupportedDistanceMeters, FullLoadoutConfiguration] | None = None
        self._configs: FrozenSet[FullLoadoutConfiguration] | None = None
        self._distance_to_p_kill_map: Dict[SupportedDistanceMeters, float] | None = None
        self._base_name: str | None = None
        self._config_descriptor: str | None = None
        self._term: RequiredTerm | None = None

    def get_term(self) -> RequiredTerm:
        self.lazy_load_internals()
        return self._term

    def get_name(self, config: FullLoadoutConfiguration | None = None) -> str:
        self.lazy_load_internals()
        config_descriptor = config if config is not None else self._config_descriptor
        return f'{self._base_name} ({config_descriptor})'

    @staticmethod
    def _get_distance_range_str(distances: Sequence[SupportedDistanceMeters]) -> str:
        if len(distances) == 0:
            raise ValueError('No distances found.')

        first_dist = distances[0]
        last_dist = distances[-1]
        if len(distances) == 1:
            dists_str = f'{first_dist}'
        else:
            dists_str = f'{first_dist}-{last_dist}'
        return f'{dists_str}m'

    @staticmethod
    def _get_config_description(
            distance_to_config_map: Mapping[SupportedDistanceMeters, FullLoadoutConfiguration]
    ) -> Tuple[str, bool]:
        cur_distances: list[SupportedDistanceMeters] = []
        cur_config: FullLoadoutConfiguration | None = None
        str_builder: list[str] = []
        invert_configs: bool = False
        for distance, config in distance_to_config_map.items():
            if invert_configs:
                config = config.invert()
            if config is not cur_config:
                if cur_config is not None:
                    str_builder.append(
                        f'{FullLoadout._get_distance_range_str(cur_distances)}: {cur_config}')
                elif config.should_invert():
                    config = config.invert()
                    invert_configs = True

                cur_config = config
                cur_distances.clear()
            cur_distances.append(distance)

        if len(cur_distances) == len(distance_to_config_map):
            # All configurations were the same regardless of distance.
            return cur_config, invert_configs

        if len(cur_distances) != 0:
            str_builder.append(
                f'{FullLoadout._get_distance_range_str(cur_distances)}: {cur_config}')

        return ', '.join(str_builder), invert_configs

    def lazy_load_internals(self):
        if self._distance_to_config_map is not None:
            return

        weapons_iter = iter(self._weapons_set)
        weapon_a = next(weapons_iter)
        weapon_b = next(weapons_iter, weapon_a)

        advisable_configurations: Tuple[FullLoadoutConfiguration, ...] = tuple(
            config
            for config in CONFIGURATIONS
            if config.is_advisable(weapon_a, weapon_b))
        if len(advisable_configurations) == 0:
            raise RuntimeError('No configurations were advisable.')

        distances_meters = CONFIG.distances_meters
        distance_to_config_map: Mapping[SupportedDistanceMeters, FullLoadoutConfiguration] = {
            supported_distance: FullLoadout._get_best_configuration(
                weapon_a=weapon_a,
                weapon_b=weapon_b,
                distance_meters=supported_distance,
                advisable_configurations=advisable_configurations)
            for supported_distance in distances_meters
        }

        self._config_descriptor, inverted = \
            FullLoadout._get_config_description(distance_to_config_map)
        if inverted:
            weapon_a, weapon_b = weapon_b, weapon_a
            distance_to_config_map = MappingProxyType({
                distance: config.invert()
                for distance, config in distance_to_config_map.items()
            })
        self._weapon_a = weapon_a
        self._weapon_b = weapon_b
        self._distance_to_config_map = distance_to_config_map
        self._configs = frozenset(distance_to_config_map.values())
        self._base_name = f'{weapon_a.get_name()} {WITH_SIDEARM} {weapon_b.get_name()}'
        self._term = weapon_a.get_term().append(WITH_SIDEARM, weapon_b.get_term())

        self._distance_to_p_kill_map = {
            distance_meters: config.get_p_kill_config(weapon_a=weapon_a,
                                                      weapon_b=weapon_b,
                                                      distance_meters=distance_meters)
            for distance_meters, config in distance_to_config_map.items()}

    @staticmethod
    def _get_best_configuration(
            weapon_a: 'Weapon',
            weapon_b: 'Weapon',
            distance_meters: SupportedDistanceMeters,
            advisable_configurations: Iterable[FullLoadoutConfiguration]
    ) -> FullLoadoutConfiguration:
        return max(advisable_configurations,
                   key=lambda _config: _config.get_p_kill_config(weapon_a=weapon_a,
                                                                 weapon_b=weapon_b,
                                                                 distance_meters=distance_meters))

    def get_cumulative_damage(self,
                              time_seconds: float,
                              distance_meters: SupportedDistanceMeters) -> float:
        check_float(min_value=0, time_seconds=time_seconds)
        check_type(SupportedDistanceMeters, distance_meters=distance_meters)
        self.lazy_load_internals()

        configuration = self._distance_to_config_map[distance_meters]
        return configuration.get_cumulative_damage(weapon_a=self._weapon_a,
                                                   weapon_b=self._weapon_b,
                                                   time_seconds=time_seconds,
                                                   distance_meters=distance_meters)

    def get_cumulative_damage_vec(self, times_seconds: NDArray[np.float64]) -> \
            Generator[Tuple[FullLoadoutConfiguration, NDArray[np.float64]], None, None]:
        if (times_seconds < 0).any():
            raise ValueError('No time values can be negative.')

        self.lazy_load_internals()

        for config in self._configs:
            result = self.get_cumulative_damage_with_config_vec(
                config=config,
                times_seconds=times_seconds)
            assert (result >= 0).all()
            yield config, result

    def get_cumulative_damage_with_config_vec(self,
                                              config: FullLoadoutConfiguration,
                                              times_seconds: NDArray[np.float64]) -> \
            NDArray[np.float64]:
        return config.get_cumulative_damage_vec(weapon_a=self._weapon_a,
                                                weapon_b=self._weapon_b,
                                                times_seconds=times_seconds)

    def get_p_kill(self, distance_meters: Optional[SupportedDistanceMeters] = None) -> float:
        self.lazy_load_internals()

        if distance_meters is None:
            return get_p_kill_for_distances(self._distance_to_p_kill_map.values())

        return self._distance_to_p_kill_map[distance_meters]

    def get_p_kill_vec(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.lazy_load_internals()

        p_kills = np.fromiter(
            (self.get_p_kill(distance_meters) for distance_meters in CONFIG.distances_meters),
            dtype=float,
            count=len(CONFIG.distances_meters))
        distances_meters_float = np.array(CONFIG.distances_meters, dtype=float)

        return p_kills, distances_meters_float

    def _get_configs_and_distances(self, distances_meters: Iterable[SupportedDistanceMeters]) -> \
            Generator[Tuple[FullLoadoutConfiguration, SupportedDistanceMeters], None, None]:
        for dist in distances_meters:
            yield self._distance_to_config_map[dist], dist

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._weapons_set)

    def __eq__(self, other):
        return isinstance(other, FullLoadout) and hash(self._weapons_set)

    def get_weapons(self) -> FrozenSet['Weapon']:
        return self._weapons_set

    def get_weapon_a(self) -> 'Weapon':
        self.lazy_load_internals()
        return self._weapon_a

    def get_weapon_b(self):
        self.lazy_load_internals()
        return self._weapon_b


_PROBABILITIES_CACHE: \
    Dict['SingleWeaponLoadout', Mapping[SupportedDistanceMeters, NDArray[np.float64]]] = {}


class SingleWeaponLoadout(Loadout, abc.ABC):
    def __init__(self, name: str, term: RequiredTerm):
        check_str(allow_blank=False, name=name)
        check_type(RequiredTerm, term=term)
        self._name = name
        self._term = term
        self._dists_to_pmfs: Mapping[SupportedDistanceMeters, NDArray[np.float64]] | None = None

    def __calc_shot_probabilities(self, distance_meters: SupportedDistanceMeters):
        num_shots = self.get_magazine_capacity(tactical=0)

        # Row indices plus one are for numbers of trials (shots); column indices are for k values.
        return binom.pmf(k=np.arange(num_shots + 1),
                         n=np.arange(1, num_shots + 1)[:, np.newaxis],
                         p=self.get_accuracy_fraction(distance_meters) * PLAYER_ACCURACY)

    def __lazy_load_pmfs(self):
        if self._dists_to_pmfs is not None:
            return

        self._dists_to_pmfs: Mapping[SupportedDistanceMeters, NDArray[np.float64]] = {
            distance_meters: self.__calc_shot_probabilities(distance_meters)
            for distance_meters in CONFIG.distances_meters}

    def get_num_hit_probabilities(self,
                                  num_shots: int,
                                  distance_meters: SupportedDistanceMeters) -> NDArray[np.float64]:
        if num_shots == 0:
            return np.array([], dtype=float)

        self.__lazy_load_pmfs()
        pmf = self._dists_to_pmfs[distance_meters][num_shots - 1, :num_shots + 1]
        return pmf

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

    @final
    def get_single_mag_cumulative_damage(self,
                                         time_seconds: float,
                                         distance_meters: SupportedDistanceMeters,
                                         tactical: int) -> float:
        num_rounds = self.get_num_rounds_shot(time_seconds=time_seconds, tactical=tactical)
        if num_rounds < 0:
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a non-negative number of rounds.')
        if num_rounds > self.get_magazine_capacity(tactical):
            raise RuntimeError(f'get_num_rounds_shot() in {self.__class__.__name__} must not '
                               'return a number of rounds greater than the magazine capacity.')
        get_cumulative_damage = self.get_damage_per_round() * num_rounds

        return get_cumulative_damage * self.get_accuracy_fraction(distance_meters=distance_meters)

    @final
    def get_single_mag_cumulative_damage_vec(self,
                                             times_seconds: NDArray[np.float64],
                                             tactical: int) -> NDArray[np.float64]:
        nums_rounds = self.get_nums_rounds_shot(times_seconds=times_seconds, tactical=tactical)
        if (nums_rounds < 0).any():
            raise RuntimeError(
                f'{self.get_nums_rounds_shot.__name__}() in {self.__class__.__name__} must not '
                'return a non-negative number of rounds.')
        if (nums_rounds > self.get_magazine_capacity(tactical)).any():
            raise RuntimeError(
                f'{self.get_nums_rounds_shot.__name__}() in {self.__class__.__name__} must not '
                'return a number of rounds greater than the magazine capacity.')
        cumulative_damage_vec = self.get_damage_per_round() * nums_rounds

        return cumulative_damage_vec

    @abc.abstractmethod
    def get_magazine_capacity(self, tactical: int) -> int:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_num_rounds_shot(self, time_seconds: float, tactical: int) -> int:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_nums_rounds_shot(self, times_seconds: NDArray[np.float64], tactical: int) \
            -> NDArray[np.integer[Any]]:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_shot_times_seconds(self, tactical: int) -> NDArray[np.float64]:
        raise NotImplementedError('Must implement.')

    def get_swap_or_tactical_reload_time_seconds(
            self,
            weapon_swap_to: Optional['SingleWeaponLoadout']) -> float:
        check_type(SingleWeaponLoadout, optional=True, weapon_swap_to=weapon_swap_to)
        if weapon_swap_to is None:
            return self.get_weapon().get_tactical_reload_time_secs()
        return self.get_holster_time_secs() + weapon_swap_to.get_ready_to_fire_time_secs()

    def get_tactical_reload_time_seconds(self) -> float:
        return self.get_weapon().get_tactical_reload_time_secs()

    def get_swap_time_seconds(self, weapon_swap_to: 'SingleWeaponLoadout') -> float:
        check_type(SingleWeaponLoadout, weapon_swap_to=weapon_swap_to)
        return self.get_holster_time_secs() + weapon_swap_to.get_ready_to_fire_time_secs()

    @abc.abstractmethod
    def get_holster_time_secs(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ready_to_fire_time_secs(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_damage_per_round(self) -> float:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_accuracy_fraction(self, distance_meters: SupportedDistanceMeters) -> float:
        raise NotImplementedError('Must implement.')

    def can_reload(self) -> bool:
        return self.get_weapon().get_tactical_reload_time_secs() is not None

    def has_boosted_loader(self) -> bool:
        return self.get_weapon().has_boosted_loader()

    def get_tactical(self) -> int:
        return 1 if not self.has_boosted_loader() else 0


class Weapon(SingleWeaponLoadout):
    EIGHTY_PERCENT = 0.8

    def __init__(self,
                 archetype: 'WeaponArchetype',
                 name: str,
                 term: RequiredTerm,
                 weapon_class: WeaponClass,
                 dist_to_accuracy_mapping: Mapping[SupportedDistanceMeters, float],
                 damage_body: float,
                 holster_time_secs: float,
                 ready_to_fire_time_secs: float,
                 rounds_per_minute: float,
                 magazine_capacity: int,
                 spinup: RoundTiming,
                 heat_based: bool,
                 tactical_reload_time_secs: Optional[float],
                 legend: Optional[Legend],
                 has_boosted_loader: bool):
        check_type(WeaponArchetype, archetype=archetype)
        check_str(name=name)
        check_type(RequiredTerm, term=term)
        check_type(WeaponClass, weapon_class=weapon_class)
        check_mapping(allowed_key_types=int,
                      allowed_value_types=(int, float),
                      dist_to_accuracy_mapping=dist_to_accuracy_mapping)
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
        check_bool(heat_based=heat_based, has_boosted_loader=has_boosted_loader)
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
        self.dist_to_accuracy_mapping = dist_to_accuracy_mapping
        self.damage_body = damage_body
        self.holster_time_secs = holster_time_secs
        self.ready_to_fire_time_secs = ready_to_fire_time_secs
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self._round_timing = spinup
        self.heat_based = heat_based
        self.tactical_reload_time_secs = tactical_reload_time_secs
        self._has_boosted_loader = has_boosted_loader

        damage_per_minute_body = damage_body * rounds_per_minute
        self._damage_per_second_body = damage_per_minute_body / 60
        self._shot_times_seconds = self._round_timing.get_shot_times_seconds(self)
        if len(self._shot_times_seconds) != magazine_capacity:
            raise RuntimeError('Invalid result of get_shot_times_seconds()!')
        self.__legend = legend

    def get_num_rounds_shot(self, time_seconds: float, tactical: int) -> int:
        shot_times_seconds = self.get_shot_times_seconds(tactical=tactical)
        return np.count_nonzero(time_seconds >= shot_times_seconds)

    def get_nums_rounds_shot(self, times_seconds: NDArray[np.float64], tactical: int) \
            -> NDArray[np.integer[Any]]:
        shot_times_seconds = self.get_shot_times_seconds(tactical=tactical)
        nums_rounds_shot = np.count_nonzero(times_seconds.reshape(-1, 1) >=
                                            shot_times_seconds.reshape(1, -1), axis=1)
        assert nums_rounds_shot.shape == times_seconds.shape
        return nums_rounds_shot

    def get_shot_times_seconds(self, tactical: int) -> NDArray[np.float64]:
        num_rounds = max(0, len(self._shot_times_seconds) - tactical)
        return self._shot_times_seconds[:num_rounds]

    def get_archetype(self) -> 'WeaponArchetype':
        return self.archetype

    def get_holster_time_secs(self) -> float:
        return self.holster_time_secs

    def get_ready_to_fire_time_secs(self) -> float:
        return self.ready_to_fire_time_secs

    def get_tactical_reload_time_secs(self) -> float | None:
        return self.tactical_reload_time_secs

    def get_accuracy_fraction(self, distance_meters: SupportedDistanceMeters) -> float:
        check_type(SupportedDistanceMeters, distance_meters=distance_meters)
        return self.dist_to_accuracy_mapping[distance_meters]

    def get_magazine_duration_seconds(self, tactical: int) -> float:
        shot_times = self.get_shot_times_seconds(tactical=tactical)
        return float(shot_times[-1])

    def get_damage_per_second_body(self) -> float:
        return self._damage_per_second_body

    def get_damage_per_round(self) -> float:
        return self.damage_body

    def get_magazine_capacity(self, tactical: int) -> int:
        return max(self.magazine_capacity - tactical, 0)

    def get_rounds_per_minute(self):
        return self.rounds_per_minute

    def get_rounds_per_second(self) -> float:
        return self.get_rounds_per_minute() / 60

    def __hash__(self):
        return hash((self.__class__, self._name, self.__legend))

    def __eq__(self, other: 'Weapon'):
        return (isinstance(other, Weapon) and
                (self._name, self.__legend) == (other._name, other.__legend))

    def get_weapon_class(self) -> WeaponClass:
        return self.weapon_class

    def get_weapon(self) -> 'Weapon':
        return self

    def single_shot(self) -> '_SingleShotLoadout':
        return _SingleShotLoadout(self)

    def is_single_shot_advisable(self) -> bool:
        seconds_per_round = 1 / self.get_rounds_per_second()
        return seconds_per_round >= self.holster_time_secs

    def has_boosted_loader(self) -> bool:
        return self._has_boosted_loader


class _SingleShotLoadout(SingleWeaponLoadout):
    def __init__(self, wrapped_weapon: Weapon):
        if not wrapped_weapon.is_single_shot_advisable():
            raise ValueError(f'Weapon {wrapped_weapon} must have single shots being advisable!')
        check_type(Weapon, wrapped_weapon=wrapped_weapon)
        super().__init__(wrapped_weapon.get_name(), wrapped_weapon.get_term())
        self.wrapped_weapon = wrapped_weapon
        self.first_shot_time_seconds = self.wrapped_weapon.get_shot_times_seconds(tactical=0)[0]

    def get_archetype(self) -> 'WeaponArchetype':
        return self.wrapped_weapon.get_archetype()

    def get_holster_time_secs(self) -> float:
        return self.wrapped_weapon.get_holster_time_secs()

    def get_ready_to_fire_time_secs(self) -> float:
        return self.wrapped_weapon.get_ready_to_fire_time_secs()

    def get_magazine_duration_seconds(self, tactical: int) -> float:
        return self.first_shot_time_seconds

    def get_magazine_capacity(self, tactical: int) -> int:
        return min(self.wrapped_weapon.get_magazine_capacity(tactical=tactical), 1)

    def get_shot_times_seconds(self, tactical: int) -> NDArray[np.float64]:
        shot_times_seconds = self.wrapped_weapon.get_shot_times_seconds(tactical=tactical)
        num_rounds = min(self.wrapped_weapon.get_magazine_capacity(tactical=tactical), 1)
        return shot_times_seconds[:num_rounds]

    def get_num_rounds_shot(self, time_seconds: float, tactical: int) -> int:
        nums_rounds = self.wrapped_weapon.get_num_rounds_shot(time_seconds=time_seconds,
                                                              tactical=tactical)
        return min(nums_rounds, 1)

    def get_nums_rounds_shot(self, times_seconds: NDArray[np.float64], tactical: int) \
            -> NDArray[np.integer[Any]]:
        nums_rounds = self.wrapped_weapon.get_nums_rounds_shot(times_seconds=times_seconds,
                                                               tactical=tactical)
        return nums_rounds.clip(max=1)

    def get_damage_per_round(self) -> float:
        return self.wrapped_weapon.get_damage_per_round()

    def get_accuracy_fraction(self, distance_meters: SupportedDistanceMeters) -> float:
        return self.wrapped_weapon.get_accuracy_fraction(distance_meters=distance_meters)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.wrapped_weapon)

    def __eq__(self, other):
        return (isinstance(other, _SingleShotLoadout) and
                self.wrapped_weapon == other.wrapped_weapon)

    def get_weapon(self) -> 'Weapon':
        return self.wrapped_weapon


class WeaponArchetype:
    BOOSTED_LOADER_FINDER = SingleTermFinder(BOOSTED_LOADER)

    def __init__(self,
                 name: str,
                 base_term: RequiredTerm,
                 suffix: Optional[Suffix],
                 weapon_class: WeaponClass,
                 dist_to_accuracy_mapping: Mapping[SupportedDistanceMeters, float],
                 damage_body: float,
                 rounds_per_minute: RoundsPerMinuteBase,
                 magazine_capacity: MagazineCapacityBase,
                 stock_dependant_stats: StockStatsBase,
                 spinup: RoundTiming, heat_based: bool,
                 associated_legend: Optional[Legend],
                 care_package: bool):
        check_type(Suffix, optional=True, suffix=suffix)
        check_bool(care_package=care_package)
        check_mapping(allowed_key_types=int,
                      allowed_value_types=(int, float),
                      dist_to_accuracy_mapping=dist_to_accuracy_mapping)

        self.name = name
        self.base_term = base_term
        self.suffix = suffix
        self.full_term: RequiredTerm = (base_term.append(*suffix.get_terms()) if suffix is not None
                                        else base_term)
        self.weapon_class = weapon_class
        self.dist_to_accuracy_mapping = dist_to_accuracy_mapping
        self.damage_body = damage_body
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.stock_dependant_stats = stock_dependant_stats
        self.spinup = spinup
        self.heat_based = heat_based
        self.associated_legend = associated_legend
        self.care_package = care_package
        self._has_boosted_loader = suffix is not None and suffix.has_term(BOOSTED_LOADER)

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
        dist_to_accuracy_mapping = self.dist_to_accuracy_mapping
        damage_body = self.damage_body
        spinup = self.spinup
        heat_based = self.heat_based
        has_boosted_loader = self._has_boosted_loader

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
                      dist_to_accuracy_mapping=dist_to_accuracy_mapping,
                      damage_body=damage_body,
                      holster_time_secs=holster_time_secs,
                      ready_to_fire_time_secs=ready_to_fire_time_secs,
                      rounds_per_minute=rounds_per_minute,
                      magazine_capacity=magazine_capacity,
                      spinup=spinup,
                      heat_based=heat_based,
                      tactical_reload_time_secs=tactical_reload_time_secs,
                      legend=legend,
                      has_boosted_loader=has_boosted_loader)

    def get_base_term(self) -> RequiredTerm:
        return self.base_term

    def get_suffix(self) -> Optional[Suffix]:
        return self.suffix

    def has_suffix_type(self, suffix_type: SuffixedArchetypeType) -> bool:
        return self.suffix is not None and suffix_type in self.suffix.get_types()

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
                raise ValueError('suffixed_archetype must have a suffix.')

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
        self._suffix_types = {suffix_type
                              for suffix, _ in suffixes_and_archetypes
                              if suffix is not None
                              for suffix_type in suffix.get_types()}

        self._is_care_package = base_archetype.is_care_package()

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
                                 exclude_flags: int = ExcludeFlag.NONE) -> \
            Generator[Weapon, None, None]:
        for _, archetype in self._suffixes_and_archetypes:
            if self._should_include(archetype, exclude_flags):
                yield archetype.get_best_weapon(legend=legend)

    @staticmethod
    def filter_loadouts(loadouts: Tuple[FullLoadout, ...], exclude_flags: int) -> \
            Generator[FullLoadout, None, None]:
        for loadout in loadouts:
            if all(WeaponArchetypes._should_include(weapon.get_archetype(),
                                                    exclude_flags=exclude_flags)
                   for weapon in loadout.get_weapons()):
                yield loadout

    @staticmethod
    def _should_include(archetype: WeaponArchetype, exclude_flags: int) -> bool:
        if archetype.is_care_package() and ExcludeFlag.CARE_PACKAGE.find(exclude_flags):
            return False

        suffix_type_to_flags: Mapping[SuffixedArchetypeType, Tuple[ExcludeFlag, ExcludeFlag]] = {
            SuffixedArchetypeType.HOPPED_UP: (ExcludeFlag.HOPPED_UP, ExcludeFlag.NON_HOPPED_UP),
            SuffixedArchetypeType.REVVED_UP: (ExcludeFlag.REVVED_UP, ExcludeFlag.NON_REVVED_UP),
            SuffixedArchetypeType.AKIMBO: (ExcludeFlag.AKIMBO, ExcludeFlag.NON_AKIMBO),
            SuffixedArchetypeType.RELIC: (ExcludeFlag.RELIC, ExcludeFlag.NONE),
        }

        for suffix_type, (exclude_it, exclude_not_it) in suffix_type_to_flags.items():
            has_suffix_type = archetype.has_suffix_type(suffix_type)
            if (exclude_it.find(exclude_flags) and has_suffix_type or
                    exclude_not_it.find(exclude_flags) and not has_suffix_type):
                return False

        return True

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


def calc_probability_of_n_trials_given_r_successes(n: float | NDArray[np.integer[Any]],
                                                   r: float | NDArray[np.integer[Any]],
                                                   p: float | NDArray[np.floating[Any]]) -> \
        NDArray[np.floating[Any]]:
    # From https://en.wikipedia.org/wiki/Negative_binomial_distribution
    return comb(n - 1, r - 1) * (p ** r) * ((1 - p) ** (n - r))
