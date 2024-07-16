import abc
import math
from enum import StrEnum
from types import MappingProxyType
from typing import Generator, Generic, Optional, Set, Tuple, TypeVar, Union

from apex_assistant.checker import (check_bool,
                                    check_equal_length,
                                    check_float,
                                    check_int,
                                    check_str,
                                    check_type,
                                    to_kwargs)
from apex_assistant.speech.apex_terms import (ALL_BOLT_TERMS,
                                              ALL_MAG_TERMS,
                                              ALL_STOCK_TERMS,
                                              BASE,
                                              LEVEL_TERMS,
                                              SWITCHING_TO_SIDEARM,
                                              WITH_RELOAD_OPT)
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import Translator


T = TypeVar('T')


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

    def get_all_stats(self) -> MappingProxyType[Optional[RequiredTerm], T]:
        return self._term_to_val_dict

    def translate_stats(self, words: Words) -> \
            Tuple[MappingProxyType[Optional[RequiredTerm], T], Words]:
        check_type(Words, words=words)
        translated_terms = tuple(self._translator.translate_terms(words))
        terms = {parsed.get_term(): parsed.get_value() for parsed in translated_terms}
        untranslated_words = Translator.get_untranslated_words(
            original_words=words,
            translated_terms=translated_terms)
        return MappingProxyType(terms), untranslated_words

    @staticmethod
    def translate_levels(words: Words) -> Set[int]:
        return set(term.get_value() for term in StatsBase._LEVEL_TRANSLATOR.translate_terms(words))

    def get_stats_for_levels(self, levels: set[int]) -> MappingProxyType[Optional[RequiredTerm], T]:
        check_int(min_value=0, **to_kwargs(levels=tuple(levels)))
        max_level = len(self._all_terms) - 1
        if len(levels) == 0:
            # Get the highest level by default.
            levels = {max_level}
        else:
            levels: set[int] = set(min(level, max_level) for level in levels)

        return MappingProxyType({self._all_terms[level]: self._all_values[level]
                                 for level in levels})


class MagazineCapacityBase(StatsBase[int]):
    pass


class SingleMagazineCapacity(MagazineCapacityBase):
    def __init__(self, base_capacity: int):
        check_int(min_val=1, base_capacity=base_capacity)
        super().__init__(None, base_capacity)


class MagazineCapacity(MagazineCapacityBase):
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


class ReloadTimeBase(StatsBase[float | None]):
    pass


class SingleReloadTime(ReloadTimeBase):
    def __init__(self, reload_time_secs: float | None):
        check_float(min_value=0,
                    min_is_exclusive=True,
                    optional=True,
                    reload_time_secs=reload_time_secs)
        super().__init__(None, reload_time_secs)


class ReloadTime(ReloadTimeBase):
    def __init__(self,
                 base_reload_time_secs: float,
                 level_1_reload_time_secs: float,
                 level_2_reload_time_secs: float,
                 level_3_reload_time_secs: float):
        check_float(min_value=0,
                    min_is_exclusive=True,
                    base_rounds_per_minute=base_reload_time_secs,
                    level_1_rounds_per_minute=level_1_reload_time_secs,
                    level_2_rounds_per_minute=level_2_reload_time_secs,
                    level_3_rounds_per_minute=level_3_reload_time_secs)
        super().__init__(ALL_STOCK_TERMS,
                         base_reload_time_secs,
                         level_1_reload_time_secs,
                         level_2_reload_time_secs,
                         level_3_reload_time_secs)


class BaseRoundsPerMinute(StatsBase[float]):
    pass


class SingleRoundsPerMinute(BaseRoundsPerMinute):
    def __init__(self, base_rounds_per_minute: float):
        check_float(base_rounds_per_minute=base_rounds_per_minute,
                    min_value=0,
                    min_is_exclusive=True)
        super().__init__(None, base_rounds_per_minute)


class RoundsPerMinute(BaseRoundsPerMinute):
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


class SpinupType(StrEnum):
    DEVOTION = 'devotion'
    HAVOC = 'havoc'
    NONE = 'none'


class Spinup(abc.ABC):
    @abc.abstractmethod
    def get_instantaneous_damage_per_second(self,
                                            base_weapon: 'ConcreteWeapon',
                                            time_seconds: float) -> float:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_cumulative_damage(self, base_weapon: 'ConcreteWeapon', time_seconds: float) -> float:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def get_magazine_duration_seconds(self,
                                      base_weapon: 'ConcreteWeapon',
                                      tactical: bool = False) -> float:
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError('Must implement.')

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError('Must implement.')


class SpinupDevotion(Spinup):
    def __init__(self, rounds_per_minute_initial: float, spinup_time_seconds: float):
        check_float(rounds_per_minute_initial=rounds_per_minute_initial,
                    min_value=0,
                    min_is_exclusive=True)
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)

        self._rounds_per_minute_initial = rounds_per_minute_initial
        self._spinup_time_seconds = spinup_time_seconds

    def get_damage_per_second_initial(self, base_weapon: 'ConcreteWeapon'):
        damage_per_minute_initial = (self._rounds_per_minute_initial *
                                     base_weapon.get_damage_per_round())
        damage_per_second_initial = damage_per_minute_initial / 60
        return damage_per_second_initial

    def get_magazine_duration_seconds(self,
                                      base_weapon: 'ConcreteWeapon',
                                      tactical: bool = False) -> float:
        damage_per_second = base_weapon.get_damage_per_second_body()
        damage_per_second_initial = self.get_damage_per_second_initial(base_weapon)
        spinup_time_seconds = self._spinup_time_seconds
        mag_capacity = base_weapon.get_magazine_capacity() - tactical
        damage_per_magazine = mag_capacity * base_weapon.get_damage_per_round()

        # I got this off of Wolfram Alpha: solved the cumulative damage formula (spun up but not
        # finished) for time.
        time_seconds = (damage_per_second * spinup_time_seconds -
                        spinup_time_seconds * damage_per_second_initial +
                        2 * damage_per_magazine) / (2 * damage_per_second)

        return time_seconds

    def get_instantaneous_damage_per_second(self,
                                            base_weapon: 'ConcreteWeapon',
                                            time_seconds: float) -> float:
        spinup_time_seconds = self._spinup_time_seconds
        damage_per_second = base_weapon.get_damage_per_second_body()
        damage_per_second_initial = self.get_damage_per_second_initial(base_weapon)
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)

        if time_seconds >= magazine_duration_seconds:
            # Finished.
            return 0
        elif time_seconds >= spinup_time_seconds:
            # Spun up all the way.
            return damage_per_second
        else:
            # Still spinning up.
            return ((damage_per_second * time_seconds +
                     damage_per_second_initial * (spinup_time_seconds - time_seconds)) /
                    spinup_time_seconds)

    def get_cumulative_damage(self, base_weapon: 'ConcreteWeapon', time_seconds: float) -> float:
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)
        spinup_time_seconds = self._spinup_time_seconds
        damage_per_second = base_weapon.get_damage_per_second_body()
        damage_per_second_initial = self.get_damage_per_second_initial(base_weapon)

        if magazine_duration_seconds <= time_seconds:
            # Finished.
            return base_weapon.get_magazine_capacity() * base_weapon.get_damage_per_round()
        elif time_seconds >= spinup_time_seconds:
            # Spun up all the way.
            return (damage_per_second * (time_seconds - spinup_time_seconds) +
                    (damage_per_second + damage_per_second_initial) / 2 * spinup_time_seconds)
        else:
            # Still spinning up.
            return (time_seconds *
                    ((damage_per_second * time_seconds +
                      damage_per_second_initial * (spinup_time_seconds - time_seconds)) /
                     spinup_time_seconds +
                     damage_per_second_initial) / 2)

    def __hash__(self):
        return hash(self.__class__) ^ hash((self._rounds_per_minute_initial,
                                            self._spinup_time_seconds))

    def __eq__(self, other):
        return (isinstance(other, SpinupDevotion) and
                self._rounds_per_minute_initial == other._rounds_per_minute_initial and
                self._spinup_time_seconds == other._spinup_time_seconds)


class SpinupHavoc(Spinup):
    def __init__(self, spinup_time_seconds: float):
        check_float(spinup_time_seconds=spinup_time_seconds, min_value=0)
        self._spinup_time_seconds = spinup_time_seconds

    def get_magazine_duration_seconds(self,
                                      base_weapon: 'ConcreteWeapon',
                                      tactical: bool = False) -> float:
        magazine_capacity = base_weapon.get_magazine_capacity() - tactical
        rounds_per_minute = base_weapon.get_rounds_per_minute()
        rounds_per_second = rounds_per_minute / 60
        magazine_duration_seconds = magazine_capacity / rounds_per_second
        return magazine_duration_seconds

    def get_instantaneous_damage_per_second(self,
                                            base_weapon: 'ConcreteWeapon',
                                            time_seconds: float) -> float:
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)
        spinup_time_seconds = self._spinup_time_seconds
        damage_per_second = base_weapon.get_damage_per_second_body()
        if magazine_duration_seconds + spinup_time_seconds > time_seconds >= spinup_time_seconds:
            return damage_per_second
        else:
            return 0

    def get_cumulative_damage(self, base_weapon: 'ConcreteWeapon', time_seconds: float) -> float:
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)
        spinup_time_seconds = self._spinup_time_seconds
        damage_per_second = base_weapon.get_damage_per_second_body()

        if time_seconds >= spinup_time_seconds:
            return damage_per_second * min(time_seconds - spinup_time_seconds,
                                           magazine_duration_seconds)
        else:
            return 0

    def __hash__(self):
        return hash(self.__class__) ^ hash(self._spinup_time_seconds)

    def __eq__(self, other):
        return (isinstance(other, SpinupHavoc) and
                other._spinup_time_seconds == self._spinup_time_seconds)


class SpinupNone(Spinup):
    def get_magazine_duration_seconds(self,
                                      base_weapon: 'ConcreteWeapon',
                                      tactical: bool = False) -> float:
        check_type(ConcreteWeapon, base_weapon=base_weapon)
        check_bool(tactical=tactical)

        magazine_capacity = base_weapon.get_magazine_capacity() - tactical
        rounds_per_minute = base_weapon.get_rounds_per_minute()
        rounds_per_second = rounds_per_minute / 60
        magazine_duration_seconds = magazine_capacity / rounds_per_second
        return magazine_duration_seconds

    def get_instantaneous_damage_per_second(self,
                                            base_weapon: 'ConcreteWeapon',
                                            time_seconds: float) -> float:
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)
        damage_per_second = base_weapon.get_damage_per_second_body()
        if time_seconds < magazine_duration_seconds:
            return damage_per_second
        return 0

    def get_cumulative_damage(self, base_weapon: 'ConcreteWeapon', time_seconds: float) -> float:
        magazine_duration_seconds = self.get_magazine_duration_seconds(base_weapon)
        damage_per_second = base_weapon.get_damage_per_second_body()
        return damage_per_second * min(time_seconds, magazine_duration_seconds)

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, SpinupNone)


class WeaponBase(abc.ABC):
    def __init__(self, name: str, term: RequiredTerm):
        check_str(allow_blank=False, name=name)
        check_type(RequiredTerm, term=term)
        self.name = name
        self.term = term

    def get_term(self) -> RequiredTerm:
        return self.term

    @abc.abstractmethod
    def get_archetype(self) -> 'WeaponArchetype':
        raise NotImplementedError()

    @abc.abstractmethod
    def get_deploy_time_secs(self) -> float:
        raise NotImplementedError()

    def get_name(self):
        return self.name

    def __repr__(self):
        return self.name

    @abc.abstractmethod
    def get_instantaneous_damage(self, time_seconds: float, include_deployment=False) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cumulative_damage(self, time_seconds: float, include_deployment=False) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_tactical_reload_time_secs(self) -> float | None:
        raise NotImplementedError()

    def reload(self) -> 'WeaponBase':
        return ReloadingWeapon(self) if self.get_tactical_reload_time_secs() is not None else self

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()


class ReloadingWeapon(WeaponBase):
    def __init__(self, reloading_weapon: 'WeaponBase'):
        super().__init__(f'{reloading_weapon.get_name()} {WITH_RELOAD_OPT}',
                         reloading_weapon.get_term().combine(WITH_RELOAD_OPT))
        assert reloading_weapon.get_tactical_reload_time_secs() is not None
        self.reloading_weapon = reloading_weapon
        self.reload_time_secs = float(reloading_weapon.get_tactical_reload_time_secs())

    def get_archetype(self) -> 'WeaponArchetype':
        return self.reloading_weapon.get_archetype()

    def get_deploy_time_secs(self) -> float:
        return self.reloading_weapon.get_deploy_time_secs()

    def get_instantaneous_damage(self, time_seconds: float, include_deployment=False) -> float:
        reload_time_seconds = self.reload_time_secs
        mag_duration_seconds = self.reloading_weapon.get_magazine_duration_seconds(tactical=True)
        cycle_duration_seconds = mag_duration_seconds + reload_time_seconds
        rel_time_seconds = time_seconds % cycle_duration_seconds
        return (self.reloading_weapon.get_instantaneous_damage(rel_time_seconds)
                if rel_time_seconds < mag_duration_seconds
                else 0)

    def get_cumulative_damage(self, time_seconds: float, include_deployment=False) -> float:
        dt = self.get_deploy_time_secs() if include_deployment else 0
        time_seconds -= dt
        if time_seconds <= 0:
            return 0

        reload_time_seconds = self.reload_time_secs
        mag_duration_seconds = self.reloading_weapon.get_magazine_duration_seconds(tactical=True)
        cycle_duration_seconds = mag_duration_seconds + reload_time_seconds
        num_completed_cycles, rel_time_seconds = divmod(time_seconds, cycle_duration_seconds)
        if rel_time_seconds >= mag_duration_seconds:
            cum_damage = 0
            num_completed_cycles += 1
        else:
            cum_damage = self.reloading_weapon.get_cumulative_damage(rel_time_seconds)
        cum_damage += (num_completed_cycles *
                       self.reloading_weapon.get_cumulative_damage(mag_duration_seconds))
        return cum_damage

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return math.inf

    def get_tactical_reload_time_secs(self) -> float | None:
        return self.reloading_weapon.get_tactical_reload_time_secs()

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.reloading_weapon)

    def __eq__(self, other):
        return (isinstance(other, ReloadingWeapon) and
                other.reloading_weapon == self.reloading_weapon)


class CombinedWeapon(WeaponBase):
    def __init__(self, main_weapon: 'ConcreteWeapon', sidearm: 'ConcreteWeapon'):
        assert isinstance(main_weapon, ConcreteWeapon)
        assert isinstance(sidearm, ConcreteWeapon)
        self.main_weapon = main_weapon
        self.sidearm = sidearm
        super().__init__(f'{main_weapon.name} {SWITCHING_TO_SIDEARM} {sidearm.name}',
                         main_weapon.get_term().combine(SWITCHING_TO_SIDEARM, sidearm.get_term()))

    def get_deploy_time_secs(self) -> float:
        return self.main_weapon.get_deploy_time_secs()

    def get_archetype(self) -> 'WeaponArchetype':
        return self.main_weapon.get_archetype()

    def get_instantaneous_damage(self, time_seconds: float, include_deployment=False) -> float:
        main_total_duration_seconds = self.main_weapon.get_magazine_duration_seconds()
        if time_seconds < main_total_duration_seconds:
            return self.main_weapon.get_instantaneous_damage(time_seconds, include_deployment=False)

        time_seconds -= main_total_duration_seconds
        return self.sidearm.get_instantaneous_damage(time_seconds, include_deployment=True)

    def get_cumulative_damage(self, time_seconds: float, include_deployment=False) -> float:
        main_total_duration_seconds = self.main_weapon.get_magazine_duration_seconds()
        cum_damage = self.main_weapon.get_cumulative_damage(
            min(time_seconds, main_total_duration_seconds),
            include_deployment=include_deployment)
        if time_seconds >= main_total_duration_seconds:
            time_seconds -= main_total_duration_seconds
            cum_damage += self.sidearm.get_cumulative_damage(time_seconds,
                                                             include_deployment=True)
        return cum_damage

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return (self.main_weapon.get_magazine_duration_seconds(tactical=False) +
                self.sidearm.get_deploy_time_secs() +
                self.sidearm.get_magazine_duration_seconds(tactical=tactical))

    def get_tactical_reload_time_secs(self) -> float | None:
        return self.sidearm.get_tactical_reload_time_secs()

    def __hash__(self):
        return hash(self.__class__) ^ hash((self.main_weapon, self.sidearm))

    def __eq__(self, other):
        return (isinstance(other, CombinedWeapon) and
                other.main_weapon == self.main_weapon and other.sidearm == self.sidearm)

    def get_main_weapon(self) -> 'ConcreteWeapon':
        return self.main_weapon

    def get_sidearm(self) -> 'ConcreteWeapon':
        return self.sidearm


class ConcreteWeapon(WeaponBase):
    def __init__(self,
                 archetype: 'WeaponArchetype',
                 name: str,
                 term: RequiredTerm,
                 weapon_class: str,
                 damage_body: float,
                 deploy_time_secs: float,
                 rounds_per_minute: float,
                 magazine_capacity: int,
                 spinup: Spinup,
                 tactical_reload_time_secs: float | None):
        super().__init__(name=name, term=term)
        self.deploy_time_secs = deploy_time_secs
        self.archetype = archetype
        self.weapon_class = weapon_class
        self.damage_body = damage_body
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.spinup = spinup
        self.tactical_reload_time_secs = tactical_reload_time_secs

        damage_per_minute_body = damage_body * rounds_per_minute
        self._damage_per_second_body = damage_per_minute_body / 60

    def get_archetype(self) -> 'WeaponArchetype':
        return self.archetype

    def get_deploy_time_secs(self) -> float:
        return self.deploy_time_secs

    def get_tactical_reload_time_secs(self) -> float | None:
        return self.tactical_reload_time_secs

    def get_instantaneous_damage(self, time_seconds: float, include_deployment=False) -> float:
        dt = self.deploy_time_secs if include_deployment else 0
        if time_seconds < dt:
            return 0
        return self.spinup.get_instantaneous_damage_per_second(self, time_seconds - dt)

    def get_cumulative_damage(self, time_seconds: float, include_deployment=False) -> float:
        dt = self.deploy_time_secs if include_deployment else 0
        if time_seconds < dt:
            return 0
        return self.spinup.get_cumulative_damage(self, time_seconds - dt)

    def get_magazine_duration_seconds(self, tactical: bool = False) -> float:
        return self.spinup.get_magazine_duration_seconds(self, tactical=tactical)

    def get_damage_per_second_body(self) -> float:
        return self._damage_per_second_body

    def get_damage_per_round(self) -> float:
        return self.damage_body

    def get_magazine_capacity(self):
        return self.magazine_capacity

    def get_rounds_per_minute(self):
        return self.rounds_per_minute

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.name)

    def __eq__(self, other):
        return (isinstance(other, ConcreteWeapon) and
                self.deploy_time_secs == other.deploy_time_secs and
                self.archetype == other.archetype and
                self.weapon_class == other.weapon_class and
                self.damage_body == other.damage_body and
                self.rounds_per_minute == other.rounds_per_minute and
                self.magazine_capacity == other.magazine_capacity and
                self.spinup == other.spinup and
                self.tactical_reload_time_secs == other.tactical_reload_time_secs)

    def combine_with_sidearm(self, sidearm: 'ConcreteWeapon') -> 'CombinedWeapon':
        return CombinedWeapon(self, sidearm)


class WeaponArchetype:
    def __init__(self,
                 name: str,
                 term: RequiredTerm,
                 weapon_class: str,
                 damage_body: float,
                 deploy_time_secs: float,
                 rounds_per_minute: SingleRoundsPerMinute,
                 magazine_capacity: SingleMagazineCapacity,
                 tactical_reload_time: SingleReloadTime,
                 full_reload_time: SingleReloadTime,
                 spinup: Spinup):
        self.name = name
        self.term = term
        self.weapon_class = weapon_class
        self.damage_body = damage_body
        self.deploy_time_secs = deploy_time_secs
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.tactical_reload_time = tactical_reload_time
        self.full_reload_time = full_reload_time
        self.spinup = spinup
        self.base_weapons = tuple(self._get_base_weapons(
            rpms=self.rounds_per_minute.get_all_stats(),
            mags=self.magazine_capacity.get_all_stats(),
            tactical_reload_times=self.tactical_reload_time.get_all_stats()))

    def _get_base_weapons(
            self,
            rpms: MappingProxyType[Optional[RequiredTerm], float],
            mags: MappingProxyType[Optional[RequiredTerm], int],
            tactical_reload_times: MappingProxyType[Optional[RequiredTerm], float]
    ) -> Generator[ConcreteWeapon, None, None]:
        name = self.name
        term = self.term
        weapon_class = self.weapon_class
        damage_body = self.damage_body
        deploy_time_secs = self.deploy_time_secs
        spinup = self.spinup

        for rpm_idx, (rpm_term, rpm) in enumerate(reversed(rpms.items())):
            rpm_name = f' ({rpm_term})' if rpm_term is not None else ''

            for mag_idx, (mag_term, mag) in enumerate(reversed(mags.items())):
                mag_name = f' ({mag_term})' if mag_term is not None else ''

                for stock_idx, (stock_term, tactical_reload_time) in enumerate(reversed(
                        tactical_reload_times.items())):
                    stock_name = f' ({stock_term})' if stock_term is not None else ''

                    full_name = f'{name}{rpm_name}{mag_name}{stock_name}'
                    more_terms: Tuple[RequiredTerm, ...] = tuple(
                        _term
                        for _term in (rpm_term, mag_term, stock_term)
                        if _term is not None)
                    if len(more_terms) > 0:
                        full_term = term.combine(*more_terms)
                    else:
                        full_term = term
                    base_weapon: ConcreteWeapon = ConcreteWeapon(
                        archetype=self,
                        name=full_name,
                        term=full_term,
                        weapon_class=weapon_class,
                        damage_body=damage_body,
                        deploy_time_secs=deploy_time_secs,
                        rounds_per_minute=rpm,
                        magazine_capacity=mag,
                        spinup=spinup,
                        tactical_reload_time_secs=tactical_reload_time)
                    yield base_weapon

    def get_term(self) -> RequiredTerm:
        return self.term

    def get_name(self) -> str:
        return self.name

    def get_deploy_time_seconds(self) -> float:
        return self.deploy_time_secs

    def get_class(self) -> str:
        return self.weapon_class

    def __repr__(self) -> str:
        return self.name

    def get_best_match(self, words: Words) -> ConcreteWeapon:
        # We assume that it'll give us something.
        return next(self.get_base_weapons(words))

    def get_base_weapons(self, words: Words | None = None) -> Generator[ConcreteWeapon, None, None]:
        if words is None:
            for weapon in self.base_weapons:
                yield weapon
            return None

        rpms, words = self.rounds_per_minute.translate_stats(words)
        mags, words = self.magazine_capacity.translate_stats(words)
        tactical_reload_times, words = self.tactical_reload_time.translate_stats(words)

        # TODO: Consider allowing the word "base" or "level N" before weapon archetype instead of
        #  requiring it to be after.
        default_levels = StatsBase.translate_levels(words)
        if len(rpms) == 0:
            rpms = self.rounds_per_minute.get_stats_for_levels(default_levels)
        if len(mags) == 0:
            mags = self.magazine_capacity.get_stats_for_levels(default_levels)
        if len(tactical_reload_times) == 0:
            tactical_reload_times = self.tactical_reload_time.get_stats_for_levels(default_levels)

        for weapon in self._get_base_weapons(
                rpms=rpms,
                mags=mags,
                tactical_reload_times=tactical_reload_times):
            yield weapon
