import abc
from enum import StrEnum
import logging
import math
import typing
from types import MappingProxyType
from typing import Callable

from apex_stat_analysis.speech.term_translator import ApexTranslator, ParsedAndFollower
from apex_stat_analysis.speech.terms import ApexTerms, ApexTermBase, ConcreteApexTerm, Words


def add_sidearm_and_reload(base_weapons: typing.Iterable['WeaponBase'],
                           sidearm: 'WeaponBase | None' = None,
                           reload: bool = False) -> tuple['WeaponBase']:
    result = base_weapons
    if sidearm is not None:
        result = tuple(weapon.combine_with_sidearm(sidearm) for weapon in result)
    if reload:
        result = tuple(weapon.reload() for weapon in result)
    if not isinstance(result, tuple):
        result = tuple(result)
    return result


T = typing.TypeVar('T')


class StatBase(abc.ABC, typing.Generic[T]):
    def __init__(self,
                 all_terms: ConcreteApexTerm | None | tuple[ConcreteApexTerm],
                 *all_values: T):
        if not isinstance(all_terms, tuple):
            all_terms: tuple[ConcreteApexTerm | None] = (all_terms,)
        else:
            assert all(term is not None for term in all_terms)
        assert len(all_terms) == len(all_values)
        assert len(all_terms) == len(set(all_terms))

        term_to_val_dict: MappingProxyType[ConcreteApexTerm | None, T] = MappingProxyType({
            term: time for term, time in zip(all_terms, all_values)
        })
        self._default_stats = MappingProxyType({all_terms[-1]: all_values[-1]})
        self._translator = ApexTranslator({term: val
                                           for term, val in term_to_val_dict.items()
                                           if term is not None})
        self._str_to_val_dict: MappingProxyType[str | None, T] = MappingProxyType({
            (str(term) if term is not None else None): val
            for term, val in term_to_val_dict.items()})

    def get_stats(self, words: Words | None = None) -> MappingProxyType[str | None, T]:
        if words is None:
            return self._str_to_val_dict

        terms = {str(parsed.get_term()): parsed.get_parsed()
                 for parsed in self._translator.translate_terms(words)}
        if len(terms) == 0:
            return self._default_stats
        return MappingProxyType(terms)


class MagazineCapacityBase(StatBase[int]):
    pass


class SingleMagazineCapacity(MagazineCapacityBase):
    def __init__(self, base_capacity: int):
        super().__init__(None, base_capacity)


class MagazineCapacity(MagazineCapacityBase):
    def __init__(self,
                 base_capacity: int,
                 level_1_capacity: int,
                 level_2_capacity: int,
                 level_3_capacity: int):
        super().__init__(ApexTerms.ALL_MAG_TERMS,
                         base_capacity,
                         level_1_capacity,
                         level_2_capacity,
                         level_3_capacity)


class ReloadTimeBase(StatBase[float | None]):
    pass


class SingleReloadTime(ReloadTimeBase):
    def __init__(self, reload_time_secs: float | None):
        super().__init__(None, reload_time_secs)


class ReloadTime(ReloadTimeBase):
    def __init__(self,
                 base_reload_time_secs: float,
                 level_1_reload_time_secs: float,
                 level_2_reload_time_secs: float,
                 level_3_reload_time_secs: float):
        super().__init__(ApexTerms.ALL_STOCK_TERMS,
                         base_reload_time_secs,
                         level_1_reload_time_secs,
                         level_2_reload_time_secs,
                         level_3_reload_time_secs)


class BaseRoundsPerMinute(StatBase[float]):
    pass


class SingleRoundsPerMinute(BaseRoundsPerMinute):
    def __init__(self, base_rounds_per_minute: float):
        super().__init__(None, base_rounds_per_minute)


class RoundsPerMinute(BaseRoundsPerMinute):
    def __init__(self,
                 base_rounds_per_minute: float,
                 level_1_rounds_per_minute: float,
                 level_2_rounds_per_minute: float,
                 level_3_rounds_per_minute: float):
        super().__init__(ApexTerms.ALL_BOLT_TERMS,
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
    def __init__(self, name: str):
        self.name = name

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

    def combine_with_sidearm(self, sidearm: 'WeaponBase') -> 'WeaponBase':
        return CombinedWeapon(self, sidearm)

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
        super().__init__(f'{reloading_weapon.get_name()} {ApexTerms.WITH_RELOAD_TERM}')
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
    def __init__(self, main_weapon: WeaponBase, sidearm: WeaponBase):
        self.main_weapon = main_weapon
        self.sidearm = sidearm
        super().__init__(f'{main_weapon.name} {ApexTerms.WITH_SIDEARM} {sidearm.name}')

    def get_deploy_time_secs(self) -> float:
        return self.main_weapon.get_deploy_time_secs()

    def get_archetype(self) -> 'WeaponArchetype':
        return self.main_weapon.get_archetype()

    def _get_damage(self,
                    time_seconds: float,
                    method: Callable[[WeaponBase, float, bool], float]) -> float:
        main_total_duration_seconds = self.main_weapon.get_magazine_duration_seconds()
        if time_seconds < main_total_duration_seconds:
            return method(self.main_weapon, time_seconds, False)

        time_seconds -= main_total_duration_seconds
        return method(self.sidearm, time_seconds, True)

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


class ConcreteWeapon(WeaponBase):
    def __init__(self,
                 archetype: 'WeaponArchetype',
                 name: str,
                 weapon_class: str,
                 damage_body: float,
                 deploy_time_secs: float,
                 rounds_per_minute: float,
                 magazine_capacity: int,
                 spinup: Spinup,
                 tactical_reload_time_secs: float | None):
        super().__init__(name=name)
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


class WeaponArchetype:
    _TAKEN_TERMS: set[ApexTermBase] = set()
    _ARCHETYPE_TERM_FINDER = ApexTranslator({
        term: term
        for term in ApexTerms.WEAPON_ARCHETYPE_TERMS})

    def __init__(self,
                 name: str,
                 active: bool,
                 weapon_class: str,
                 damage_body: float,
                 deploy_time_secs: float,
                 rounds_per_minute: SingleRoundsPerMinute,
                 magazine_capacity: SingleMagazineCapacity,
                 tactical_reload_time: SingleReloadTime,
                 full_reload_time: SingleReloadTime,
                 spinup: Spinup):
        self.name = name
        self.active = active
        self.weapon_class = weapon_class
        self.damage_body = damage_body
        self.deploy_time_secs = deploy_time_secs
        self.rounds_per_minute = rounds_per_minute
        self.magazine_capacity = magazine_capacity
        self.tactical_reload_time = tactical_reload_time
        self.full_reload_time = full_reload_time
        self.spinup = spinup

        # Figure out what term matches the weapon name.
        name_words = Words(name)
        possible_terms = tuple(map(
            ParsedAndFollower.get_parsed,
            WeaponArchetype._ARCHETYPE_TERM_FINDER.translate_terms(name_words)))
        if len(possible_terms) == 0:
            logging.warning(f'No term found for weapon archetype: {name}. Speech-to-text will not '
                            'work for it.')
            term = None
        elif len(possible_terms) > 1:
            term = possible_terms[0]
            logging.warning(f'More than one term for weapon archetype {name} found. We\'ll assume '
                            f'that {term} is the right one.')
        else:
            (term,) = possible_terms

        if term is not None:
            if term in WeaponArchetype._TAKEN_TERMS:
                logging.warning(f'Term {term} refers to another weapon. Can\'t use it.')
                term = None
            else:
                WeaponArchetype._TAKEN_TERMS.add(term)
        self.term = term
        self.base_weapons = tuple(self._get_base_weapons(
            rpms=self.rounds_per_minute.get_stats(),
            mags=self.magazine_capacity.get_stats(),
            tactical_reload_times=self.tactical_reload_time.get_stats()))

    def _get_base_weapons(self,
                          rpms: MappingProxyType[str | None, float],
                          mags: MappingProxyType[str | None, int],
                          tactical_reload_times: MappingProxyType[str | None, float]) -> \
            typing.Generator[ConcreteWeapon, None, None]:
        if not self.active:
            return

        name = self.name
        weapon_class = self.weapon_class
        damage_body = self.damage_body
        deploy_time_secs = self.deploy_time_secs
        spinup = self.spinup

        for rpm_idx, (rpm_name, rpm) in enumerate(reversed(rpms.items())):
            rpm_name = f' ({rpm_name})' if rpm_name is not None else ''

            for mag_idx, (mag_name, mag) in enumerate(reversed(mags.items())):
                mag_name = f' ({mag_name})' if mag_name is not None else ''

                for stock_idx, (stock_name, tactical_reload_time) in enumerate(reversed(
                        tactical_reload_times.items())):
                    stock_name = f' ({stock_name})' if stock_name is not None else ''

                    full_name = f'{name}{rpm_name}{mag_name}{stock_name}'
                    base_weapon: ConcreteWeapon = ConcreteWeapon(
                        archetype=self,
                        name=full_name,
                        weapon_class=weapon_class,
                        damage_body=damage_body,
                        deploy_time_secs=deploy_time_secs,
                        rounds_per_minute=rpm,
                        magazine_capacity=mag,
                        spinup=spinup,
                        tactical_reload_time_secs=tactical_reload_time)
                    yield base_weapon

    def get_term(self) -> ApexTermBase | None:
        return self.term

    def get_name(self):
        return self.name

    def get_deploy_time_seconds(self):
        return self.deploy_time_secs

    def get_class(self):
        return self.weapon_class

    def __repr__(self):
        return self.name

    def get_base_weapons(self, words: Words | None = None) -> \
            typing.Generator[ConcreteWeapon, None, None]:
        if words is None:
            for weapon in self.base_weapons:
                yield weapon
            return None

        for weapon in self._get_base_weapons(
                rpms=self.rounds_per_minute.get_stats(words),
                mags=self.magazine_capacity.get_stats(words),
                tactical_reload_times=self.tactical_reload_time.get_stats(words)):
            yield weapon
