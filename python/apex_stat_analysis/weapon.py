import abc
from enum import StrEnum
import math
from typing import Callable


class BaseMagazineCapacity:
    def __init__(self, base_capacity: int):
        self.base_capacity = base_capacity

    def get_capacities(self) -> dict[str, int]:
        return {'no mag': self.base_capacity}


class MagazineCapacity(BaseMagazineCapacity):
    def __init__(self,
                 base_capacity: int,
                 level_1_capacity: int,
                 level_2_capacity: int,
                 level_3_capacity: int):
        super().__init__(base_capacity=base_capacity)
        self.level_1_capacity = level_1_capacity
        self.level_2_capacity = level_2_capacity
        self.level_3_capacity = level_3_capacity

    def get_capacities(self) -> dict[str, int]:
        mags = super().get_capacities()
        mags.update({
            'level 1 mag': self.level_1_capacity,
            'level 2 mag': self.level_2_capacity,
            'level 3 mag': self.level_3_capacity
        })
        return mags


class BaseReloadTime:
    def __init__(self, reload_time_secs: float | None):
        self.reload_time_secs = reload_time_secs

    def get_reload_times_secs(self) -> dict[str, float | None]:
        return ({'no stock': self.reload_time_secs}
                if self.reload_time_secs is not None
                else {'no reload': None})


class ReloadTime(BaseReloadTime):
    def __init__(self,
                 base_reload_time_secs: float,
                 level_1_reload_time_secs: float,
                 level_2_reload_time_secs: float,
                 level_3_reload_time_secs: float):
        super().__init__(base_reload_time_secs)
        self.level_1_reload_time_secs = level_1_reload_time_secs
        self.level_2_reload_time_secs = level_2_reload_time_secs
        self.level_3_reload_time_secs = level_3_reload_time_secs

    def get_reload_times_secs(self) -> dict[str, float]:
        reload_times = super().get_reload_times_secs()
        reload_times.update({
            'level 1 stock': self.level_1_reload_time_secs,
            'level 2 stock': self.level_2_reload_time_secs,
            'level 3 stock': self.level_3_reload_time_secs
        })
        return reload_times


class BaseRoundsPerMinute:
    def __init__(self, base_rounds_per_minute: float):
        self.base_rounds_per_minute = base_rounds_per_minute

    def get_rounds_per_minutes(self) -> dict[str, float]:
        return {'no bolt': self.base_rounds_per_minute}


class RoundsPerMinute(BaseRoundsPerMinute):
    def __init__(self,
                 base_rounds_per_minute: float,
                 level_1_rounds_per_minute: float,
                 level_2_rounds_per_minute: float,
                 level_3_rounds_per_minute: float):
        super().__init__(base_rounds_per_minute=base_rounds_per_minute)
        self.level_1_rounds_per_minute = level_1_rounds_per_minute
        self.level_2_rounds_per_minute = level_2_rounds_per_minute
        self.level_3_rounds_per_minute = level_3_rounds_per_minute

    def get_rounds_per_minutes(self) -> dict[str, float]:
        rpms = super().get_rounds_per_minutes()
        rpms.update({
            'level 1 bolt': self.level_1_rounds_per_minute,
            'level 2 bolt': self.level_2_rounds_per_minute,
            'level 3 bolt': self.level_3_rounds_per_minute,
        })
        return rpms


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
        raise NotImplementedError()


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


class WeaponBase(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def get_archetype(self):
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

    def combine(self, sidearm: 'WeaponBase') -> 'WeaponBase':
        return CombinedWeapon(self, sidearm)

    def reload(self) -> 'WeaponBase':
        return ReloadingWeapon(self)


class ReloadingWeapon(WeaponBase):
    def __init__(self, reloading_weapon: 'WeaponBase'):
        super().__init__(reloading_weapon.get_name() + ' with reloads')
        self.reloading_weapon = reloading_weapon

    def get_archetype(self):
        return self.reloading_weapon.get_archetype()

    def get_deploy_time_secs(self) -> float:
        return self.reloading_weapon.get_deploy_time_secs()

    def get_instantaneous_damage(self, time_seconds: float, include_deployment=False) -> float:
        reload_time_seconds = self.reloading_weapon.get_tactical_reload_time_secs()
        if reload_time_seconds is None:
            return self.reloading_weapon.get_instantaneous_damage(time_seconds)

        mag_duration_seconds = self.reloading_weapon.get_magazine_duration_seconds(tactical=True)
        cycle_duration_seconds = mag_duration_seconds + reload_time_seconds
        rel_time_seconds = time_seconds % cycle_duration_seconds
        return (self.reloading_weapon.get_instantaneous_damage(rel_time_seconds)
                if rel_time_seconds < mag_duration_seconds
                else 0)

    def get_cumulative_damage(self, time_seconds: float, include_deployment=False) -> float:
        reload_time_seconds = self.reloading_weapon.get_tactical_reload_time_secs()
        if reload_time_seconds is None:
            return self.reloading_weapon.get_cumulative_damage(time_seconds)

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


class CombinedWeapon(WeaponBase):
    def __init__(self, main_weapon: WeaponBase, sidearm: WeaponBase):
        self.main_weapon = main_weapon
        self.sidearm = sidearm
        super().__init__(name=f'{main_weapon.name} with sidearm of {sidearm.name}')

    def get_deploy_time_secs(self) -> float:
        return self.main_weapon.get_deploy_time_secs()

    def get_archetype(self):
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
            include_deployment=False)
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

    def get_archetype(self):
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


class WeaponArchetype:
    def __init__(self,
                 name: str,
                 active: bool,
                 weapon_class: str,
                 damage_body: float,
                 deploy_time_secs: float,
                 rounds_per_minute: BaseRoundsPerMinute,
                 magazine_capacity: BaseMagazineCapacity,
                 tactical_reload_time: BaseReloadTime,
                 full_reload_time: BaseReloadTime,
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

    def get_name(self):
        return self.name

    def get_deploy_time_seconds(self):
        return self.deploy_time_secs

    def get_class(self):
        return self.weapon_class

    def __repr__(self):
        return self.name

    def get_base_weapons(self, reload: bool = False, sidearm: WeaponBase = None) -> \
            tuple[ConcreteWeapon]:
        if not self.active:
            return tuple()

        base_weapons: list[ConcreteWeapon] = []

        name = self.name
        weapon_class = self.weapon_class
        damage_body = self.damage_body
        deploy_time_secs = self.deploy_time_secs
        spinup = self.spinup

        rpms = self.rounds_per_minute.get_rounds_per_minutes()
        mags = self.magazine_capacity.get_capacities()
        tactical_reload_times = (self.tactical_reload_time.get_reload_times_secs()
                                 if reload
                                 else BaseReloadTime(None).get_reload_times_secs())
        for rpm_name, rpm in reversed(rpms.items()):
            rpm_name = (f' ({rpm_name})' if len(rpms) > 1 else '')
            for mag_name, mag in reversed(mags.items()):
                mag_name = (f' ({mag_name})' if len(mags) > 1 else '')
                for stock_name, tactical_reload_time in reversed(tactical_reload_times.items()):
                    stock_name = (f' ({stock_name})' if len(tactical_reload_times) > 1 else '')

                    full_name = f'{name}{rpm_name}{mag_name}{stock_name}'
                    base_weapon = ConcreteWeapon(archetype=self,
                                                 name=full_name,
                                                 weapon_class=weapon_class,
                                                 damage_body=damage_body,
                                                 deploy_time_secs=deploy_time_secs,
                                                 rounds_per_minute=rpm,
                                                 magazine_capacity=mag,
                                                 spinup=spinup,
                                                 tactical_reload_time_secs=tactical_reload_time)
                    if reload:
                        base_weapon = base_weapon.reload()
                    if sidearm is not None:
                        base_weapon = base_weapon.combine(sidearm)
                    base_weapons.append(base_weapon)

        return tuple(base_weapons)
